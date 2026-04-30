"""Global Graph Assembler for SWAGGER.

This helper consumes per-frame/local graphs (such as the ones produced by
``WaypointGraphGenerator``) and incrementally builds a stitched, world-frame
graph stored entirely as GPU tensors. It handles:

* merging nearby free-space nodes (ignores skeleton nodes)
* persisting/freeing nodes using a retention factor (0 → only current frame,
  1 → keep everything forever)
* maintaining a probabilistic heatmap of boundary observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set
import math

import cv2
import numpy as np
import networkx as nx
import torch
import cupy as cp
import string

@dataclass
class GlobalGraphGenerator:
    """Incrementally build a stitched global graph.

    The global graph is stored entirely as GPU tensors. Local graphs arrive as
    networkx graphs and are converted to tensors during the merge step.

    Args:
        merge_distance: Maximum world-distance between two free-space nodes
            to consider them the "same" node when merging (meters).
        retention_factor: Controls how long unseen nodes stay around.
            0.0 → drop nodes immediately if not observed this frame.
            1.0 → keep nodes forever.
        boundary_increment: How much to increment the obstacle probability
            whenever a boundary node is re-observed.
        boundary_decay: How quickly unseen obstacle probabilities decay.
    """

    merge_distance: float = 0.05
    retention_factor: float = 0.5
    boundary_increment: float = 0.2
    boundary_decay: float = 0.9
    boundary_cell_size: float = 0.05
    max_connections: int = 15
    pruning_frequency: int = 6
    max_candidate_edge_distance: float = 0.5
    max_candidate_edge_search_distance: float = 10

    occ_grid: np.ndarray = None            # 255 = free, 0 = occupied
    occ_resolution: float = 0.04
    occ_center: Tuple[float, float] = (0.0, 0.0)   # world coords of grid center (user-provided)

    _colliding_edges: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    # --- Global graph stored as tensors (on GPU when available) ---
    _global_pos: torch.Tensor = field(init=False)            # (N, 3) node positions
    _global_ids: torch.Tensor = field(init=False)            # (N,)   node integer IDs
    _global_node_types: torch.Tensor = field(init=False)     # (N,)   node type IDs (0=unknown, 1=free_space, 2=frontier)
    _global_edge_ids: torch.Tensor = field(init=False)       # (E, 2) pairs of node IDs
    _global_edge_weights: torch.Tensor = field(init=False)   # (E,)   edge weights

    ## NOTE: THROUGHOUT THE CODE ENSURE THAT i-th NODE IN _global_pos CORRESPONDS TO i-th NODE ID in _global_ids and _global_node_types
    # SIMILARLY _global_edge_ids and _global_edge_weights.
    #
    # Node IDs are stable, non-contiguous, and monotonically assigned via
    # _next_node_id.  IDs are never reused after removal.  Edge endpoints
    # store node IDs (not array indices).

    _next_node_id: int = field(default=0, init=False)
    _node_usage: Dict[int, float] = field(default_factory=dict, init=False)
    _boundary_probs: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)

    # Node type encoding (matches graph_msg_utils)
    _NODE_TYPE_ENCODE: Dict[str, int] = field(default_factory=lambda: {"": 0, "free_space": 1, "frontier": 2}, init=False, repr=False)

    def __post_init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._global_pos = torch.empty((0, 3), dtype=torch.float32, device=device)
        self._global_ids = torch.empty((0,), dtype=torch.long, device=device)
        self._global_node_types = torch.empty((0,), dtype=torch.int32, device=device)
        self._global_edge_ids = torch.empty((0, 2), dtype=torch.long, device=device)
        self._global_edge_weights = torch.empty((0,), dtype=torch.float32, device=device)

    def _occ_origin(self):
        if self.occ_grid is None:
            return (0.0, 0.0)
        
        h, w = self.occ_grid.shape
        ox = self.occ_center[0] + (w * self.occ_resolution) / 2.0
        oy = self.occ_center[1] - (h * self.occ_resolution) / 2.0
        return (ox, oy)        

    # ------------------------------------------------------------------
    # Primitive graph mutation API
    # All node/edge mutations should go through these methods so that
    # the parallel-array invariant is maintained in one place.
    # ------------------------------------------------------------------

    def _id_to_mask(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask over ``_global_ids`` for the requested IDs.
        NOTE: If node_ids has Node IDs which do not exist in _global_ids then they would just be discarded. 
        torch.sum(lookup[self._global_ids.long()]) might not be equal to node_ids.shape[0]

        Args:
            node_ids: (K,) long tensor of node IDs to locate. **Auto-assigns the tensor to the GPU**

        Returns:
            (N,) bool tensor on **GPU** — True where ``_global_ids`` matches.
        """
        device = self._global_pos.device
        node_ids = node_ids.to(device=device)
        # Size lookup only to _global_ids.max() — queried IDs beyond that can't match
        max_existing = self._global_ids.max() + 1
        lookup = torch.zeros(max_existing, dtype=torch.bool, device=device)
        lookup[node_ids[node_ids < max_existing].long()] = True
        return lookup[self._global_ids.long()]

    # ── Single-node operations ──

    def add_node(self, position: torch.Tensor, node_type: int = 1) -> int:
        """Add one node and return its new ID.

        Args:
            position:  (3,) float32 tensor. Can be **CPU or GPU** (moved
                       automatically to match internal storage).
            node_type: 0=unknown, 1=free_space (default), 2=frontier.

        Returns:
            int — the assigned node ID.  Internal tensors live on **GPU**.
        """
        nid = self._next_node_id
        device = self._global_pos.device

        self._global_pos = torch.cat(
            [self._global_pos, position.to(device).unsqueeze(0)], dim=0
        )
        self._global_ids = torch.cat(
            [self._global_ids, torch.tensor([nid], dtype=torch.long, device=device)]
        )
        self._global_node_types = torch.cat(
            [self._global_node_types, torch.tensor([node_type], dtype=torch.int32, device=device)]
        )
        self._next_node_id += 1
        return nid

    def add_nodes(
        self,
        positions: torch.Tensor,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add multiple nodes and return their new IDs.

        Args:
            positions:  (K, 3) float32 tensor — world xyz.  Can be **CPU or
                        GPU** (moved automatically).
            node_types: (K,) int32 tensor — type per node.  Can be **CPU or
                        GPU**.  If ``None``, every node defaults to
                        ``1`` (free_space).

        Returns:
            (K,) long tensor of newly assigned node IDs on **GPU**.
        """
        K = positions.shape[0]
        if K == 0:
            return torch.empty((0,), dtype=torch.long, device=self._global_pos.device)

        device = self._global_pos.device

        if node_types is None:
            node_types = torch.ones(K, dtype=torch.int32, device=device)

        new_ids = torch.arange(
            self._next_node_id, self._next_node_id + K,
            dtype=torch.long, device=device,
        )

        self._global_pos = torch.cat([self._global_pos, positions.to(device)], dim=0)
        self._global_ids = torch.cat([self._global_ids, new_ids], dim=0)
        self._global_node_types = torch.cat(
            [self._global_node_types, node_types.to(device).int()], dim=0
        )
        self._next_node_id += K
        return new_ids

    def remove_node(self, node_id: int) -> None:
        """Remove a single node and all its incident edges.

        Args:
            node_id: integer ID of the node to remove.

        Operates entirely on **GPU** tensors.
        """
        self.remove_nodes(
            torch.tensor([node_id], dtype=torch.long, device=self._global_pos.device)
        )

    def remove_nodes(self, node_ids: torch.Tensor) -> None:
        """Remove multiple nodes and every edge that references them.

        Args:
            node_ids: (K,) long tensor of node IDs to remove. Can be **CPU or
                      GPU** (moved automatically).

        Operates entirely on **GPU** tensors.
        """
        if node_ids.numel() == 0:
            return

        device = self._global_pos.device
        node_ids = node_ids.to(device)

        # ── Remove nodes ──
        keep_mask = ~self._id_to_mask(node_ids)
        self._global_pos = self._global_pos[keep_mask]
        self._global_ids = self._global_ids[keep_mask]
        self._global_node_types = self._global_node_types[keep_mask]

        # ── Remove incident edges ──
        if self._global_edge_ids.shape[0] > 0:
            ids_set = node_ids.unsqueeze(0)                 # (1, K)
            src = self._global_edge_ids[:, 0].unsqueeze(1)  # (E, 1)
            tgt = self._global_edge_ids[:, 1].unsqueeze(1)  # (E, 1)
            src_hit = (src == ids_set).any(dim=1)            # (E,)
            tgt_hit = (tgt == ids_set).any(dim=1)            # (E,)
            edge_keep = ~(src_hit | tgt_hit)
            self._global_edge_ids = self._global_edge_ids[edge_keep]
            self._global_edge_weights = self._global_edge_weights[edge_keep]

    # ── Edge operations ──

    def add_edge(self, src_id: int, tgt_id: int, weight: float) -> None:
        """Add a single edge between two existing nodes.

        Args:
            src_id: source node ID.
            tgt_id: target node ID.
            weight: edge weight (e.g. Euclidean distance).

        Silently drops the edge if either endpoint does not exist.
        Operates on **GPU** tensors.
        """
        device = self._global_edge_ids.device
        self.add_edges(
            torch.tensor([[src_id, tgt_id]], dtype=torch.long, device=device),
            torch.tensor([weight], dtype=torch.float32, device=device),
        )

    def add_edges(self, edge_pairs: torch.Tensor, weights: torch.Tensor) -> None:
        """Add multiple edges, dropping any that reference non-existent nodes.

        Args:
            edge_pairs: (K, 2) long tensor — ``[src_id, tgt_id]`` per row.
                        Can be **CPU or GPU** (moved automatically).
            weights:    (K,) float32 tensor — one weight per edge.
                        Can be **CPU or GPU** (moved automatically).

        Internal tensors are updated on **GPU**.
        """
        if edge_pairs.shape[0] == 0:
            return

        device = self._global_edge_ids.device
        edge_pairs = edge_pairs.to(device)
        weights = weights.to(device)

        # O(max_id) lookup instead of O(K*N) broadcast
        existing = self._global_ids  # (N,)
        max_existing = existing.max() + 1
        lookup = torch.zeros(max_existing, dtype=torch.bool, device=device)
        lookup[existing] = True
        # Any edge endpoint >= max_existing can't be a valid node
        src_in_range = edge_pairs[:, 0] < max_existing
        tgt_in_range = edge_pairs[:, 1] < max_existing
        src_valid = src_in_range & lookup[edge_pairs[:, 0].clamp(max=max_existing - 1)]
        tgt_valid = tgt_in_range & lookup[edge_pairs[:, 1].clamp(max=max_existing - 1)]
        valid = src_valid & tgt_valid

        if not valid.any():
            return

        new_pairs = edge_pairs[valid]
        new_weights = weights[valid]

        # ── Dedup: drop edges that already exist ──
        if self._global_edge_ids.shape[0] > 0:
            existing_sorted = torch.sort(self._global_edge_ids, dim=1)[0]  # (E, 2)
            new_sorted = torch.sort(new_pairs, dim=1)[0]                   # (K', 2)
            # Encode pairs as single ints for fast comparison
            max_id_enc = max(existing_sorted.max(), new_sorted.max()) + 1
            existing_keys = existing_sorted[:, 0] * max_id_enc + existing_sorted[:, 1]  # (E,)
            new_keys = new_sorted[:, 0] * max_id_enc + new_sorted[:, 1]                 # (K',)
            # O(max_key) lookup
            key_lookup = torch.zeros(existing_keys.max() + 1, dtype=torch.bool, device=device)
            key_lookup[existing_keys] = True
            in_range = new_keys < key_lookup.shape[0]
            already_exists = in_range & key_lookup[new_keys.clamp(max=key_lookup.shape[0] - 1)]
            novel = ~already_exists
            new_pairs = new_pairs[novel]
            new_weights = new_weights[novel]

        if new_pairs.shape[0] == 0:
            return

        # ── Degree check: drop edges that would exceed max_connections ──
        # NOTE: degree is computed assuming all new edges are added at once, so mutual contention is not resolved greedily.
        all_edges = torch.cat([self._global_edge_ids, new_pairs], dim=0)
        all_nodes = torch.cat([all_edges[:, 0], all_edges[:, 1]])
        max_node = all_nodes.max() + 1
        degree = torch.zeros(max_node, dtype=torch.long, device=device)
        degree.scatter_add_(0, all_nodes, torch.ones_like(all_nodes))
        # For each new edge, check both endpoints' resulting degree
        src_degree_ok = degree[new_pairs[:, 0]] <= self.max_connections
        tgt_degree_ok = degree[new_pairs[:, 1]] <= self.max_connections
        under_limit = src_degree_ok & tgt_degree_ok
        new_pairs = new_pairs[under_limit]
        new_weights = new_weights[under_limit]

        if new_pairs.shape[0] > 0:
            self._global_edge_ids = torch.cat(
                [self._global_edge_ids, new_pairs], dim=0
            )
            self._global_edge_weights = torch.cat(
                [self._global_edge_weights, new_weights], dim=0
            )

    def remove_edges(self, edge_pairs: torch.Tensor) -> None:
        """Remove edges matching the given ``(src, tgt)`` pairs (order-independent).

        Args:
            edge_pairs: (K, 2) long tensor of edges to remove.  Can be **CPU
                        or GPU** (moved automatically).  Direction is ignored
                        — ``(a, b)`` matches ``(b, a)``.

        Operates on **GPU** tensors.
        """
        if edge_pairs.shape[0] == 0 or self._global_edge_ids.shape[0] == 0:
            return

        device = self._global_edge_ids.device
        edge_pairs = edge_pairs.to(device)

        # Normalize both to (min, max) so direction doesn't matter
        existing_sorted = torch.sort(self._global_edge_ids, dim=1)[0]  # (E, 2)
        remove_sorted = torch.sort(edge_pairs, dim=1)[0]               # (K, 2)

        # (E, 1, 2) == (1, K, 2) → all(dim=2) → any(dim=1) → (E,)
        match = (
            existing_sorted.unsqueeze(1) == remove_sorted.unsqueeze(0)
        ).all(dim=2).any(dim=1)

        keep = ~match
        self._global_edge_ids = self._global_edge_ids[keep]
        self._global_edge_weights = self._global_edge_weights[keep]
    
    def return_closest_node_ids(self, query_node_positions: torch.Tensor, max_distance_graph: float = 10.0) -> torch.Tensor:
        """
        Returns the node_ids in self._global_ids which are closest to the position given in query_node_positions.

        Args:
            query_node_positions: (N, 3) float32 tensor stored as [x, y, z] can be **CPU or GPU**.
            max_distance_graph: If a query's nearest node is at least this far (meters),
                the returned ID for that query is -1.

        Returns:
            closest_node_ids: (N,) long tensor; entry is -1 when no node is within
                max_distance_graph.

        Note:
            torch.cdist produces an (N, M) float32 matrix (M = num global nodes).
            For large N this can exhaust VRAM, so the query is chunked based on
            available CUDA memory (1 GB safety buffer, capped at 8 GB budget).
        """
        device = self._global_pos.device

        if self._global_pos.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        if query_node_positions.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)

        query_node_positions = query_node_positions.to(device)
        N = query_node_positions.shape[0]
        M = self._global_pos.shape[0]
        bytes_per_row = M * 4  # cdist output is float32

        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget_bytes = max(0, free_bytes - 1 * 1024**3)   # leave 1 GB headroom
            budget_bytes = min(budget_bytes, 8 * 1024**3)     # hard cap at 8 GB
        else:
            budget_bytes = 8 * 1024**3
        chunk_size = max(1, budget_bytes // bytes_per_row)

        if N <= chunk_size:
            dists = torch.cdist(query_node_positions, self._global_pos)  # (N, M)
            min_dist, index = torch.min(dists, dim=1)
        else:
            min_dist_parts, index_parts = [], []
            for start in range(0, N, chunk_size):
                chunk = query_node_positions[start:start + chunk_size]
                d = torch.cdist(chunk, self._global_pos)                 # (chunk, M)
                md, idx = torch.min(d, dim=1)
                min_dist_parts.append(md)
                index_parts.append(idx)
            min_dist = torch.cat(min_dist_parts)
            index = torch.cat(index_parts)

        far_off_points = min_dist >= max_distance_graph
        closest_ids = self._global_ids[index]
        closest_ids[far_off_points] = -1
        return closest_ids

    def edge_valid_kernel(self, width, height):
        """
        CuPy ElementWiseKernel to check if the line segment (edge) between
        two points [x0, y0] and [x1, y1] is valid (i.e., contains no obstacles).
        Uses Bresenham's line algorithm for efficient grid traversal.
        """
        _edge_valid_kernel = cp.ElementwiseKernel(
            # U = unsigned char (bool)
            in_params="raw U edges, raw U map",
            out_params="raw U valid",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int x, int y) {
                    // The map is flat: (y + x * height). Assuming (x,y) are in (width, height) format
                    // but the kernel uses x for width index and y for height index.
                    // Assuming row-major storage: idx = y * width + x
                    // Or column-major (often for grid maps in robotics): idx = x * height + y
                    // Let's stick to the common: idx = y * width + x for (row, col) = (y, x)
                    return y * ${width} + x;
                }
                __device__ bool is_inside_map(int x, int y) {
                    return (x >= 0 && y >= 0 && x<${width} && y<${height});
                }
                """
            ).substitute(width=width, height=height),
            operation=string.Template(
                """
                // Input: edges[i * 4 + 0, 1, 2, 3] = [x0, y0, x1, y1]
                // Input: map[y * width + x] = 1 (clear/free) or 0 (occupied)
                // Output: valid[i] = 1 (valid/clear) or 0 (invalid/obstacle)
                int x0 = edges[i * 4 + 0];
                int y0 = edges[i * 4 + 1];
                int x1 = edges[i * 4 + 2];
                int y1 = edges[i * 4 + 3];
                // Bresenham's algorithm setup
                int dx = abs(x1 - x0);
                int sx = x0 < x1 ? 1 : -1;
                int dy = -abs(y1 - y0);
                int sy = y0 < y1 ? 1 : -1;
                int error = dx + dy;
                bool is_clear = true;
                // Iterate over all cells along line
                while (1){
                    // 1. Check if the current cell (x0, y0) is inside and clear
                    if (is_inside_map(x0, y0)){
                        int idx = get_map_idx(x0, y0);
                        // map is 1=free, 0=occupied. We check if it is NOT free (i.e., occupied)
                        if (!map[idx]){
                            is_clear = false;
                            break;
                        }
                    }
                    // 2. Termination condition
                    if (x0 == x1 && y0 == y1){
                        break;
                    }
                    // 3. Compute next grid cell index in line (Bresenham step)
                    int e2 = 2 * error;
                    if (e2 >= dy){ // x-step
                        if(x0 == x1) break; // Re-check termination
                        error += dy;
                        x0 += sx;
                    }
                    if (e2 <= dx){ // y-step
                        if (y0 == y1) break; // Re-check termination
                        error += dx;
                        y0 += sy;
                    }
                }
                // Mark the validity
                valid[i] = is_clear ? 1 : 0;
                """
            ).substitute(height=height, width=width),
            name="edge_valid_kernel",
        )
        return _edge_valid_kernel

    def batch_collision_check(self, p1_xy, p2_xy, p1_ids, p2_ids):

        if self.occ_grid is None:
            return torch.ones(len(p1_xy), dtype=torch.bool, device=p1_xy.device)

        device = p1_xy.device

        # -------------------------------
        # 1. WORLD → GRID (VECTORIZED)
        # -------------------------------
        ox, oy = self._occ_origin()
        res = self.occ_resolution

        wx0 = p1_xy[:, 0]
        wy0 = p1_xy[:, 1]
        wx1 = p2_xy[:, 0]
        wy1 = p2_xy[:, 1]

        # NOTE: world -> grid conversion
        gx0 = ((-wx0 + ox) / res).long()
        gy0 = ((+wy0 - oy) / res).long()
        gx1 = ((-wx1 + ox) / res).long()
        gy1 = ((+wy1 - oy) / res).long()

        edges_torch = torch.stack([gx0, gy0, gx1, gy1], dim=1)

        # -------------------------------
        # 2. TORCH → CUPY TRANSFER
        # -------------------------------
        ## TODO: bug here, if the point in consideration is outside the grid of the cost map would lead to problems. 
        # the kernel would identify the part of the edge inside the grid. BUT NEED TO CHECK THIS.
        ## TODO: MAINTAIN A COSTMAP-LARGER THAN THE OEN BEING USED TO CHECK FOR COLLISIONS
        # NOT SURE HOW TO ACCOUNT FOR STATE OBSERVATION ERRORS. 

        edges_cp = cp.asarray(edges_torch.to(torch.uint8).contiguous())
        grid_cp  = cp.asarray((self.occ_grid == 255).astype(np.uint8).flatten())

        H, W = self.occ_grid.shape

        # -------------------------------
        # 3. RUN CUDA KERNEL
        # -------------------------------
        valid_cp = cp.zeros(len(edges_cp), dtype=cp.uint8)

        kernel = self.edge_valid_kernel(W, H)
        kernel(edges_cp, grid_cp, valid_cp, size=len(edges_cp))

        # -------------------------------
        # 4. CUPY → TORCH MASK
        # -------------------------------
        valid_np = cp.asnumpy(valid_cp)
        valid_torch = torch.from_numpy(valid_np.astype(np.bool_)).to(device)

        return valid_torch
    
    def quantize(world_xy, resolution=0.04):
        """
        Convert world coordinates to quantized grid cell key.
        """
        xq = int(world_xy[0] // resolution)
        yq = int(world_xy[1] // resolution)
        return (xq, yq)



    def add_local_graph(self, local_graph: nx.Graph, occ_center_x, occ_center_y, occ_grid, resolution) -> None:
        """Merge a local graph into the persistent global graph with optimized vectorization."""

        ## TODO: NEED GENERAL RULE ON HOW TO INPUT OCCUPANCY GRID MAP. THE CODE BELOW IS DUE TO OUR ELEVATION MAPPING OUTPUT.
        occ_grid = np.rot90(occ_grid, 2)  # To account for the occ grid convention as per GridMap
        
        self.occ_center = (occ_center_x, occ_center_y)
        self.occ_grid = occ_grid
        self.occ_resolution = resolution

        if not 0.0 <= self.retention_factor <= 1.0:
            raise ValueError("retention_factor must be between 0 and 1")

        device = self._global_pos.device

        global_pos = self._global_pos
        global_ids_tensor = self._global_ids

        """
        ############# LOCAL NODE MERGING TO GLOBAL START ##################
        """
        merge_start = torch.cuda.Event(enable_timing=True)
        merge_end = torch.cuda.Event(enable_timing=True)
        merge_start.record()

        ## NOTE: local_ids: [old(self._next_node_id) : old(self._next_node_id) + number of nodes added]
        local_ids, local_pos, self._next_node_id = self.tensor_merge_local_nodes(
            local_graph,
            global_pos,
            global_ids_tensor,
            self._next_node_id,
            self.merge_distance,
            device=device
        )

        merge_end.record()
        torch.cuda.synchronize()
        local_merge_ms = merge_start.elapsed_time(merge_end)

        if local_ids.numel() == 0:
            return

        """
        ############# LOCAL NODE MERGING TO GLOBAL END ##################
        """

        """
        ############# CANDIDATE EDGE SELECTION START ##################
        """

        cand_start = torch.cuda.Event(enable_timing=True)
        cand_end = torch.cuda.Event(enable_timing=True)
        cand_start.record()

        # Re-read global state after merge (new nodes may have been appended from the local graph)
        global_pos = self._global_pos
        global_ids_tensor = self._global_ids

        # Compute centroid and filter candidates (all on GPU)
        centroid = local_pos.mean(dim=0)

        # Mask has the binary tensor for global nodes which are within the max edge search distance of the centroid of local points. 
        mask = torch.norm(global_pos - centroid, dim=1) < self.max_candidate_edge_search_distance

        candidate_pos = global_pos[mask]
        candidate_ids_tensor = global_ids_tensor[mask]

        if len(candidate_pos) == 0:
            return

        dists = torch.cdist(local_pos, candidate_pos, p=2)

        # Find connections within threshold
        valid_mask = dists < self.max_candidate_edge_distance
        local_idx, cand_idx = torch.nonzero(valid_mask, as_tuple=True)
        # Edges are between local_idx[i] and cand_idx[i]

        if len(local_idx) == 0:
            return

        local_nodes_temp = local_ids[local_idx]                             # ID of the local_nodes forming edges
        global_nodes_temp = candidate_ids_tensor[cand_idx]                  # ID of the global_nodes forming edges
        no_self_loop_mask = local_nodes_temp != global_nodes_temp           # prevents edges between new added nodes in global graph from the local graph

        local_idx = local_idx[no_self_loop_mask]
        cand_idx = cand_idx[no_self_loop_mask]

        local_nodes = local_ids[local_idx]
        global_nodes = candidate_ids_tensor[cand_idx]
        weights = dists[local_idx, cand_idx]

        cand_end.record()
        torch.cuda.synchronize()
        cand_select_ms = cand_start.elapsed_time(cand_end)

        """
        ############# CANDIDATE EDGE SELECTION END ##################
        """



        """
        ############# COLLISION FILTER START ##################
        """


        # PyTorch CUDA event (your wrapper)
        coll_start = torch.cuda.Event(enable_timing=True)
        coll_end = torch.cuda.Event(enable_timing=True)
        coll_start.record()

        # ------------------- ADDED CUPY TIMING --------------------
        cupy_start = cp.cuda.Event()
        cupy_end = cp.cuda.Event()
        cupy_start.record()
        # -----------------------------------------------------------

        p1 = local_pos[local_idx]
        p2 = candidate_pos[cand_idx]

        collision_free_mask = self.batch_collision_check(
            p1, p2,
            local_nodes,
            global_nodes
        )

        # ------------------- END CUPY TIMING -----------------------
        cupy_end.record()
        cupy_end.synchronize()
        cupy_kernel_ms = cp.cuda.get_elapsed_time(cupy_start, cupy_end)
        # -----------------------------------------------------------

        local_nodes = local_nodes[collision_free_mask]
        global_nodes = global_nodes[collision_free_mask]
        weights = weights[collision_free_mask]

        coll_end.record()
        torch.cuda.synchronize()
        collision_ms = coll_start.elapsed_time(coll_end)


        """
        ############# COLLISION FILTER END ##################
        """


        """
        ############# ADDING EDGES TO THE GRAPH START ##################
        """

        add_start = torch.cuda.Event(enable_timing=True)
        add_end = torch.cuda.Event(enable_timing=True)
        add_start.record()

        if len(local_nodes) > 0:
            new_edges = torch.stack([local_nodes, global_nodes], dim=1)  # (K, 2)
            self._global_edge_ids = torch.cat([self._global_edge_ids, new_edges], dim=0)
            self._global_edge_weights = torch.cat([self._global_edge_weights, weights], dim=0)
            self._dedup_edges()

        add_end.record()
        torch.cuda.synchronize()
        add_edges_ms = add_start.elapsed_time(add_end)

        """
        ############# ADDING EDGES TO THE GRAPH END ##################
        """

        self._prune_redundant_edges()


    def add_local_graph_gpu(
        self,
        local_pos: torch.Tensor,           # (N, 3) float32 CUDA — world XYZ
        local_node_types: torch.Tensor,    # (N,)   int32  CUDA — 1=free_space
        occ_center_x: float,
        occ_center_y: float,
        occ_grid,                          # numpy occupancy grid (same convention as add_local_graph)
        resolution: float,
    ) -> None:
        """Tensor-native equivalent of add_local_graph.

        Accepts node positions and types directly as CUDA tensors (the exact
        output of WaypointGraphGeneratorGPU.build_graph_from_grid_map) instead
        of a NetworkX graph, skipping the per-node Python extraction loop.

        All other behaviour — merging, edge candidate selection, collision
        filtering, deduplication, pruning — is identical to add_local_graph.
        """
        occ_grid = np.rot90(occ_grid, 2)
        self.occ_center = (occ_center_x, occ_center_y)
        self.occ_grid = occ_grid
        self.occ_resolution = resolution

        if not 0.0 <= self.retention_factor <= 1.0:
            raise ValueError("retention_factor must be between 0 and 1")

        device = self._global_pos.device

        local_ids, local_id_positions, self._next_node_id = self.tensor_merge_local_nodes_gpu(
            local_pos,
            local_node_types,
            self._global_pos,
            self._global_ids,
            self._next_node_id,
            self.merge_distance,
            device=device,
        )

        if local_ids.numel() == 0:
            return

        # Re-read global state after merge (new nodes may have been appended).
        global_pos = self._global_pos
        global_ids_tensor = self._global_ids

        centroid = local_id_positions.mean(dim=0)
        mask = torch.norm(global_pos - centroid, dim=1) < self.max_candidate_edge_search_distance
        candidate_pos = global_pos[mask]
        candidate_ids_tensor = global_ids_tensor[mask]

        if len(candidate_pos) == 0:
            return

        dists = torch.cdist(local_id_positions, candidate_pos, p=2)
        valid_mask = dists < self.max_candidate_edge_distance
        local_idx, cand_idx = torch.nonzero(valid_mask, as_tuple=True)

        if len(local_idx) == 0:
            return

        local_nodes_temp = local_ids[local_idx]
        global_nodes_temp = candidate_ids_tensor[cand_idx]
        no_self_loop_mask = local_nodes_temp != global_nodes_temp

        local_idx  = local_idx[no_self_loop_mask]
        cand_idx   = cand_idx[no_self_loop_mask]
        local_nodes  = local_ids[local_idx]
        global_nodes = candidate_ids_tensor[cand_idx]
        weights      = dists[local_idx, cand_idx]

        p1 = local_id_positions[local_idx]
        p2 = candidate_pos[cand_idx]

        collision_free_mask = self.batch_collision_check(p1, p2, local_nodes, global_nodes)

        local_nodes  = local_nodes[collision_free_mask]
        global_nodes = global_nodes[collision_free_mask]
        weights      = weights[collision_free_mask]

        if len(local_nodes) > 0:
            new_edges = torch.stack([local_nodes, global_nodes], dim=1)
            self._global_edge_ids    = torch.cat([self._global_edge_ids, new_edges], dim=0)
            self._global_edge_weights = torch.cat([self._global_edge_weights, weights], dim=0)
            self._dedup_edges()

        self._prune_redundant_edges()


    def tensor_merge_local_nodes(self, local_graph: nx.Graph, global_worlds: torch.Tensor, global_ids_tensor: torch.Tensor, next_node_id, merge_distance, device="cuda"):
        """
        Merge local graph nodes into global graph using fully vectorized GPU operations.
        Args:
            local_graph: The local NetworkX graph to merge (on CPU)
            global_worlds: Tensor of existing global node positions. (on GPU)
            global_ids_tensor: Tensor of existing global node IDs. (on GPU)
            next_node_id: Next available unique node ID. (int)
            merge_distance: Maximum distance for treating two nodes as the same. (int)
            device: Torch device used for tensor allocation.

        Returns (local_ids, local_id_positions, updated_next_id)
          - local_ids:          (N,) long tensor of global node IDs assigned to each local node (on GPU)
          - local_id_positions: (N, 3) positions (global pos for merged, local pos for new)     (on GPU)
          - updated_next_id:    int, next available node ID                                     (int)
        """

        # -------------------------
        # Extract local nodes
        # -------------------------
        local_nodes = list(local_graph.nodes())
        if len(local_nodes) == 0:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, 3), dtype=torch.float32, device=device), next_node_id

        encode = self._NODE_TYPE_ENCODE
        local_worlds = torch.tensor(
            [local_graph.nodes[n]["world"] for n in local_nodes],
            dtype=torch.float32,
            device=device
        )  # (N, D)

        local_types = torch.tensor(
            [encode.get(local_graph.nodes[n].get("node_type", "free_space"), 1) for n in local_nodes],
            dtype=torch.int32,
            device=device
        )  # (N,)

        N = local_worlds.shape[0]

        # -------------------------
        # CASE 1: Empty global graph
        # -------------------------
        if global_worlds.numel() == 0:
            new_ids = torch.arange(next_node_id, next_node_id + N, device=device, dtype=torch.long)
            final_ids = new_ids
            new_local_mask = torch.ones(N, dtype=torch.bool, device=device)
            updated_next_id = next_node_id + N
            local_id_positions = local_worlds
        else:
            # -------------------------
            # Compute pairwise distances
            # -------------------------
            dists = torch.cdist(local_worlds, global_worlds)  # (N, M)

            # -------------------------
            # Find closest global for each local
            # -------------------------
            # min_idx here is the 
            min_dists, min_idx = torch.min(dists, dim=1)
            merge_mask = min_dists < merge_distance    # (N,)
            merged_ids = global_ids_tensor[min_idx]    # (N,)

            # -------------------------
            # Assign new IDs
            # -------------------------
            new_local_mask = ~merge_mask
            num_new = new_local_mask.sum()
            new_ids = torch.arange(next_node_id, next_node_id + num_new, device=device, dtype=torch.long)

            final_ids = merged_ids.clone()
            final_ids[new_local_mask] = new_ids
            updated_next_id = next_node_id + num_new

            # Build positions: merged nodes use global position, new nodes use local position
            local_id_positions = local_worlds.clone()
            local_id_positions[~new_local_mask] = global_worlds[min_idx[~new_local_mask]]

            # Update node types for merged nodes (local observation takes precedence)
            if merge_mask.any():
                self._global_node_types[min_idx[merge_mask]] = local_types[merge_mask]

        # -------------------------
        # Append new nodes to global tensor storage
        # -------------------------
        if new_local_mask.any():
            new_positions = local_worlds[new_local_mask]
            new_types = local_types[new_local_mask]
            self._global_pos = torch.cat([self._global_pos, new_positions], dim=0)
            self._global_ids = torch.cat([self._global_ids, new_ids], dim=0)
            self._global_node_types = torch.cat([self._global_node_types, new_types], dim=0)

        return final_ids, local_id_positions, updated_next_id


    def tensor_merge_local_nodes_gpu(
        self,
        local_worlds: torch.Tensor,       # (N, 3) float32 CUDA — world XYZ
        local_types: torch.Tensor,        # (N,)   int32  CUDA — node type IDs
        global_worlds: torch.Tensor,
        global_ids_tensor: torch.Tensor,
        next_node_id: int,
        merge_distance: float,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Tensor-native equivalent of tensor_merge_local_nodes.

        Identical merge logic but accepts positions and types as CUDA tensors
        directly, avoiding the per-node Python extraction from a NetworkX graph.

        Returns:
            local_ids:          (N,) long tensor — global ID assigned to each local node.
            local_id_positions: (N, 3) float32  — global pos for merged nodes, local pos for new.
            updated_next_id:    int — next available node ID.
        """
        N = local_worlds.shape[0]

        if N == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0, 3), dtype=torch.float32, device=device),
                next_node_id,
            )

        if global_worlds.numel() == 0:
            new_ids = torch.arange(next_node_id, next_node_id + N, device=device, dtype=torch.long)
            final_ids      = new_ids
            new_local_mask = torch.ones(N, dtype=torch.bool, device=device)
            updated_next_id = next_node_id + N
            local_id_positions = local_worlds
        else:
            dists = torch.cdist(local_worlds, global_worlds)   # (N, M)
            min_dists, min_idx = torch.min(dists, dim=1)

            merge_mask  = min_dists < merge_distance
            merged_ids  = global_ids_tensor[min_idx]

            new_local_mask  = ~merge_mask
            num_new         = int(new_local_mask.sum().item())
            new_ids         = torch.arange(next_node_id, next_node_id + num_new, device=device, dtype=torch.long)

            final_ids = merged_ids.clone()
            final_ids[new_local_mask] = new_ids
            updated_next_id = next_node_id + num_new

            # Merged nodes snap to the existing global position.
            local_id_positions = local_worlds.clone()
            local_id_positions[~new_local_mask] = global_worlds[min_idx[~new_local_mask]]

            # Local observation overrides the stored node type for merged nodes.
            if merge_mask.any():
                self._global_node_types[min_idx[merge_mask]] = local_types[merge_mask]

        if new_local_mask.any():
            self._global_pos        = torch.cat([self._global_pos,        local_worlds[new_local_mask]], dim=0)
            self._global_ids        = torch.cat([self._global_ids,        new_ids],                      dim=0)
            self._global_node_types = torch.cat([self._global_node_types, local_types[new_local_mask]],  dim=0)

        return final_ids, local_id_positions, updated_next_id


    def _dedup_edges(self) -> None:
        """Remove duplicate edges, keeping the one with the smallest weight."""
        if self._global_edge_ids.shape[0] == 0:
            return

        device = self._global_edge_ids.device

        # Normalize direction: smaller ID first in each pair
        sorted_pairs = torch.sort(self._global_edge_ids, dim=1)[0]  # (E, 2)

        # Encode pairs as unique keys for grouping
        unique_pairs, inverse = torch.unique(sorted_pairs, dim=0, return_inverse=True)

        # For each unique edge, keep minimum weight
        min_weights = torch.full((unique_pairs.shape[0],), float('inf'), dtype=torch.float32, device=device)
        min_weights.scatter_reduce_(0, inverse, self._global_edge_weights, reduce='amin')

        self._global_edge_ids = unique_pairs
        self._global_edge_weights = min_weights

    def _prune_redundant_edges(self) -> None:
        """Limit node degree by keeping only the closest neighbors.

        Fully vectorized on GPU — no Python loops over nodes.
        For each node, ranks its incident edges by weight and marks
        edges beyond max_connections for removal.
        """
        max_degree = self.max_connections

        # Only prune every N frames to reduce overhead
        if not hasattr(self, '_prune_counter'):
            self._prune_counter = 0
        self._prune_counter += 1

        if self._prune_counter % self.pruning_frequency != 0:
            return

        E = self._global_edge_ids.shape[0]
        if E == 0:
            return

        device = self._global_edge_ids.device
        edges = self._global_edge_ids   # (E, 2)
        weights = self._global_edge_weights  # (E,)

        # Consider both directions: edge i appears for both endpoints
        all_nodes = torch.cat([edges[:, 0], edges[:, 1]])       # (2E,)
        all_weights = weights.repeat(2)                          # (2E,)
        all_edge_idx = torch.arange(E, device=device).repeat(2)  # (2E,)

        # Map node IDs to compact 0..N-1 indices
        unique_nodes, node_idx = torch.unique(all_nodes, return_inverse=True)  # node_idx: (2E,)
        N = unique_nodes.shape[0]

        # Compute degree per node
        degree = torch.zeros(N, dtype=torch.long, device=device)
        degree.scatter_add_(0, node_idx, torch.ones(2 * E, dtype=torch.long, device=device))

        # Quick exit: if no node exceeds max_degree, nothing to prune
        if (degree <= max_degree).all():
            return

        # For each (node, edge) pair, compute rank of the edge by weight within that node.
        # Strategy: sort all entries by (node_idx, weight), then compute per-node rank.
        sort_keys = node_idx.float() * (all_weights.max() + 1.0) + all_weights
        sorted_order = torch.argsort(sort_keys)

        sorted_node_idx = node_idx[sorted_order]

        # Per-node rank: count how many entries with the same node_idx came before
        ones = torch.ones(2 * E, dtype=torch.long, device=device)
        cumcount = torch.zeros(2 * E, dtype=torch.long, device=device)
        # Running count per node via scatter: shift by 1 to get 0-based rank
        running = torch.zeros(N, dtype=torch.long, device=device)
        # Vectorized cumcount: for each position in sorted order, rank = how many
        # of the same node appeared before it
        node_start = torch.zeros(N, dtype=torch.long, device=device)
        node_start.scatter_add_(0, sorted_node_idx, ones)
        # Compute exclusive cumsum per group using the sorted order
        group_offsets = torch.zeros(2 * E, dtype=torch.long, device=device)
        # The rank within each node group is just position - first_position_of_that_node
        # Use cumsum trick: after sorting by (node, weight), consecutive same-node entries
        # get ranks 0, 1, 2, ...
        boundaries = torch.ones(2 * E, dtype=torch.long, device=device)
        if 2 * E > 1:
            same_as_prev = sorted_node_idx[1:] == sorted_node_idx[:-1]
            boundaries[1:] = same_as_prev.long()
        # cumsum within groups: reset at boundaries
        # Simpler approach: cumcount per group = cumsum of ones, reset at group change
        group_change = torch.ones(2 * E, dtype=torch.long, device=device)
        group_change[0] = 1
        if 2 * E > 1:
            group_change[1:] = (~same_as_prev).long()
        group_ids = torch.cumsum(group_change, dim=0) - 1  # 0-based group id for each sorted entry
        pos_in_sort = torch.arange(2 * E, device=device)
        group_starts = torch.zeros(group_ids[-1].item() + 1, dtype=torch.long, device=device)
        group_starts.scatter_(0, group_ids, pos_in_sort)  # first occurrence per group
        rank_in_node = pos_in_sort - group_starts[group_ids]  # 0-based rank within group

        # Mark edges where rank >= max_degree for removal (from either endpoint)
        exceeds = rank_in_node >= max_degree
        edge_indices_to_remove = all_edge_idx[sorted_order[exceeds]]

        if edge_indices_to_remove.shape[0] == 0:
            return

        edges_to_keep = torch.ones(E, dtype=torch.bool, device=device)
        edges_to_keep[edge_indices_to_remove] = False

        self._global_edge_ids = self._global_edge_ids[edges_to_keep]
        self._global_edge_weights = self._global_edge_weights[edges_to_keep]

    def _decay_boundaries(self, seen: Set[Tuple[int, int]]) -> None:
        for key in list(self._boundary_probs.keys()):
            if key in seen:
                continue
            self._boundary_probs[key] *= self.boundary_decay
            if self._boundary_probs[key] < 1e-3:
                del self._boundary_probs[key]

    def _quantize(self, world_xy: Tuple[float, float]) -> Tuple[int, int]:
        cell = self.boundary_cell_size
        return (int(math.floor(world_xy[0] / cell)), int(math.floor(world_xy[1] / cell)))

    def _cell_center(self, key: Tuple[int, int]) -> Tuple[float, float]:
        cell = self.boundary_cell_size
        return (key[0] * cell + cell / 2.0, key[1] * cell + cell / 2.0)

    

    def get_global_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the global graph as raw GPU tensors.

        Returns:
            (global_pos, global_ids, global_node_types, global_edge_ids, global_edge_weights)
            - global_pos:          (N, 3)  float32 node positions
            - global_ids:          (N,)    long    node integer IDs
            - global_node_types:   (N,)    int32   node type IDs (0=unknown, 1=free_space, 2=frontier)
            - global_edge_ids:     (E, 2)  long    edge endpoint pairs
            - global_edge_weights: (E,)    float32 edge weights
        """
        return (
            self._global_pos,
            self._global_ids,
            self._global_node_types,
            self._global_edge_ids,
            self._global_edge_weights,
        )

    def boundary_probabilities(self) -> Dict[Tuple[int, int], float]:
        """Return the current obstacle probability map."""
        return dict(self._boundary_probs)

    def debug_visualize(self, path: str, scale: float = 100.0) -> None:
        """Render a simple 2D visualization of the global graph."""

        num_nodes = self._global_ids.shape[0]
        if num_nodes == 0:
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(path, blank)
            return

        pos_cpu = self._global_pos.cpu().numpy()  # (N, 3)
        xs = pos_cpu[:, 0]
        ys = pos_cpu[:, 1]

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        width = max(64, int((max_x - min_x) * scale) + 64)
        height = max(64, int((max_y - min_y) * scale) + 64)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        def to_px(wx: float, wy: float) -> Tuple[int, int]:
            x = int((wx - min_x) * scale) + 32
            y = int((max_y - wy) * scale) + 32
            return x, y

        # Draw boundary probability shading (probabilities updated via exponential decay)
        if self._boundary_probs:
            overlay = canvas.copy()
            for key, prob in self._boundary_probs.items():
                wx, wy = self._cell_center(key)
                x, y = to_px(wx, wy)
                radius = max(2, int(self.boundary_cell_size * scale * 0.8))
                clamped = max(0.0, min(1.0, prob))
                color_val = int(clamped * 255)
                color = (0, color_val, 255 - color_val)
                cv2.circle(overlay, (x, y), radius, color, -1)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

        # Draw edges
        if self._global_edge_ids.shape[0] > 0:
            # Build ID → index map for position lookup
            ids_cpu = self._global_ids.cpu().numpy()
            id_to_idx = {int(nid): i for i, nid in enumerate(ids_cpu)}

            edge_ids_cpu = self._global_edge_ids.cpu().numpy()
            for u, v in edge_ids_cpu:
                idx_u = id_to_idx.get(int(u))
                idx_v = id_to_idx.get(int(v))
                if idx_u is None or idx_v is None:
                    continue
                x1, y1 = to_px(float(pos_cpu[idx_u, 0]), float(pos_cpu[idx_u, 1]))
                x2, y2 = to_px(float(pos_cpu[idx_v, 0]), float(pos_cpu[idx_v, 1]))
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Draw nodes
        for i in range(num_nodes):
            x, y = to_px(float(pos_cpu[i, 0]), float(pos_cpu[i, 1]))
            color = (255, 0, 0)  # all nodes drawn as "new"
            cv2.circle(canvas, (x, y), 3, color, -1)

        cv2.imwrite(path, canvas)