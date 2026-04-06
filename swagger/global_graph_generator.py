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
    _global_pos: torch.Tensor = field(init=False)           # (N, 3) node positions
    _global_node_ids: torch.Tensor = field(init=False)            # (N,)   node integer IDs
    _global_node_types: torch.Tensor = field(init=False)     # (N,)   node type IDs (0=unknown, 1=free_space, 2=frontier)
    _global_edge_ids: torch.Tensor = field(init=False)       # (E, 2) pairs of node IDs
    _global_edge_weights: torch.Tensor = field(init=False)   # (E,)   edge weights

    _next_node_id: int = field(default=0, init=False)
    _node_usage: Dict[int, float] = field(default_factory=dict, init=False)
    _boundary_probs: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)

    # Node type encoding (matches graph_msg_utils)
    _NODE_TYPE_ENCODE: Dict[str, int] = field(default_factory=lambda: {"": 0, "free_space": 1, "frontier": 2}, init=False, repr=False)

    def __post_init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._global_pos = torch.empty((0, 3), dtype=torch.float32, device=device)
        self._global_node_ids = torch.empty((0,), dtype=torch.long, device=device)
        self._global_node_types = torch.empty((0,), dtype=torch.int32, device=device)
        self._global_edge_ids = torch.empty((0, 2), dtype=torch.long, device=device)
        self._global_edge_weights = torch.empty((0,), dtype=torch.float32, device=device)

    def _add_node(self, position: torch.Tensor, node_type: int=1) -> int:
        """
        Adds a single node using the position (3,) and the specified node_type.
        Returns the assigned node_id. 
        """
        device = self._global_pos.device
        position = position.to(device)
        node_id = self._next_node_id

        self._global_pos = torch.cat([self._global_pos, position.unsqueeze(0)], dim=0)
        self._global_node_ids = torch.cat([self._global_node_ids, torch.tensor([node_id], dtype=torch.long, device=device)])
        self._global_node_types = torch.cat([self._global_node_types, torch.tensor([node_type], dtype=torch.long, device=device)])
        self._next_node_id += 1 

        return node_id
    
    def _add_nodes(self, positions: torch.Tensor, node_types: torch.tensor = None) -> torch.tensor:
        """
        Adding multiple nodes. Returns tensors of the assigned nodes.
        positions : torch.tensor (num_new_nodes, 3)
        """
        device = self._global_pos.device
        num_new_nodes = positions.shape[0]
        positions = positions.to(device=device)
        new_ids = torch.arange(self._next_node_id, self._next_node_id + num_new_nodes, dtype=torch.long, device=device)
        self._next_node_id += num_new_nodes

        if node_types is None:
            # Assumed that if type not specified then it's free space.
            encode = self._NODE_TYPE_ENCODE
            node_types = torch.full((num_new_nodes, ), encode["free_space"], dtype=torch.long, device=device)
        node_types = node_types.to(device)
        self._global_pos = torch.cat([self._global_pos, positions], dim=0) 
        self._global_node_ids = torch.cat([self._global_node_ids, new_ids])
        self._global_node_types = torch.cat([self._global_node_types, node_types])

        return new_ids

    def _occ_origin(self):
        if self.occ_grid is None:
            return (0.0, 0.0)
        
        h, w = self.occ_grid.shape
        ox = self.occ_center[0] + (w * self.occ_resolution) / 2.0
        oy = self.occ_center[1] - (h * self.occ_resolution) / 2.0
        return (ox, oy)


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

        gx0 = ((-wx0 + ox) / res).long()
        gy0 = ((+wy0 - oy) / res).long()
        gx1 = ((-wx1 + ox) / res).long()
        gy1 = ((+wy1 - oy) / res).long()

        edges_torch = torch.stack([gx0, gy0, gx1, gy1], dim=1)

        # -------------------------------
        # 2. TORCH → CUPY TRANSFER
        # -------------------------------
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

        occ_grid = np.rot90(occ_grid, 2)  # To account for the occ grid convetion as per GridMap
        self.occ_center = (occ_center_x, occ_center_y)
        self.occ_grid = occ_grid
        self.occ_resolution = resolution

        if not 0.0 <= self.retention_factor <= 1.0:
            raise ValueError("retention_factor must be between 0 and 1")

        device = self._global_pos.device

        global_pos = self._global_pos
        global_ids_tensor = self._global_node_ids

        """
        ############# LOCAL NODE MERGING TO GLOBAL START ##################
        """
        merge_start = torch.cuda.Event(enable_timing=True)
        merge_end = torch.cuda.Event(enable_timing=True)
        merge_start.record()

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

        # Re-read global state after merge (new nodes may have been appended)
        global_pos = self._global_pos
        global_ids_tensor = self._global_node_ids

        # Compute centroid and filter candidates (all on GPU)
        centroid = local_pos.mean(dim=0)
        mask = torch.norm(global_pos - centroid, dim=1) < self.max_candidate_edge_search_distance

        candidate_pos = global_pos[mask]
        candidate_ids_tensor = global_ids_tensor[mask]

        if len(candidate_pos) == 0:
            return

        dists = torch.cdist(local_pos, candidate_pos, p=2)

        # Find connections within threshold
        valid_mask = dists < self.max_candidate_edge_distance
        local_idx, cand_idx = torch.nonzero(valid_mask, as_tuple=True)

        if len(local_idx) == 0:
            return

        local_nodes_temp = local_ids[local_idx]
        global_nodes_temp = candidate_ids_tensor[cand_idx]
        no_self_loop_mask = local_nodes_temp != global_nodes_temp

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


    def tensor_merge_local_nodes(self, local_graph, global_worlds: torch.Tensor, global_ids_tensor: torch.Tensor, next_node_id, merge_distance, device="cuda"):
        """
        Merge local graph nodes into global graph using fully vectorized GPU operations.

        Returns (local_ids, local_id_positions, updated_next_id)
          - local_ids:          (N,) long tensor of global node IDs assigned to each local node
          - local_id_positions: (N, 3) positions (global pos for merged, local pos for new)
          - updated_next_id:    int, next available node ID
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
            self._global_node_ids = torch.cat([self._global_node_ids, new_ids], dim=0)
            self._global_node_types = torch.cat([self._global_node_types, new_types], dim=0)

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
            self._global_node_ids,
            self._global_node_types,
            self._global_edge_ids,
            self._global_edge_weights,
        )

    def boundary_probabilities(self) -> Dict[Tuple[int, int], float]:
        """Return the current obstacle probability map."""
        return dict(self._boundary_probs)

    def debug_visualize(self, path: str, scale: float = 100.0) -> None:
        """Render a simple 2D visualization of the global graph."""

        num_nodes = self._global_node_ids.shape[0]
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
            ids_cpu = self._global_node_ids.cpu().numpy()
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