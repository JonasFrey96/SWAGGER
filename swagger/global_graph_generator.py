"""Global Graph Assembler for SWAGGER.

This helper consumes per-frame/local graphs (such as the ones produced by
``WaypointGraphGenerator``) and incrementally builds a stitched, world-frame
networkx graph. It handles:

* merging nearby free-space nodes (ignores skeleton nodes)
* persisting/freeing nodes using a retention factor (0 → only current frame,
  1 → keep everything forever)
* maintaining a probabilistic heatmap of boundary observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Set, FrozenSet, List
import math

import cv2
import numpy as np
import networkx as nx
import torch
import cupy as cp
import string
import math
from collections import deque
from scipy.spatial import Delaunay, cKDTree

@dataclass
class GlobalGraphGenerator:
    """Incrementally build a stitched global graph.

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


    global_graph: nx.Graph = field(default_factory=nx.Graph, init=False)
    _next_node_id: int = field(default=0, init=False)
    _node_usage: Dict[int, float] = field(default_factory=dict, init=False)
    _boundary_probs: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)

    ## parameters for efficient SWAGGER graph
    sparsification_distance: float = 0.5                  # must be > merge.distance. check navigation_graph_node
    frontier_hop_buffer: int = 3                          # BFS from frontier and how many edges to hop
    sparsification_frequency: int = 20                    # num of local graph updates for sparisficiation cycle to begin
    _sparsify_counter: int = field(default=0, init=True)
    _prune_counter: int = field(default=0, init=True_pr)

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global_ids = list(self.global_graph.nodes())
        if len(global_ids) > 0:
            global_pos = torch.tensor(
                # [self.global_graph.nodes[n]["world"][:2] for n in global_ids],
                [self.global_graph.nodes[n]["world"] for n in global_ids],
                dtype=torch.float32,
                device=device
            )
            global_ids_tensor = torch.tensor(global_ids, dtype=torch.long, device=device)
        else:
            global_pos = torch.empty((0, 3), dtype=torch.float32, device=device)
            global_ids_tensor = torch.empty((0,), dtype=torch.long, device=device)

        """
        ############# LOCAL NODE MERGING TO GLOBAL START ##################
        """
        merge_start = torch.cuda.Event(enable_timing=True)
        merge_end = torch.cuda.Event(enable_timing=True)
        merge_start.record()

        local_ids, self._next_node_id = self.tensor_merge_local_nodes(
            local_graph,
            self.global_graph,
            global_pos,
            global_ids_tensor,
            self._next_node_id,
            self.merge_distance,
            device=device
        )

        merge_end.record()
        torch.cuda.synchronize()
        local_merge_ms = merge_start.elapsed_time(merge_end)
    
        if not local_ids:
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
        
        local_pos = torch.tensor(
            # [self.global_graph.nodes[n]["world"][:2] for n in local_ids],
            [self.global_graph.nodes[n]["world"] for n in local_ids],
            dtype=torch.float32,
            device=device
        )
        
        # Compute centroid and filter candidates (all on GPU)
        centroid = local_pos.mean(dim=0)
        mask = torch.norm(global_pos - centroid, dim=1) < self.max_candidate_edge_search_distance

        
        candidate_pos = global_pos[mask]
        global_ids_tensor = torch.tensor(global_ids, device=device, dtype=torch.int64)
        candidate_ids_tensor = global_ids_tensor[mask]
        
        if len(candidate_pos) == 0:
            return
        
        dists = torch.cdist(local_pos, candidate_pos, p=2)
        
        # Find connections within threshold
        valid_mask = dists < self.max_candidate_edge_distance
        local_idx, cand_idx = torch.nonzero(valid_mask, as_tuple=True)


        
        if len(local_idx) == 0:
            return
        
        local_ids_tensor = torch.tensor(local_ids, device=device, dtype=torch.int64)

        local_nodes_temp = local_ids_tensor[local_idx]
        global_nodes_temp = candidate_ids_tensor[cand_idx]
        no_self_loop_mask = local_nodes_temp != global_nodes_temp

        local_idx = local_idx[no_self_loop_mask]
        cand_idx = cand_idx[no_self_loop_mask]
        
        local_nodes = local_ids_tensor[local_idx]
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

        # edges = list(zip(
        #     local_nodes.cpu().tolist(),
        #     global_nodes.cpu().tolist(),
        #     weights.cpu().tolist()
        # ))

        edge_data = torch.stack([local_nodes, global_nodes, weights], dim=1).cpu().tolist()
        edges = [(int(row[0]), int(row[1]), row[2]) for row in edge_data]
        self.global_graph.add_weighted_edges_from(edges)
        add_end.record()
        torch.cuda.synchronize()
        add_edges_ms = add_start.elapsed_time(add_end)

        """
        ############# ADDING EDGES TO THE GRAPH END ##################
        """
        
        # self._prune_redundant_edges()
        self._prune_and_sparsify()


    def tensor_merge_local_nodes(self, local_graph, global_graph: nx.Graph, global_worlds: torch.tensor, global_ids_tensor: torch.tensor ,next_node_id, merge_distance, device="cuda"):
        """
        Merge local graph nodes into global graph using fully vectorized GPU operations.

        - Ignores node_type filtering and boundary logic (all nodes treated as free_space)
        - Returns local_ids like the original loop
        """

        # -------------------------
        # Extract local nodes
        # -------------------------
        local_nodes = list(local_graph.nodes())
        if len(local_nodes) == 0:
            return [], next_node_id

        local_worlds = torch.tensor(
            [local_graph.nodes[n]["world"] for n in local_nodes],
            dtype=torch.float32,
            device=device
        )  # (N, D)
        N = local_worlds.shape[0]

        # -------------------------
        # Extract global nodes
        # -------------------------
        # global_nodes = list(global_graph.nodes())
        # if len(global_nodes) > 0:
        #     global_worlds = torch.tensor(
        #         [global_graph.nodes[n]["world"] for n in global_nodes],
        #         dtype=torch.float32,
        #         device=device
        #     )  # (M, D)
        #     global_ids_tensor = torch.tensor(global_nodes, dtype=torch.long, device=device)
        # else:
        #     global_worlds = torch.empty((0, local_worlds.shape[1]), device=device)
        #     global_ids_tensor = torch.empty((0,), dtype=torch.long, device=device)

        # -------------------------
        # CASE 1: Empty global graph
        # -------------------------
        if global_worlds.numel() == 0:
            new_ids = torch.arange(next_node_id, next_node_id + N, device=device, dtype=torch.long)
            final_ids = new_ids
            new_local_mask = torch.ones(N, dtype=torch.bool, device=device)
            updated_next_id = next_node_id + N
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

        # -------------------------
        # Update global graph with new nodes
        # -------------------------
        # for i, nid in enumerate(final_ids.cpu().tolist()):
        #     if new_local_mask[i]:
        #         world = local_worlds[i].cpu().tolist()
        #         global_graph.add_node(nid, world=tuple(world), node_type=local_graph.nodes[local_nodes[i]]["node_type"], origin="new")
        #     # Optional: mark usage _node_usage[nid] = 1.0

        # # -------------------------
        # # Return local_ids in same order as original loop
        # # -------------------------
        final_ids_list = final_ids.cpu().tolist()
        new_local_mask_cpu = new_local_mask.cpu().tolist()
        local_worlds_cpu = local_worlds.cpu().tolist()  # single transfer, all at once
        ## Batch building is more efficient. 
        new_nodes = []
        for i, nid in enumerate(final_ids_list):
            if new_local_mask_cpu[i]:
                new_nodes.append((
                    nid,
                    {
                        "world": tuple(local_worlds_cpu[i]),
                        "node_type": local_graph.nodes[local_nodes[i]]["node_type"],
                        "origin": "new",
                    }
                ))

        # Single networkx call
        global_graph.add_nodes_from(new_nodes)
        local_ids = final_ids.cpu().tolist()
        return local_ids, updated_next_id

    

    def _prune_redundant_edges(self) -> None:
        """Limit node degree by keeping only the closest neighbors."""
        max_degree = self.max_connections

        # Only prune every N frames to reduce overhead
        if not hasattr(self, '_prune_counter'):
            self._prune_counter = 0
        self._prune_counter += 1
        
        # Only run every 10 frames (less frequent)
        if self._prune_counter % self.pruning_frequency != 0:
            return
        
        edges_to_remove = set()  # Use set to avoid duplicates
        
        for node in self.global_graph.nodes():
            neighbors = list(self.global_graph.neighbors(node))
            degree = len(neighbors)
            
            # Only prune if extremely over-connected
            # print(degree)
            if degree <= max_degree:
                continue
            
            # Use list comprehension for efficiency
            neighbor_weights = [(self.global_graph[node][neighbor]["weight"], neighbor) 
                              for neighbor in neighbors]
            
            neighbor_weights.sort()  # Sort by distance (shortest first)
            
            # Keep only the max_degree closest neighbors
            for weight, neighbor in neighbor_weights[max_degree:]:
                edges_to_remove.add(tuple(sorted((node, neighbor))))
        
        # Batch remove edges
        self.global_graph.remove_edges_from(edges_to_remove)

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

    

    def get_global_graph(self) -> nx.Graph:
        """Return the stitched global graph."""
        return self.global_graph

    def boundary_probabilities(self) -> Dict[Tuple[int, int], float]:
        """Return the current obstacle probability map."""
        return dict(self._boundary_probs)

    def debug_visualize(self, path: str, scale: float = 100.0) -> None:
        """Render a simple 2D visualization of the global graph."""

        if len(self.global_graph) == 0:
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(path, blank)
            return

        xs, ys = [], []
        for _, data in self.global_graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                continue
            xs.append(float(world[0]))
            ys.append(float(world[1]))
        if not xs or not ys:
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(path, blank)
            return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
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
        for u, v in self.global_graph.edges():
            n1 = self.global_graph.nodes[u]
            n2 = self.global_graph.nodes[v]
            if "world" not in n1 or "world" not in n2:
                continue
            x1, y1 = to_px(float(n1["world"][0]), float(n1["world"][1]))
            x2, y2 = to_px(float(n2["world"][0]), float(n2["world"][1]))
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Draw nodes
        for _, data in self.global_graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                continue
            x, y = to_px(float(world[0]), float(world[1]))
            origin = str(data.get("origin", "new"))
            color = (0, 0, 255) if origin == "known" else (255, 0, 0)
            cv2.circle(canvas, (x, y), 3, color, -1)

        cv2.imwrite(path, canvas)
    
    def _compute_mst_edges(self) -> Set[FrozenSet]:
        """
        Compute MST edges across all connected components.
        Returns a set of frozensets for O(1) membership checks.
        """
        G = self.global_graph
        mst_edges: Set[FrozenSet] = set()

        for comp in nx.connected_components(G):
            sg = G.subgraph(comp)
            if sg.number_of_edges() > 0:
                try:
                    mst = nx.minimum_spanning_tree(sg, weight="weight")
                    mst_edges.update(frozenset(e) for e in mst.edges())
                except Exception:
                    pass

        return mst_edges
    
    def _prune_and_sparsify(self) -> None:
        """
        1. Logic is at some time intervals, do BFS on the frontier nodes and then till frontier_hop_buffer hops
        from the frontier nodes and mark them + frontiers.
        2. Then for the free space nodes we form a KDTree of distance sparsification_distance around the free spaces and mark these nodes.
        3. We also do the MST of the graph to make sure when we prune the edges we do not hurt the connectivity of the graph.
        """
        
        G = self.global_graph
        
        if G.number_of_nodes() == 0:
            self._sparsify_counter += 1
            return
        
        self._prune_counter += 1
        self._sparsify_counter += 1

        run_prune = (self._prune_counter % self.pruning_frequency == 0)
        run_sparsify = (self._sparsify_counter % self.pruning_frequency == 0)

        if not run_prune and not run_sparsify:
            return 

        if run_prune:
            mst_edges = self._compute_mst_edges()
            self._prune_redundant_edges_mst_safe()
        
        if run_sparsify:
            self.sparsify_explored_regions()
    
    def _find_settled_nodes(self) -> Tuple[Set[int], List[int]]:
        """
        pass over all the global nodes (i know it's inefficient but over time should reduce computation)
        mark frontiers -> BFS source
        mark free_space -> unprotected

        mark frontiers to free_space nodes as protected.
        Returns Set(settled nodes), frontier nodes(list)
        """
        G = self.global_graph
        hop_buffer = self.frontier_hop_buffer

        frontier_nodes = []
        free_space_nodes = set()

        for n, d in G.nodes(data=True):
            node_type = d.get("node_type")

            if node_type == "frontier":
                ## NOTE: need to check whether frontier is being given 1 or "frontier"
                frontier_nodes.append(n)
            if node_type == "free_space":
                free_space_nodes.add(n)
        
        if not free_space_nodes:
            print(f"couldn't find any free space node")
            return set(), frontier_nodes
        
        protected: Set[int] = set()

        if frontier_nodes:
            visited = Set[int] = set(frontier_nodes)
            queue = deque((n, 0) for n in frontier_nodes)

            while queue:
                node, depth = queue.popleft()
                protected.add(node)
                if depth >= hop_buffer:
                    continue
                
                for nbr in G.neighbors(node):
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
        
        settled = free_space_nodes - protected
        return settled, frontier_nodes

    def sparsify_explored_regions(self) -> int:
            """
            Takes in the settled nodes and does KDTree on them.
            """
            G = self.global_graph
            if G.number_of_nodes() < 2:
                return 0

            settled, _ = self._find_settled_nodes()
            if len(settled) < 2:
                return 0

            settled_list = list(settled)
            positions = np.array(
                [G.nodes[n]["world"][:2] for n in settled_list],
                dtype=np.float64,
            )

            tree = cKDTree(positions)
            pairs = tree.query_pairs(r=self.sparsification_distance)
            if not pairs:
                return 0
            
            pair_list = []
            for i, j in pairs:
                d = np.linalg.norm(positions[i] - positions[j])
                pair_list.append((d, settled_list[i], settled_list[j]))
            pair_list.sort()

            removed_set: Set[int] = set()
            new_edges: List[Tuple[int, int]] = []
            removed_count = 0

            for _, node_a, node_b in pair_list:
                if node_a in removed_set or node_b in removed_set:
                    continue
                if not G.has_node(node_a) or not G.has_node(node_b):
                    continue

                # Keep the more connected node
                if G.degree(node_a) >= G.degree(node_b):
                    keep, remove = node_a, node_b
                else:
                    keep, remove = node_b, node_a
                wk = G.nodes[keep]["world"]
                wr = G.nodes[remove]["world"]
                midpoint = tuple((wk[i] + wr[i]) / 2.0 for i in range(len(wk)))
                G.nodes[keep]["world"] = midpoint
                mid_xy = midpoint[:2]

                for nbr in list(G.neighbors(remove)):
                    if nbr == keep or nbr in removed_set:
                        continue

                    w_nbr = G.nodes[nbr]["world"][:2]
                    new_weight = math.sqrt(
                        (mid_xy[0] - w_nbr[0]) ** 2 +
                        (mid_xy[1] - w_nbr[1]) ** 2
                    )

                    if G.has_edge(keep, nbr):
                        existing_weight = G[keep][nbr].get("weight", float("inf"))
                        if new_weight < existing_weight:
                            G[keep][nbr]["weight"] = new_weight
                            new_edges.append((keep, nbr))  # geometry changed, needs check
                        # else: existing edge unchanged, no collision check needed
                    else:
                        G.add_edge(keep, nbr, weight=new_weight)
                        new_edges.append((keep, nbr))

                G.remove_node(remove)
                removed_set.add(remove)
                removed_count += 1

            # ─── Validate ONLY new/modified edges ───
            print(f"the number of removed nodes in sparsify is {removed_count}")
            if new_edges and removed_count > 0:
                mst_edges = self._compute_mst_edges()
                self._validate_new_edges(new_edges, mst_edges)

            return removed_count
    
    def _validate_new_edges(
        self,
        edges_to_check: List[Tuple[int, int]],
        mst_edges: Set[FrozenSet],
    ) -> int:
        """
        Collision-check only the specified edges (newly created/rewired).
        Remove colliding ones unless they're in the MST.
        Returns number of edges removed.
        """
        G = self.global_graph
        if not edges_to_check or self.occ_grid is None:
            return 0

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Deduplicate and filter to edges that still exist
        p1_list, p2_list, valid_edges = [], [], []
        seen: Set[FrozenSet] = set()
        for u, v in edges_to_check:
            edge_key = frozenset((u, v))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if not G.has_node(u) or not G.has_node(v):
                continue
            if not G.has_edge(u, v):
                continue
            if "world" not in G.nodes[u] or "world" not in G.nodes[v]:
                continue

            p1_list.append(G.nodes[u]["world"][:2])
            p2_list.append(G.nodes[v]["world"][:2])
            valid_edges.append((u, v))

        if not valid_edges:
            return 0

        p1 = torch.tensor(p1_list, dtype=torch.float32, device=device)
        p2 = torch.tensor(p2_list, dtype=torch.float32, device=device)
        dummy_ids = torch.zeros(len(valid_edges), dtype=torch.int64, device=device)

        collision_free = self.batch_collision_check(p1, p2, dummy_ids, dummy_ids)

        # Remove colliding non-MST edges
        collision_mask = collision_free.cpu().tolist()
        removed = 0
        for i, is_free in enumerate(collision_mask):
            if not is_free:
                u, v = valid_edges[i]
                if frozenset((u, v)) not in mst_edges:
                    G.remove_edge(u, v)
                    removed += 1

        return removed

    def _prune_redundant_edges_mst_safe(self, mst_edges: Set[FrozenSet] = None) -> None:
        """
        Limit node degree to max_connections by dropping the longest
        non-MST edges.
        """
        G = self.global_graph
        if G.number_of_edges() == 0:
            return

        if mst_edges is None:
            mst_edges = self._compute_mst_edges()

        edges_to_remove = []

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) <= self.max_connections:
                continue

            neighbor_weights = [
                (G[node][nbr].get("weight", float("inf")), nbr)
                for nbr in neighbors
            ]
            neighbor_weights.sort()

            kept = 0
            for weight, nbr in neighbor_weights:
                if kept < self.max_connections:
                    kept += 1
                else:
                    if frozenset((node, nbr)) not in mst_edges:
                        edges_to_remove.append((node, nbr))

        G.remove_edges_from(edges_to_remove)

