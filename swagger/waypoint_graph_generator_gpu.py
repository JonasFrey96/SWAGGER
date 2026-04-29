import math
from dataclasses import dataclass

import cupy as cp
import cupyx.scipy.ndimage as cp_ndi
import numpy as np
import torch

@dataclass
class WaypointGraphGeneratorConfig:
    """Configuration for GPU waypoint graph generation.

    All distance parameters are in meters; pixel conversions happen internally.
    """
    boundary_inflation_factor: float = 1.1      # multiplier on safety_distance for obstacle dilation
    boundary_sample_distance: float = 0.4       # stride between boundary samples (m)
    free_space_sampling_threshold: float = 0.4  # max allowed dist-to-node for any free pixel (m)

    merge_node_distance: float = 0.25           # merge nodes closer than this (m)

    use_boundary_sampling: bool = True
    use_free_space_sampling: bool = True
    prune_graph: bool = True

    # When True, boundary nodes are placed by tracing each wall contour on CPU
    # (via connected-component labelling + greedy nearest-neighbour walk) and
    # sampling every boundary_sample_distance metres along the arc.
    # When False (default), a faster pure-GPU random subsample is used instead.
    structured_boundary_sampling: bool = False

    def __post_init__(self):
        for name in (
            "boundary_sample_distance",
            "free_space_sampling_threshold",
            "merge_node_distance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"'{name}' must be > 0, got {getattr(self, name)}")
        if self.boundary_inflation_factor <= 1.0:
            raise ValueError("'boundary_inflation_factor' must be > 1.0")


class WaypointGraphGeneratorGPU:
    """GPU-native waypoint graph generator.

    Takes an occupancy grid as a CUDA tensor and returns node positions
    in the same tensor layout used by GlobalGraphGenerator:

        pos        (N, 3)  float32  world XYZ (z=0)
        node_types (N,)    int32    1 = free_space (all nodes here are free_space)

    No edges are produced here — edge construction is handled upstream by
    GlobalGraphGenerator.add_local_graph, exactly as with the CPU version.

    Occupancy convention (matches the rest of this repo's input side):
        int8  -1 = unknown,  0 = free,  100 = occupied
    OR
        uint8  values > occupancy_threshold = free  (legacy WaypointGraphGenerator style)
    Both are accepted; pass occupancy_threshold=127 for the uint8 form.
    """

    def __init__(self, config: WaypointGraphGeneratorConfig | None = None):
        self._config = config or WaypointGraphGeneratorConfig()

        # set during build_graph_from_grid_map
        self._resolution: float | None = None
        self._safety_distance: float | None = None
        self._x_offset: float = 0.0
        self._y_offset: float = 0.0
        self._cos_rot: float = 1.0
        self._sin_rot: float = 0.0
        self._H: int = 0
        self._W: int = 0

        # intermediate maps (cupy arrays, reused across calls)
        self._dist_transform: cp.ndarray | None = None   # (H, W) float32, dist-to-obstacle per pixel
        self._inflated_map: cp.ndarray | None = None     # (H, W) uint8, 1 = inside inflation radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph_from_grid_map(
        self,
        image: torch.Tensor,          # (H, W) on CUDA, int8 or uint8
        resolution: float,
        safety_distance: float,
        occupancy_threshold: int = 127,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        rotation: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build waypoint graph nodes from an occupancy grid.

        Args:
            image:              CUDA tensor (H, W). int8 (-1/0/100) or uint8 (0-255).
            resolution:         Meters per pixel.
            safety_distance:    Robot radius in meters.
            occupancy_threshold:Threshold for uint8 maps; ignored for int8 maps.
            x_offset, y_offset: World-frame translation of the map origin.
            rotation:           Map-to-world rotation in radians.

        Returns:
            pos        (N, 3) float32 CUDA tensor — world XYZ of each waypoint (z=0).
            node_types (N,)   int32   CUDA tensor — all 1 (free_space).
        """
        assert image.is_cuda, "image must be a CUDA tensor"

        self._resolution = resolution
        self._safety_distance = safety_distance
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._cos_rot = math.cos(rotation)
        self._sin_rot = math.sin(rotation)
        self._H, self._W = image.shape[:2]
        device = image.device

        # ----------------------------------------------------------------
        # Step 1: binarize → free_map  (1=free, 0=obstacle/unknown)
        # ----------------------------------------------------------------
        free_map = self._binarize(image, occupancy_threshold)  # cupy uint8

        # ----------------------------------------------------------------
        # Trivial case: completely free map → uniform grid
        # ----------------------------------------------------------------
        if bool(free_map.all()):
            return self._create_grid_graph(device)

        # ----------------------------------------------------------------
        # Step 2: distance transform + inflate obstacles
        # ----------------------------------------------------------------
        self._compute_distance_and_inflation(free_map)

        # ----------------------------------------------------------------
        # Step 3 + 4: sample boundary waypoints
        # ----------------------------------------------------------------
        # pixel coordinates of the free space
        pixel_rows = []
        pixel_cols = []

        if self._config.use_boundary_sampling:
            b_rows, b_cols = self._sample_boundary_nodes()
            pixel_rows.append(b_rows)
            pixel_cols.append(b_cols)

        # ----------------------------------------------------------------
        # Step 5: sample free-space interior waypoints
        # ----------------------------------------------------------------
        if self._config.use_free_space_sampling:
            f_rows, f_cols = self._sample_free_space_nodes(pixel_rows, pixel_cols)
            pixel_rows.append(f_rows)
            pixel_cols.append(f_cols)

        if not pixel_rows:
            empty = torch.empty((0, 3), dtype=torch.float32, device=device)
            return empty, torch.empty((0,), dtype=torch.int32, device=device)

        rows = cp.concatenate(pixel_rows)  # (N,)
        cols = cp.concatenate(pixel_cols)  # (N,)

        # ----------------------------------------------------------------
        # Step 6: remove nodes that landed inside the inflated zone
        # ----------------------------------------------------------------
        rows, cols = self._remove_inflated_nodes(rows, cols)

        # ----------------------------------------------------------------
        # Step 7: prune nodes that are too close together
        # ----------------------------------------------------------------
        if self._config.prune_graph and rows.shape[0] > 1:
            rows, cols = self._prune_close_nodes(rows, cols)

        # ----------------------------------------------------------------
        # Convert pixel coords → world XYZ tensor
        # ----------------------------------------------------------------
        pos = self._pixels_to_world_tensor(rows, cols, device)
        node_types = torch.ones(pos.shape[0], dtype=torch.int32, device=device)

        return pos, node_types

    # ------------------------------------------------------------------
    # Trivial case — uniform grid for fully free maps
    # ------------------------------------------------------------------

    def _create_grid_graph(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a regular grid of waypoints with a safety-distance margin from the border."""
        H, W = self._H, self._W
        margin = int(self._safety_distance / self._resolution)
        step = max(1, int(self._config.free_space_sampling_threshold / self._resolution))

        rows = torch.arange(margin, H - margin, step, dtype=torch.float32)
        cols = torch.arange(margin, W - margin, step, dtype=torch.float32)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        r = grid_rows.reshape(-1)
        c = grid_cols.reshape(-1)

        pos = self._pixels_to_world_tensor(cp.asarray(r), cp.asarray(c), device)
        node_types = torch.ones(pos.shape[0], dtype=torch.int32, device=device)
        return pos, node_types

    # ------------------------------------------------------------------
    # Step 1 — binarize
    # ------------------------------------------------------------------

    def _binarize(self, image: torch.Tensor, threshold: int) -> cp.ndarray:
        """Return cupy uint8 array: 1=free, 0=obstacle/unknown."""
        img_cp = cp.asarray(image)
        if image.dtype == torch.int8:
            # -1=unknown, 0=free, 100=occupied  →  only 0 is free
            free = (img_cp == 0).astype(cp.uint8)
        else:
            free = (img_cp > threshold).astype(cp.uint8)
        return free

    # ------------------------------------------------------------------
    # Step 2 — distance transform + inflation
    # ------------------------------------------------------------------

    def _compute_distance_and_inflation(self, free_map: cp.ndarray):
        """Compute per-pixel L2 distance to nearest obstacle and the inflated mask."""
        # pad border with zeros (obstacle) so the EDT reflects map boundary as obstacle
        padded = cp.pad(free_map, 1, mode="constant", constant_values=0)        # (H+2. W+2)
        dist = cp_ndi.distance_transform_edt(padded).astype(cp.float32)         # (H+2, W+2)
        self._dist_transform = dist[1:-1, 1:-1]  # strip pad
        # self._dist_transform has the distance to each obstacle

        inflation_px = (
            self._config.boundary_inflation_factor * self._safety_distance / self._resolution
        )
        self._inflated_map = (self._dist_transform < inflation_px).astype(cp.uint8)

    # ------------------------------------------------------------------
    # Step 3+4 — boundary sampling
    # ------------------------------------------------------------------

    def _sample_boundary_nodes(self) -> tuple[cp.ndarray, cp.ndarray]:
        """Sample waypoints along the boundary of the inflation zone.

        The boundary is the one-pixel ring of safe-free pixels that directly
        border the inflated obstacle zone, found via morphological erosion.

        Dispatches to one of two strategies based on config.structured_boundary_sampling:
          False (default) — pure GPU: random subsample at the requested stride.
          True            — CPU-assisted: label connected components, trace each
                            contour in arc-length order, sample every
                            boundary_sample_distance metres along the path.
        """
        inflation_px = (
            self._config.boundary_inflation_factor * self._safety_distance / self._resolution
        )
        safe_free = (self._dist_transform >= inflation_px).astype(cp.uint8)

        struct = cp.ones((3, 3), dtype=cp.uint8)
        eroded = cp_ndi.binary_erosion(safe_free, structure=struct).astype(cp.uint8)
        boundary_mask = safe_free - eroded  # 1 only on boundary pixels

        if int(boundary_mask.sum()) == 0:
            return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

        if self._config.structured_boundary_sampling:
            return self._sample_boundary_structured(boundary_mask)
        return self._sample_boundary_random(boundary_mask)

    def _sample_boundary_random(
        self, boundary_mask: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Pure-GPU path: random subsample of boundary pixels, then greedy spacing filter.

        Random choice provides variety across calls; the greedy post-filter ensures
        no two kept nodes are closer than max(boundary_sample_distance, merge_node_distance)
        so the prune step won't discard any boundary nodes.
        """
        boundary_coords = cp.argwhere(boundary_mask)  # (M, 2)
        stride = max(1, int(self._config.boundary_sample_distance / (self._resolution * 2)))
        indices = cp.random.choice(
            boundary_coords.shape[0],
            size=max(1, boundary_coords.shape[0] // stride),
            replace=False,
        )
        sampled = boundary_coords[indices]
        rows, cols = sampled[:, 0].astype(cp.int32), sampled[:, 1].astype(cp.int32)

        # Post-filter: drop nodes closer than max(bsd, merge) so nothing is wasted by pruning.
        min_spacing_px = max(
            self._config.boundary_sample_distance / self._resolution,
            self._config.merge_node_distance / self._resolution,
        )
        return self._greedy_subsample(rows, cols, min_spacing_px)

    def _sample_boundary_structured(
        self, boundary_mask: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """CPU-assisted path: label contours, trace each in arc-length order, then sample.

        Each 8-connected component of the boundary (one per wall segment or obstacle
        outline) is traced independently.  This guarantees that samples are evenly
        spaced along the actual curve rather than randomly distributed.
        """
        labeled_cp, num_features = cp_ndi.label(
            boundary_mask, structure=cp.ones((3, 3), dtype=cp.int32)
        )
        labeled_np = cp.asnumpy(labeled_cp).astype(np.int32)

        step_px = max(1, int(self._config.boundary_sample_distance / self._resolution))

        sampled_rows: list[int] = []
        sampled_cols: list[int] = []

        for comp_id in range(1, num_features + 1):
            coords = np.argwhere(labeled_np == comp_id)  # (K, 2) row/col
            if coords.shape[0] == 0:
                continue
            ordered = self._trace_boundary_walk(coords)
            for i in range(0, ordered.shape[0], step_px):
                sampled_rows.append(int(ordered[i, 0]))
                sampled_cols.append(int(ordered[i, 1]))

        if not sampled_rows:
            return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

        return (
            cp.array(sampled_rows, dtype=cp.int32),
            cp.array(sampled_cols, dtype=cp.int32),
        )

    @staticmethod
    def _trace_boundary_walk(coords: np.ndarray) -> np.ndarray:
        """Order boundary pixels by arc-length via a greedy nearest-neighbour walk.

        Builds a cKDTree over the component's pixels and repeatedly moves to the
        nearest unvisited neighbour.  For a thin (1-px wide) contour this is
        identical to walking the curve; for thick regions it still produces a
        compact, low-backtrack path.

        Args:
            coords: (K, 2) int array of (row, col) pixel positions.

        Returns:
            (K, 2) int array of the same pixels reordered by arc-length.
        """
        from scipy.spatial import cKDTree

        if coords.shape[0] <= 2:
            return coords

        n = coords.shape[0]
        tree = cKDTree(coords)
        visited = np.zeros(n, dtype=bool)

        # Topmost-then-leftmost pixel — deterministic start regardless of argwhere order.
        start_idx = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
        order = [start_idx]
        visited[start_idx] = True
        curr = coords[start_idx]

        k = min(12, n)  # query k neighbours; handles thick boundaries with visited neighbours

        for _ in range(n - 1):
            _, idxs = tree.query(curr, k=k)
            moved = False
            for i in idxs:
                if not visited[i]:
                    order.append(int(i))
                    visited[i] = True
                    curr = coords[i]
                    moved = True
                    break
            if not moved:
                # All k-NN are visited — jump to global nearest unvisited pixel.
                # This handles disconnected islands within a component (rare but possible).
                dists = np.sum((coords - curr) ** 2, axis=1).astype(np.float64)
                dists[visited] = np.inf
                next_i = int(np.argmin(dists))
                order.append(next_i)
                visited[next_i] = True
                curr = coords[next_i]

        return coords[np.array(order, dtype=np.int32)]

    # ------------------------------------------------------------------
    # Step 5 — free-space sampling
    # ------------------------------------------------------------------

    def _sample_free_space_nodes(
        self,
        existing_rows: list[cp.ndarray],
        existing_cols: list[cp.ndarray],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Iteratively place interior waypoints so every free pixel is covered.

        Mirrors the CPU version exactly:
          1. EDT on blocked mask where 0 = obstacle/covered, 1 = uncovered free.
          2. Local maxima with dist > threshold_px are the best candidate sites.
          3. Greedily accept candidates spaced ≥ half_t apart.
          4. Mark accepted node PIXELS (not disks) as covered — so EDT in the
             next iteration measures true distance-to-nearest-node.
          5. Repeat until max(EDT) ≤ threshold_px, meaning every free pixel is
             within threshold_px of a node.
        """
        threshold_px = self._config.free_space_sampling_threshold / self._resolution
        half_t = threshold_px / 2.0
        # Ensure placed nodes are never closer than merge_node_distance so the
        # prune step doesn't immediately discard free-space nodes.
        merge_px  = self._config.merge_node_distance / self._resolution
        min_spacing = max(half_t, merge_px)

        # blocked_mask: 1=free and uncovered, 0=covered or obstacle
        blocked = (self._inflated_map == 0).astype(cp.uint8)

        # mark existing nodes as single pixels so EDT = distance-to-nearest-node
        if existing_rows:
            all_rows = cp.concatenate(existing_rows).astype(cp.int32)
            all_cols = cp.concatenate(existing_cols).astype(cp.int32)
            blocked[all_rows, all_cols] = 0

        struct3 = cp.ones((3, 3), dtype=cp.float32)
        max_iters = 12
        stall_limit = 2
        stall_iters = 0

        new_rows_list: list[cp.ndarray] = []
        new_cols_list: list[cp.ndarray] = []

        for _ in range(max_iters):
            dist = cp_ndi.distance_transform_edt(blocked).astype(cp.float32)
            max_dist = float(dist.max())

            if max_dist <= threshold_px:
                break

            # local maxima: pixel == max of 3×3 neighborhood AND dist > threshold
            neighborhood_max = cp_ndi.maximum_filter(dist, footprint=struct3)
            local_max_mask = (dist == neighborhood_max) & (dist > threshold_px)
            coords = cp.argwhere(local_max_mask)  # (K, 2)

            if coords.shape[0] == 0:
                stall_iters += 1
                if stall_iters >= stall_limit:
                    break
                continue

            accepted_rows, accepted_cols = self._greedy_subsample(
                coords[:, 0], coords[:, 1], min_spacing
            )

            if accepted_rows.shape[0] == 0:
                stall_iters += 1
                if stall_iters >= stall_limit:
                    break
                continue

            stall_iters = 0
            new_rows_list.append(accepted_rows)
            new_cols_list.append(accepted_cols)
            # mark only the node pixels — keeps EDT = true distance-to-nearest-node
            blocked[accepted_rows, accepted_cols] = 0

        if not new_rows_list:
            return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

        return (
            cp.concatenate(new_rows_list).astype(cp.int32),
            cp.concatenate(new_cols_list).astype(cp.int32),
        )

    @staticmethod
    def _greedy_subsample(
        rows: cp.ndarray,
        cols: cp.ndarray,
        min_spacing: float,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Accept candidates in order, rejecting any within min_spacing of an accepted node.

        Matches the CPU version's R-tree half_threshold radius check. Runs on CPU
        because the candidate set K is small (local maxima after EDT, typically < 200).
        """
        rows_np = cp.asnumpy(rows).astype(int)
        cols_np = cp.asnumpy(cols).astype(int)
        min_sq = min_spacing * min_spacing

        accepted_r: list[int] = []
        accepted_c: list[int] = []

        for r, c in zip(rows_np.tolist(), cols_np.tolist()):
            close = any(
                (r - ar) * (r - ar) + (c - ac) * (c - ac) < min_sq
                for ar, ac in zip(accepted_r, accepted_c)
            )
            if not close:
                accepted_r.append(r)
                accepted_c.append(c)

        if not accepted_r:
            return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

        return (
            cp.array(accepted_r, dtype=cp.int32),
            cp.array(accepted_c, dtype=cp.int32),
        )

    # ------------------------------------------------------------------
    # Step 6 — remove inflated nodes
    # ------------------------------------------------------------------

    def _remove_inflated_nodes(
        self, rows: cp.ndarray, cols: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Drop any node whose pixel falls inside the inflated obstacle zone."""
        valid = self._inflated_map[rows, cols] == 0
        return rows[valid], cols[valid]

    # ------------------------------------------------------------------
    # Step 7 — prune close nodes
    # ------------------------------------------------------------------

    def _prune_close_nodes(
        self, rows: cp.ndarray, cols: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Merge nodes closer than merge_node_distance using torch.cdist."""
        threshold_px = self._config.merge_node_distance / self._resolution

        coords = torch.stack([
            torch.as_tensor(rows, dtype=torch.float32),
            torch.as_tensor(cols, dtype=torch.float32),
        ], dim=1)  # (N, 2) on CUDA via DLPack

        # pairwise distances
        dist_mat = torch.cdist(coords, coords)  # (N, N)
        # upper-triangle mask to avoid double-counting
        triu = torch.triu(torch.ones(dist_mat.shape, dtype=torch.bool, device=coords.device), diagonal=1)
        close = (dist_mat < threshold_px) & triu  # (N, N)

        # greedily keep nodes: remove j whenever (i, j) is a close pair with i < j
        remove = close.any(dim=0)  # (N,) — remove if any earlier node is close to it
        keep = ~remove

        keep_cp = cp.asarray(keep)
        return rows[keep_cp], cols[keep_cp]

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _pixels_to_world_tensor(
        self, rows: cp.ndarray, cols: cp.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Convert pixel (row, col) arrays to world-frame XYZ tensor on `device`."""
        rows_t = torch.as_tensor(rows, dtype=torch.float32)
        cols_t = torch.as_tensor(cols, dtype=torch.float32)

        H, W = self._H, self._W

        # pixel centre → map-frame coordinates (origin at image centre)
        # row increases downward, so y_map = (H/2 - row) * resolution
        x_map = (cols_t - W / 2.0) * self._resolution
        y_map = (H / 2.0 - rows_t) * self._resolution

        # apply rotation then translation
        cos_r = self._cos_rot
        sin_r = self._sin_rot
        x_world = cos_r * x_map - sin_r * y_map + self._x_offset
        y_world = sin_r * x_map + cos_r * y_map + self._y_offset
        z_world = torch.zeros_like(x_world)

        pos = torch.stack([x_world, y_world, z_world], dim=1)  # (N, 3)
        return pos.to(device)
