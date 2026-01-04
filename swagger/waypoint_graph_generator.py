# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

import cv2

import networkx as nx
import numba
from scipy.spatial import Delaunay, cKDTree
from rtree import index

from swagger.logger import Logger
from swagger.models import Point
from swagger.utils import pixel_to_world, world_to_pixel

@dataclass
class WaypointGraphGeneratorConfig:
    """Configuration for waypoint graph generation algorithm.

    All distance parameters are specified in meters and are converted
    to pixel units internally based on the resolution.
    """
    # Graph generation parameters (in meters)
    boundary_inflation_factor: float = 1.5      # Factor to inflate boundaries by safety distance (unitless)
    boundary_sample_distance: float = 2.5       # Distance between samples along contour
    free_space_sampling_threshold: float = 1.5  # Maximum distance from obstacles

    # Graph pruning parameters (in meters)
    merge_node_distance: float = 0.25   # Maximum distance to merge nodes
    min_subgraph_length: float = 0.25   # Minimum total edge length to keep

    # Function flags
    use_boundary_sampling: bool = True
    use_free_space_sampling: bool = True
    use_delaunay_shortcuts: bool = True
    prune_graph: bool = True

    # Debug flag to control debug file output
    debug: bool = False

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # List of float parameters that must be positive
        distance_params = [
            "boundary_sample_distance",
            "free_space_sampling_threshold",
            "merge_node_distance",
            "min_subgraph_length",
        ]

        # Check each parameter
        for name in distance_params:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"Parameter '{name}' must be greater than 0, got {value}")

        if self.boundary_inflation_factor <= 1.0:
            raise ValueError("Parameter 'boundary_inflation_factor' must be greater than 1.0")

class WaypointGraphGenerator:
    """Generates a waypoint graph from an occupancy grid map."""
    def __init__(
        self,
        config: WaypointGraphGeneratorConfig = WaypointGraphGeneratorConfig(),
        logger_level: int = logging.INFO
    ):
        self._graph: nx.Graph | None = None             # Store the current graph
        self._original_map: np.ndarray | None = None    # Store original map for visualization
        self._inflated_map: np.ndarray | None = None    # Store inflated occupancy map for obstacle avoidance

        self._config = config if config is not None else WaypointGraphGeneratorConfig()

        self._resolution: float | None = None       # Store resolution for coordinate conversion
        self._safety_distance: float | None = None  # Store robot radius for visualization
        self._occupancy_threshold: int = 127        # Store occupancy threshold

        # Store transform parameters (in meters)
        self._x_offset: float = 0.0
        self._y_offset: float = 0.0
        self._rotation: float = 0.0  # Rotation in radians
        self._cos_rot: float = 1.0
        self._sin_rot: float = 0.0

        self._logger = Logger(__name__, level=logger_level)

        self._node_map: np.ndarray | None = None  # Nearest node map for fast lookup

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    # ==========================================================
    # Public API
    # ==========================================================

    def build_graph_from_grid_map(
        self,
        image: np.ndarray,
        resolution: float,
        safety_distance: float,
        occupancy_threshold: int = 127,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        rotation: float = 0.0,
    ) -> nx.Graph:
        """Build a waypoint graph from an occupancy grid map.

        Args:
            image: Occupancy grid (0-255, where values <= threshold are considered occupied)
            resolution: Meters per pixel in the map
            safety_distance: Radius of the robot in meters
            occupancy_threshold: Threshold to determine occupied cells (0-255)
            x_offset: Translation in the x direction (meters)
            y_offset: Translation in the y direction (meters)
            rotation: Rotation about the Z axis (radians)

        Returns:
            NetworkX.Graph object containing the waypoint graph
        """
        # --------------------------------------------------------
        # Initialize map and transform parameters
        # --------------------------------------------------------

        self._image_shape = image.shape[:2]

        self._logger.info("Building graph from grid map...")
        self._resolution = resolution
        self._safety_distance = safety_distance
        self._original_map = copy.deepcopy(image)
        self._occupancy_threshold = occupancy_threshold
        self._x_offset = x_offset

        # When using ROS coordinates, we need to add the image height to the y_offset
        # This shifts the origin from top-left to bottom-left
        image_height = image.shape[0]
        #self._y_offset = y_offset + (image_height * resolution)    # Add image height in meters
        self._y_offset = y_offset

        self._cos_rot = np.cos(rotation)
        self._sin_rot = np.sin(rotation)
        self._node_map = None

        # --------------------------------------------------------
        # Check for trivial case: completely free map
        # --------------------------------------------------------

        free_map = (self._original_map > self._occupancy_threshold).astype(np.uint8)

        if np.all(free_map):
            # Map is completely free (all values > threshold), create grid graph directly
            self._graph = self._create_grid_graph(
                image.shape,
                grid_sample_distance=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )
            self._build_nearest_node_map(self._graph)
            return self._graph

        # --------------------------------------------------------
        # Inflate obstacles using distance transform
        # --------------------------------------------------------

        self._distance_transform(free_map)

        # --------------------------------------------------------
        # Build initial graph from boundary (contour) sampling
        # --------------------------------------------------------

        graph = nx.Graph()

        if self._config.use_boundary_sampling:
            self._sample_obstacle_boundaries(
                graph,
                sample_distance=self._to_pixels_int(self._config.boundary_sample_distance)
            )

        # --------------------------------------------------------
        # Sample free space for additional waypoints
        # --------------------------------------------------------

        new_points_px = []
        if self._config.use_free_space_sampling:
            new_points_px = self._sample_free_space(
                graph,
                distance_threshold=self._to_pixels_int(
                    self._config.free_space_sampling_threshold
                )
            )

        # --------------------------------------------------------
        # Post-sampling cleanup and validation
        # --------------------------------------------------------

        # Remove any free-space samples that landed on inflated obstacles/edges
        self._remove_invalid_free_nodes(graph)

        new_points_px = [pt for pt in new_points_px if self._is_free_pixel(pt[0], pt[1])]

        # --------------------------------------------------------
        # Ensure local connectivity between inherited and new nodes
        # --------------------------------------------------------

        self._connect_free_nodes(graph)

        # --------------------------------------------------------
        # Enforce world/pixel/pos/node_type consistency for all nodes
        # --------------------------------------------------------

        for _, data in graph.nodes(data=True):

            # Infer missing pixel/world coordinates
            if "pixel" not in data and "world" in data:
                x_w, y_w = data["world"]
                y_p, x_p = self._world_to_pixel(Point(x=x_w, y=y_w, z=0))
                data["pixel"] = (y_p, x_p)
            elif "world" not in data and "pixel" in data:
                y_p, x_p = data["pixel"]
                x_w, y_w = self._pixel_to_world(y_p, x_p)
                data["world"] = (x_w, y_w)

            if "pos" not in data and "pixel" in data:
                data["pos"] = (float(data["world"][0]), float(data["world"][1]))
            if "node_type" not in data:
                data["node_type"] = "free_space"

        if len(graph.nodes) == 0:
            self._logger.warning("[WARN] Graph has no nodes after world/pixel consistency check.")
        else:
            self._logger.info(f"[INFO] Graph consistency OK: {len(graph.nodes)} nodes.")

        # --------------------------------------------------------
        # Add geometric shortcuts using Delaunay triangulation
        # --------------------------------------------------------

        if self._config.use_delaunay_shortcuts:
            self._add_delaunay_shortcuts(graph)

        # --------------------------------------------------------
        # Graph pruning and final graph storage
        # --------------------------------------------------------

        if self._config.prune_graph:
            self._prune_graph(graph)

        # Store and convert the graph to CSR format
        self._graph = graph
        self._logger.info(f"Final graph has {len(graph.nodes)} nodes and {len(graph.edges)} unique edges")

        # --------------------------------------------------------
        # Normalize pixel attributes for all nodes
        # --------------------------------------------------------

        for node, data in self._graph.nodes(data=True):
            pix = data.get("pixel")
            if pix is None:
                if isinstance(node, tuple) and len(node) == 2:
                    data["pixel"] = node
                continue

            pix_arr = np.array(pix).flatten()

            if pix_arr.shape[0] < 2:
                data.pop("pixel", None)
                continue

            y, x = map(int, pix_arr[:2])
            data["pixel"] = (y, x)

        # --------------------------------------------------------
        # Build nearest-node acceleration structure
        # --------------------------------------------------------

        self._build_nearest_node_map(self._graph)
        return self._graph

    def _astar_heuristic(self, n1, n2):
        """A* heuristic for the graph."""
        world_n1 = self._graph.nodes[n1]["world"]
        world_n2 = self._graph.nodes[n2]["world"]
        return np.linalg.norm([world_n1[0] - world_n2[0], world_n1[1] - world_n2[1]])

    def find_route(self, start: Point, goal: Point, shortcut_distance: float = 0.0) -> list[Point]:
        """Find a route between two points in the graph. If the distance between the start and goal is less than
        shortcut_distance, we check if the line between them is collision free and return the start and goal points
        if it is. Otherwise, we use A* to find a route on the graph.
        """

        distance = np.linalg.norm([start.x - goal.x, start.y - goal.y])
        if distance < shortcut_distance:
            start_pixel = self._world_to_pixel(start)
            goal_pixel = self._world_to_pixel(goal)
            if not self._check_line_collision(start_pixel, goal_pixel):
                return [start, goal]
        try:
            start_node, goal_node = self.get_node_ids([start, goal])
        except Exception as e:
            self._logger.error(f"Error getting start and goal nodes: {start} - {goal}: {e}")
            return []

        if start_node is None:
            self._logger.error(f"{start} Start is out of bounds")
            return []
        if goal_node is None:
            self._logger.error(f"{goal} Goal is out of bounds")
            return []

        try:
            route_nodes = nx.astar_path(self._graph, start_node, goal_node, heuristic=self._astar_heuristic)
            return (
                [start]
                + [
                    Point(
                        x=self._graph.nodes[node]["world"][0],
                        y=self._graph.nodes[node]["world"][1],
                        z=self._graph.nodes[node]["world"][2],
                    )
                    for node in route_nodes
                ]
                + [goal]
            )
        except nx.NetworkXNoPath:
            self._logger.error(f"No path found for start: {start} and goal: {goal}")
            return []

    def get_node_ids(self, points: list[Point]) -> list[Optional[int]]:
        """Find the nearest graph node to each query point on the node lookup map."""
        if self._graph is None:
            raise RuntimeError("No graph has been built yet")

        if self._node_map is None:
            self._build_nearest_node_map(self._graph)

        results = []
        for point in points:
            y, x = self._world_to_pixel(point)
            if not self._is_within_bounds(y, x):
                results.append(None)
                continue
            label = self._node_map[y, x]
            if label < 0:
                results.append(None)
            else:
                results.append(int(label))

        return results

    # ==========================================================
    # Coordinate transforms & unit conversions
    # ==========================================================

    def _pixel_to_world(self, row: float, col: float) -> Point:
        """Convert pixel coordinates to world coordinates with the current transform."""
        return pixel_to_world(row, col, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot, self._image_shape)

    def _world_to_pixel(self, point: Point) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates with the inverse transform."""
        return world_to_pixel(point, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot, self._image_shape)

    def _point_to_tuple(self, value) -> tuple[float, float, float]:
        """Normalize Point-like inputs to an (x, y, z) tuple."""
        if hasattr(value, "x") and hasattr(value, "y"):
            x = float(value.x)
            y = float(value.y)
            z = float(getattr(value, "z", 0.0))
            return (x, y, z)
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
            x = float(value[0])
            y = float(value[1])
            z = float(value[2]) if len(value) >= 3 else 0.0
            return (x, y, z)
        return (0.0, 0.0, 0.0)

    def _to_pixels(self, meters: float) -> float:
        """Convert a distance from meters to pixels."""
        if self._resolution is None:
            raise ValueError("Resolution not set. Call build_graph_from_grid_map first.")
        return meters / self._resolution

    def _to_pixels_int(self, meters: float) -> int:
        """Convert a distance from meters to pixels and round to integer."""
        return int(self._to_pixels(meters))

    # ==========================================================
    # Map preprocessing
    # ==========================================================

    def _free_mask_from_map(self, occupancy: np.ndarray) -> np.ndarray:
        """Return binary mask of free space (1 = free)."""
        if occupancy is None:
            raise RuntimeError("Occupancy map is not initialized.")
        if occupancy.max() <= 1:
            return (occupancy == 0).astype(np.uint8)
        return (occupancy > self._occupancy_threshold).astype(np.uint8)

    def _distance_transform(self, free_map: np.ndarray):
        """Inflate obstacles in the occupancy grid using a distance transform."""
        # Pad the binary map by 1 pixel on all sides to avoid nodes being created on the edges of the map
        free_map = np.pad(free_map, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        # Compute the distance transform
        self._dist_transform = cv2.distanceTransform(free_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Filter the distance transform by the robot's radius
        self._inflated_map = (self._dist_transform < self._safety_distance / self._resolution).astype(np.uint8)
        # Unpad the distance transform to get the original map shape
        self._inflated_map = self._inflated_map[1:-1, 1:-1]

    def _distance_transform_cuda(self, free_map: np.ndarray):
        """Inflate obstacles in the occupancy grid using a GPU-accelerated distance transform (if cupy is available). Currently unused."""
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import distance_transform_edt
            use_gpu = True
        except Exception:
            use_gpu = False

        free_map_padded = np.pad(
            free_map,
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )

        if use_gpu:
            free_gpu = cp.asarray(free_map_padded.astype(np.uint8))

            # CUDA Euclidean Distance Transform
            dist_gpu = distance_transform_edt(free_gpu)

            # Store dist transform (for legacy users)
            self._dist_transform = cp.asnumpy(dist_gpu)

            safety_px = float(self._safety_distance) / float(self._resolution)

            inflated_gpu = (dist_gpu < safety_px).astype(cp.uint8)

            self._inflated_map = cp.asnumpy(inflated_gpu[1:-1, 1:-1])

        else:
            self._logger.warning(
                "[WARN] CuPy not available — falling back to CPU distance transform."
            )

            self._dist_transform = cv2.distanceTransform(
                free_map_padded,
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )

            self._inflated_map = (
                self._dist_transform < self._safety_distance / self._resolution
            ).astype(np.uint8)

            self._inflated_map = self._inflated_map[1:-1, 1:-1]

    def _is_free_pixel(self, row: int, col: int) -> bool:
        """Check whether a pixel lies within map bounds and is free of inflated obstacles."""
        if self._inflated_map is None:
            return True
        h, w = self._inflated_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self._inflated_map[row, col] == 0

    def _is_within_bounds(self, row: int, col: int) -> bool:
        """Check if a point is within the bounds of the map."""
        if self._inflated_map is None:
            return False
        return 0 <= row < self._inflated_map.shape[0] and 0 <= col < self._inflated_map.shape[1]

    def _is_valid_point(self, row: int, col: int) -> bool:
        """Check if a point is within bounds and in free space."""
        return self._is_within_bounds(row, col) and not self._inflated_map[row, col]

    # ==========================================================
    # Graph construction primitives
    # ==========================================================

    def _create_grid_graph(self, shape: tuple[int, int], grid_sample_distance: int) -> nx.Graph:
        """Create a grid graph for completely free maps with a margin from the borders."""
        self._logger.info("Creating grid for completely free map with margin...")
        height, width = shape

        # Calculate margin based on robot radius and resolution
        margin = int(self._safety_distance / self._resolution)

        # For very small maps, just create a single node
        if height <= 2 * margin or width <= 2 * margin:
            raise ValueError(
                f"Map is too small to create a grid graph (height: {height}, width: {width}, safety_distance:"
                f" {self._safety_distance}, resolution: {self._resolution})"
            )

        step = max(1, min(grid_sample_distance, min(height - 2 * margin, width - 2 * margin) // 2))
        graph = nx.Graph()

        # Create grid nodes starting from corners with a margin
        for y in range(margin, height - margin, step):
            for x in range(margin, width - margin, step):
                graph.add_node((y, x))
                # Connect to right neighbor
                if x + step < width - margin:
                    graph.add_edge((y, x), (y, x + step), weight=step, edge_type="grid")
                # Connect to bottom neighbor
                if y + step < height - margin:
                    graph.add_edge((y, x), (y + step, x), weight=step, edge_type="grid")
                # Connect to diagonal neighbor
                if x + step < width - margin and y + step < height - margin:
                    graph.add_edge((y, x), (y + step, x + step), weight=step * np.sqrt(2), edge_type="grid")

        self._logger.info(f"Created grid graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return self._to_world_coordinates(graph)

    def _to_world_coordinates(self, graph: nx.Graph) -> nx.Graph:
        """Convert nodes from pixel coordinates to world coordinates.

        Takes a NetworkX graph with nodes in pixel coordinates (y,x) and converts them to world coordinates (x,y,z)
        using the current transform parameters (resolution, offset, rotation). Edge weights are scaled by resolution
        to convert from pixels to meters.
        """
        world_graph = nx.Graph()
        pixel_to_id_lookup = {}
        # Convert each node to world coordinates and store mapping
        for i, (node, data) in enumerate(graph.nodes(data=True)):
            # Modified: handles nodes as either tuple (row, col) or just int
            if isinstance(node, tuple):
                row, col = node
            else:
                row, col = data["pixel"]

            x_w, y_w = self._pixel_to_world(row, col)
            world_point_array = (x_w, y_w, 0.0)
            #world_point_array = tuple((world_point.x, world_point.y, world_point.z))
            pixel = (int(row), int(col))

            # Modified: fix for duplicated 'pixel'
            data_copy = data.copy()
            data_copy.pop("pixel", None)
            data_copy.pop("world", None)
            data_copy["pixel"] = pixel
            data_copy["world"] = world_point_array
            world_graph.add_node(i, **data_copy)
            pixel_to_id_lookup[node] = i

        # Convert edge weights from pixels to meters and copy edges
        for src, dst, data in graph.edges(data=True):
            edge_data = data.copy()
            if "weight" in edge_data:
                edge_data["weight"] = float(edge_data["weight"]) * self._resolution
            world_graph.add_edge(pixel_to_id_lookup[src], pixel_to_id_lookup[dst], **edge_data)

        return world_graph

    def _sample_obstacle_boundaries(self, graph: nx.Graph, sample_distance: float = 50) -> list[tuple[int, int]]:
        """Sample nodes along obstacle boundaries."""
        # Find contours of inflated obstacles
        contours = self._find_obstacle_contours(
            self._config.boundary_inflation_factor * self._safety_distance / self._resolution
        )

        initial_num_nodes = len(graph.nodes())

        # Process each contour
        for contour in contours:
            contour_nodes = []

            # Process each vertex in the contour
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]  # Wrap around to first point

                # Convert first point to y,x format
                row_1, col_1 = int(p1[1]), int(p1[0])
                contour_nodes.append((row_1, col_1))
                graph.add_node(
                    (row_1, col_1),
                    node_type="boundary",
                    pixel=(row_1, col_1),
                    world=self._pixel_to_world(row_1, col_1),
                )

                # Calculate distance to next vertex
                segment_length = np.linalg.norm(p2 - p1)
                # Add intermediate points
                num_intermediate = int(segment_length / sample_distance)
                # Skip first and last points
                intermediate_points = np.linspace(p1, p2, num=num_intermediate, endpoint=False).astype(int).tolist()[1:]
                for point in intermediate_points:
                    # Interpolate point
                    col, row = point
                    contour_nodes.append((row, col))
                    graph.add_node(
                        (row, col),
                        node_type="boundary",
                        pixel=(row, col),
                        world=self._pixel_to_world(row, col),
                    )

            # Connect consecutive nodes along this contour
            if len(contour_nodes) > 1:
                self._connect_contour_nodes(contour_nodes, graph)

        num_nodes_added = len(graph.nodes()) - initial_num_nodes
        self._logger.info(f"Added {num_nodes_added} nodes along obstacle boundaries")

    def _find_obstacle_contours(self, boundary_inflation: float) -> list[np.ndarray]:
        """Find contours of inflated obstacles using the distance map."""
        filtered_obstacles = (self._dist_transform >= boundary_inflation).astype(np.uint8)
        contours, _ = cv2.findContours(filtered_obstacles, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        return contours

    def _connect_contour_nodes(self, contour_nodes: list[tuple[int, int]], graph: nx.Graph):
        """Connect consecutive nodes along the contour."""
        for i in range(len(contour_nodes)):
            n1 = contour_nodes[i]
            n2 = contour_nodes[(i + 1) % len(contour_nodes)]

            if not self._check_line_collision(n1, n2):
                dist = np.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
                graph.add_edge(n1, n2, weight=dist, edge_type="contour")

    # ==========================================================
    # Free-space sampling
    # ==========================================================

    def _sample_free_space(self, graph, distance_threshold: float):
        """Iteratively sample free space in a map by identifying and adding nodes at local maxima
        of large distance areas until no such areas remain.
        """
        new_points: list[tuple[int, int]] = []
        base_map = self._inflated_map if self._inflated_map is not None else self._original_map
        free_mask = self._free_mask_from_map(base_map)

        H, W = free_mask.shape
        blocked_mask = free_mask.copy()
        kernel = np.ones((3, 3), np.uint8)

        max_iters = 12      # Hard stop to avoid runaway loops
        stall_limit = 2     # Stop if we fail to add nodes for these many iterations

        def _node_pixel(node, data):
            pixel = data.get("pixel")
            if pixel is None and isinstance(node, tuple) and len(node) == 2:
                pixel = node
            if pixel is None:
                return None
            row, col = int(pixel[0]), int(pixel[1])
            if 0 <= row < H and 0 <= col < W:
                return row, col
            return None

        idx = index.Index()
        idx_counter = 0

        def _add_to_index(row: int, col: int):
            nonlocal idx_counter
            idx.insert(idx_counter, (col, row, col, row))
            idx_counter += 1

        for node, data in graph.nodes(data=True):
            pix = _node_pixel(node, data)
            if pix is None:
                continue
            row, col = pix
            blocked_mask[row, col] = 0
            _add_to_index(row, col)

        iterations = 0
        stall_iters = 0
        prev_max_distance: float | None = None

        while True:
            iterations += 1

            # Distance transform + maxima detection
            distance_map = cv2.distanceTransform(blocked_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            max_distance = float(distance_map.max())
            large_distance_areas = distance_map > distance_threshold
            if not np.any(large_distance_areas):
                break
            dilated = cv2.dilate(distance_map, kernel)
            local_maxima_mask = (distance_map == dilated) & large_distance_areas
            local_maxima_coords = np.column_stack(np.where(local_maxima_mask))

            # Early convergence checks
            if max_distance <= distance_threshold:
                break
            if iterations >= max_iters:
                self._logger.warning(f"[WARN] Reached max iters ({max_iters}) in free-space sampler; stopping early.")
                break
            if prev_max_distance is not None and abs(max_distance - prev_max_distance) < 1e-3:
                self._logger.info("[INFO] Free-space sampler stalled (max distance plateau); stopping early.")
                break

            added = False
            half_threshold = distance_threshold / 2.0

            # Evaluate maxima with spacing constraints
            for row, col in local_maxima_coords:
                if blocked_mask[row, col] == 0:
                    continue

                # Reject if too close to an existing sample
                bounding_box = (
                    max(0, col - half_threshold),
                    max(0, row - half_threshold),
                    min(W - 1, col + half_threshold),
                    min(H - 1, row + half_threshold),
                )
                if list(idx.intersection(bounding_box)):
                    continue

                graph.add_node(
                    (row, col),
                    node_type="free_space",
                    pixel=(row, col),
                    world=self._pixel_to_world(row, col),
                )
                new_points.append((row, col))
                blocked_mask[row, col] = 0  # mark as occupied for next iteration
                _add_to_index(row, col)
                added = True

            # Convergence/stall guard
            if not added:
                stall_iters += 1
            else:
                stall_iters = 0
            if stall_iters >= stall_limit:
                break

            prev_max_distance = max_distance

        return new_points

    # ==========================================================
    # Graph topology refinement
    # ==========================================================

    def _check_line_collision(self, p0, p1):
        """Check if the line between two points intersects with any obstacles."""
        if self._inflated_map is None:
            return False

        # Handles world-coordinate inputs
        def ensure_pixel(pt):
            if all(isinstance(v, (int, np.integer)) for v in pt):
                return pt
            # Converts from world to pixel
            if isinstance(pt, np.ndarray):
                x, y = pt
            else:
                x, y = pt[0], pt[1]
            row, col = self._world_to_pixel(Point(x=float(x), y=float(y), z=0.0))
            return (int(row), int(col))

        p0_px = ensure_pixel(p0)
        p1_px = ensure_pixel(p1)
        y0, x0 = p0_px
        y1, x1 = p1_px

        # Bresenham's line algorithm to get the pixels the line passes through
        #y0, x0 = p0
        #y1, x1 = p1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self._inflated_map[y0, x0]:
                return True  # Return early if a collision is detected
            if (y0 == y1) and (x0 == x1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return False

    def _connect_free_nodes(self, graph: nx.Graph) -> None:
        """Connect nearby free-space/boundary nodes to reduce seams."""
        candidates: list = []
        coords: list[tuple[float, float]] = []

        for node, data in graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).lower()
            if node_type not in {"free_space", "boundary"}:
                continue

            world = data.get("world")
            if world is None:
                pix = data.get("pixel")
                if pix is None:
                    continue
                world = self._point_to_tuple(self._pixel_to_world(pix[0], pix[1]))
                data["world"] = world

            world_tuple = self._point_to_tuple(world)
            candidates.append(node)
            coords.append((float(world_tuple[0]), float(world_tuple[1])))

        if len(coords) < 2:
            return

        try:
            tree = cKDTree(coords)
        except Exception:
            return

        radius = max(self._config.free_space_sampling_threshold * 1.5, self._config.merge_node_distance * 2.0)
        pairs = tree.query_pairs(r=radius)

        for i, j in pairs:
            n1 = candidates[i]
            n2 = candidates[j]
            if n1 == n2:
                continue

            w1 = np.asarray(self._point_to_tuple(graph.nodes[n1]["world"])[:2], dtype=float)
            w2 = np.asarray(self._point_to_tuple(graph.nodes[n2]["world"])[:2], dtype=float)
            dist = float(np.linalg.norm(w2 - w1))
            if not np.isfinite(dist) or dist <= 0.0:
                continue

            existing = graph.get_edge_data(n1, n2)
            if existing:
                current = existing.get("weight")
                if current is not None and current <= dist:
                    continue
                existing["weight"] = dist
                existing["edge_type"] = existing.get("edge_type", "free_space")
            else:
                graph.add_edge(n1, n2, weight=dist, edge_type="free_space")

    def _add_delaunay_shortcuts(self, graph: nx.Graph):
        """Create shortcuts based on Delaunay triangulation of nodes."""

        self._logger.info("Adding Delaunay shortcuts (world-frame)...")

        node_worlds = []
        idx_to_node = []

        # Collect world positions for all nodes
        for n, data in graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                pix = data.get("pixel")
                if pix is not None:
                    y, x = pix
                    point = self._pixel_to_world(y, x)
                    world = self._point_to_tuple(point)
                    data["world"] = world

            if world is None:
                continue

            world = self._point_to_tuple(world)
            if np.isfinite(world[0]) and np.isfinite(world[1]):
                node_worlds.append((world[0], world[1]))
                idx_to_node.append(n)

        if len(node_worlds) < 3:
            self._logger.warning("[WARN] Not enough nodes for Delaunay triangulation — skipping.")
            return

        node_coords = np.array(node_worlds, dtype=float)

        try:
            tri = Delaunay(node_coords)
        except Exception as e:
            self._logger.error(f"Skipping Delaunay triangulation: {e}")
            return

        edge_candidates = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = idx_to_node[simplex[i]]
                b = idx_to_node[simplex[(i + 1) % 3]]
                edge_candidates.add(tuple(sorted((a, b))))

        # Add edges if free of collision
        added = 0
        max_delaunay_distance = max(
            self._config.free_space_sampling_threshold * 1.5,  # Reduced from 2.0
            self._config.merge_node_distance * 2.5,  # Reduced from 3.0
        )

        for n1, n2 in edge_candidates:
            try:
                w1 = np.asarray(self._point_to_tuple(graph.nodes[n1]["world"])[:2], dtype=float)
                w2 = np.asarray(self._point_to_tuple(graph.nodes[n2]["world"])[:2], dtype=float)
                dist = float(np.linalg.norm(w2 - w1))
                if not np.isfinite(dist) or dist <= 0.0 or dist > max_delaunay_distance:
                    continue

                # For collision check, project back into pixel space of this map
                p1 = graph.nodes[n1].get("pixel")
                p2 = graph.nodes[n2].get("pixel")
                if p1 is None:
                    p1 = self._world_to_pixel(Point(x=w1[0], y=w1[1], z=0))
                if p2 is None:
                    p2 = self._world_to_pixel(Point(x=w2[0], y=w2[1], z=0))
                if p1 is None or p2 is None:
                    continue

                if not self._check_line_collision(p1, p2):
                    existing = graph.get_edge_data(n1, n2)
                    if existing:
                        current = existing.get("weight")
                        if current is not None and current <= dist:
                            continue
                        existing["weight"] = dist
                        existing["edge_type"] = "delaunay"
                        added += 1
                    else:
                        graph.add_edge(n1, n2, weight=dist, edge_type="delaunay")
                        added += 1
            except Exception:
                continue

        self._logger.info(f"[INFO] Delaunay shortcuts added: {added}")

    def _remove_invalid_free_nodes(self, graph: nx.Graph) -> None:
        """Remove free-space nodes that are invalid under the inflated obstacle map."""
        if self._inflated_map is None:
            return

        height, width = self._inflated_map.shape
        to_remove: list = []
        for node, data in graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).lower()
            if node_type not in {"free_space"}:
                continue

            pix = data.get("pixel")
            if pix is None:
                continue
            row, col = int(pix[0]), int(pix[1])
            if not (0 <= row < height and 0 <= col < width):
                to_remove.append(node)
                continue
            if self._inflated_map[row, col] != 0:
                to_remove.append(node)

        if to_remove:
            graph.remove_nodes_from(to_remove)

    def _prune_graph(self, graph: nx.Graph):
        """Remove isolated nodes, small subgraphs, and merge close nodes."""
        self._logger.info("Pruning graph...")

        if len(graph.nodes()) > 0:
            # Merge nodes that are close to each other (threshold expressed in meters)
            self._merge_close_nodes(graph, threshold=self._config.merge_node_distance)

        # Remove isolated nodes and small subgraphs
        # graph.remove_nodes_from(list(nx.isolates(graph)))

        # Find connected components and filter small ones
        components = list(nx.connected_components(graph))
        for component in components:
            subgraph = graph.subgraph(component)
            total_length = sum(d["weight"] for _, _, d in subgraph.edges(data=True))
            if total_length < self._config.min_subgraph_length:
                graph.remove_nodes_from(component)

    def _merge_close_nodes(self, graph: nx.Graph, threshold: float):
        """Merge nodes that are within a certain distance threshold (meters)."""

        while True:
            # Collect positions of nodes
            node_positions = []
            node_ids = []

            for n, data in graph.nodes(data=True):
                pos = data.get("pos")
                if pos is None:
                    continue

                node_positions.append((pos[0], pos[1]))
                node_ids.append(n)

            if len(node_positions) < 2:
                break

            node_coords = np.asarray(node_positions, dtype=float)

            try:
                tree = cKDTree(node_coords)
            except Exception:
                break

            pairs = tree.query_pairs(r=threshold)

            if not pairs:
                break

            merged = False

            # Merging pass
            for i, j in pairs:
                n1 = node_ids[i]
                n2 = node_ids[j]

                if not graph.has_node(n1) or not graph.has_node(n2):
                    continue

                n1_pos = graph.nodes[n1].get("pos")
                n2_pos = graph.nodes[n2].get("pos")
                if n1_pos is None or n2_pos is None:
                    continue

                neighbors = [
                    n for n in graph.neighbors(n2)
                    if n != n1 and graph.has_node(n)
                ]

                collision = False
                for neighbor in neighbors:
                    neighbor_pos = graph.nodes[neighbor].get("pos")
                    if neighbor_pos is None:
                        continue
                    if self._check_line_collision(n1_pos, neighbor_pos):
                        collision = True
                        break

                if collision:
                    continue

                for neighbor in neighbors:
                    neighbor_pos = graph.nodes[neighbor].get("pos")
                    if neighbor_pos is None:
                        continue
                    dist = np.linalg.norm(
                        np.asarray(neighbor_pos) - np.asarray(n1_pos)
                    )
                    graph.add_edge(n1, neighbor, weight=dist, edge_type="merge")

                graph.remove_node(n2)
                merged = True

            if not merged:
                break

    # ==========================================================
    # Acceleration structures
    # ==========================================================

    @numba.njit
    def _flood_fill_numba(node_map, nodes, width):
        """Flood fill the node map with the node ids."""
        directions = [-width, width, -1, 1]
        queue = []

        for i, pixel in nodes:
            if 0 <= pixel < len(node_map):
                node_map[pixel] = i
                queue.append(pixel)

        for id in queue:
            for dd in directions:
                nid = id + dd
                # Check bounds and handle edge cases
                if 0 <= nid < len(node_map):
                    # Check if we're not wrapping around rows incorrectly
                    current_row = id // width
                    new_row = nid // width
                    if abs(new_row - current_row) <= 1 and node_map[nid] == -1:
                        node_map[nid] = node_map[id]
                        queue.append(nid)

    def _build_nearest_node_map(self, graph: nx.Graph):
        """Build a nearest node map from the graph for quick lookup."""
        self._logger.info("Building nearest node map...")

        height, width = self._original_map.shape
        self._node_map = np.full((height, width), -1, dtype=np.int32)
        self._node_map[self._original_map <= self._occupancy_threshold] = -2

        if len(graph) == 0:
            return

        cleaned_nodes = []

        for i, (node, pix) in enumerate(graph.nodes(data="pixel")):
            if pix is None:
                continue
            try:
                y, x = int(pix[0]), int(pix[1])
            except Exception:
                continue
            if 0 <= y < height and 0 <= x < width:
                cleaned_nodes.append((i, y * width + x))

        if not cleaned_nodes:
            self._logger.warning("[WARN] No valid pixel nodes for nearest-node map.")
            return

        graph_pixels = np.array(cleaned_nodes, dtype=np.int32)
        node_map_flat = self._node_map.reshape(-1)

        WaypointGraphGenerator._flood_fill_numba(
            node_map_flat,
            graph_pixels,
            width
        )

        self._logger.info("Nearest node map built")
