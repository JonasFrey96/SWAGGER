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

        self._node_map: np.ndarray | None = None

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    # ==========================================================
    # API
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
            rotation: Rotation about the z axis (radians)

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
        self._y_offset = y_offset

        self._cos_rot = np.cos(rotation)
        self._sin_rot = np.sin(rotation)
        self._node_map = None

        # --------------------------------------------------------
        # Check for trivial case: completely free map
        # --------------------------------------------------------

        free_map = (self._original_map > self._occupancy_threshold).astype(np.uint8)

        if np.all(free_map):
            self._graph = self._create_grid_graph(
                image.shape,
                grid_sample_distance=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )
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

        if self._config.use_free_space_sampling:
            self._sample_free_space(
                graph,
                distance_threshold=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )

        # --------------------------------------------------------
        # Remove free-space samples on inflated obstacles/edges
        # --------------------------------------------------------

        self._remove_invalid_free_nodes(graph)

        # ------------------------------------------------------------
        # Enforce world/pixel consistency for all nodes
        # ------------------------------------------------------------

        remove_nodes = []

        for node, data in graph.nodes(data=True):
            if "pixel" not in data and "world" not in data:
                remove_nodes.append(node)
                continue
            if "pixel" not in data and "world" in data:
                world_tuple = self._point_to_tuple(data["world"])
                x_w, y_w = world_tuple[0], world_tuple[1]
                y_p, x_p = self._world_to_pixel(Point(x=x_w, y=y_w, z=0.0))
                data["pixel"] = (y_p, x_p)
                data["world"] = (float(x_w), float(y_w), float(world_tuple[2]))
            elif "world" not in data and "pixel" in data:
                y_p, x_p = data["pixel"]
                x_w, y_w = self._pixel_to_world(y_p, x_p)
                data["world"] = (float(x_w), float(y_w), 0.0)

        if remove_nodes:
            graph.remove_nodes_from(remove_nodes)

        if len(graph.nodes) == 0:
            self._logger.warning("[WARN] Graph has no nodes after world/pixel consistency check.")
        else:
            self._logger.info(f"[INFO] Graph consistency OK: {len(graph.nodes)} nodes.")

        # --------------------------------------------------------
        # Prune nodes and save graph topology
        # --------------------------------------------------------

        if self._config.prune_graph:
            self._prune_graph(graph, threshold=self._config.merge_node_distance)

        self._graph = graph
        self._logger.info(f"Final graph has {len(graph.nodes)} nodes and {len(graph.edges)} unique edges")

        return self._graph

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

    def _distance_transform(self, free_map: np.ndarray):
        """Inflate obstacles in the occupancy grid using a distance transform."""
        free_map = np.pad(free_map, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        self._dist_transform = cv2.distanceTransform(free_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self._inflated_map = (self._dist_transform < self._safety_distance / self._resolution).astype(np.uint8)
        self._inflated_map = self._inflated_map[1:-1, 1:-1]

    def _free_mask_from_map(self, occupancy: np.ndarray) -> np.ndarray:
        """Return binary mask of free space (1 = free)."""
        if occupancy is None:
            raise RuntimeError("Occupancy map is not initialized.")
        if occupancy.max() <= 1:
            return (occupancy == 0).astype(np.uint8)
        return (occupancy > self._occupancy_threshold).astype(np.uint8)

    def _is_free_pixel(self, row: int, col: int) -> bool:
        """Check whether a pixel lies within map bounds and is free of inflated obstacles."""
        if self._inflated_map is None:
            return True
        h, w = self._inflated_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self._inflated_map[row, col] == 0

    # ==========================================================
    # Graph construction primitives
    # ==========================================================

    def _create_grid_graph(self, shape: tuple[int, int], grid_sample_distance: int) -> nx.Graph:
        """Create a grid graph for completely free maps with a margin from the borders."""
        self._logger.info("Creating grid for completely free map with margin...")
        height, width = shape
        margin = int(self._safety_distance / self._resolution)

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

        self._logger.info(f"Created grid graph with {len(graph.nodes)} nodes")

        # Assign world/pixel attributes
        for node, data in graph.nodes(data=True):
            if isinstance(node, tuple):
                row, col = node
            else:
                pix = data.get("pixel")
                if pix is None:
                    continue
                row, col = pix

            x_w, y_w = self._pixel_to_world(row, col)

            data["pixel"] = (int(row), int(col))
            data["world"] = (float(x_w), float(y_w), 0.0)

        return graph

    def _find_obstacle_contours(self, boundary_inflation: float) -> list[np.ndarray]:
        """Find contours of inflated obstacles using the distance map."""
        filtered_obstacles = (self._dist_transform >= boundary_inflation).astype(np.uint8)
        contours, _ = cv2.findContours(filtered_obstacles, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        return contours

    def _sample_obstacle_boundaries(self, graph: nx.Graph, sample_distance: float = 50) -> None:
        """Sample nodes along obstacle boundaries."""
        contours = self._find_obstacle_contours(
            self._config.boundary_inflation_factor * self._safety_distance / self._resolution
        )

        initial_num_nodes = len(graph.nodes())

        for contour in contours:
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]

                row_1, col_1 = int(p1[1]), int(p1[0])
                graph.add_node(
                    (row_1, col_1),
                    pixel=(row_1, col_1),
                    world=(*self._pixel_to_world(row_1, col_1), 0.0),
                )

                segment_length = np.linalg.norm(p2 - p1)
                num_intermediate = int(segment_length / sample_distance)
                intermediate_points = np.linspace(p1, p2, num=num_intermediate, endpoint=False).astype(int).tolist()[1:]
                for point in intermediate_points:
                    col, row = point
                    graph.add_node(
                        (row, col),
                        pixel=(row, col),
                        world=(*self._pixel_to_world(row, col), 0.0),
                    )

        num_nodes_added = len(graph.nodes()) - initial_num_nodes
        self._logger.info(f"Added {num_nodes_added} nodes along obstacle boundaries")

    def _sample_free_space(self, graph, distance_threshold: float) -> None:
        """Iteratively sample free space in a map by identifying and adding nodes at local maxima
        of large distance areas until no such areas remain."""
        base_map = self._inflated_map if self._inflated_map is not None else self._original_map
        free_mask = self._free_mask_from_map(base_map)

        H, W = free_mask.shape
        blocked_mask = free_mask.copy()
        kernel = np.ones((3, 3), np.uint8)

        max_iters = 12      # Hard stop to avoid runaway loops
        stall_limit = 2     # Stop if we fail to add nodes for these many iterations

        idx = index.Index()
        idx_counter = 0

        # marks existing nodes as occupied in blocked_mask
        for node, data in graph.nodes(data=True):
            pixel = data.get("pixel")
            if pixel is None and isinstance(node, tuple) and len(node) == 2:
                pixel = node
            if pixel is None:
                continue
            row, col = int(pixel[0]), int(pixel[1])
            if not (0 <= row < H and 0 <= col < W):
                continue
            blocked_mask[row, col] = 0
            idx.insert(idx_counter, (col, row, col, row))
            idx_counter += 1

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

            # Apply spacing constraints to maxima
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
                    pixel=(row, col),
                    world=(*self._pixel_to_world(row, col), 0.0),
                )
                blocked_mask[row, col] = 0  # mark as occupied for next iteration
                idx.insert(idx_counter, (col, row, col, row))
                idx_counter += 1
                added = True

            # Convergence/stall guard
            if not added:
                stall_iters += 1
            else:
                stall_iters = 0
            if stall_iters >= stall_limit:
                break

            prev_max_distance = max_distance

    # ==========================================================
    # Graph topology refinement
    # ==========================================================

    def _remove_invalid_free_nodes(self, graph: nx.Graph) -> None:
        """Safety feature to remove free-space nodes that are invalid under the inflated obstacle map."""
        if self._inflated_map is None:
            return

        height, width = self._inflated_map.shape
        to_remove: list = []
        for node, data in graph.nodes(data=True):
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

    def _prune_graph(self, graph: nx.Graph, threshold: float) -> None:
        """Merge nodes that are within a certain distance threshold (meters)."""
        while True:
            # Collect positions of nodes
            node_positions = []
            node_ids = []

            for n, data in graph.nodes(data=True):
                world = data.get("world")
                if world is None:
                    continue
                world_tuple = self._point_to_tuple(world)
                node_positions.append((world_tuple[0], world_tuple[1]))
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

                graph.remove_node(n2)
                merged = True

            if not merged:
                break
