import math
from dataclasses import dataclass

import cupy as cp
import cupyx.scipy.ndimage as cp_ndi
import torch


@dataclass
class WaypointGraphGeneratorConfig:
    """Configuration for GPU waypoint graph generation.

    All distance parameters are in meters; pixel conversions happen internally.
    """
    boundary_inflation_factor: float = 1.5      # multiplier on safety_distance for obstacle dilation
    boundary_sample_distance: float = 2.5       # stride between boundary samples (m)
    free_space_sampling_threshold: float = 1.5  # max allowed dist-to-node for any free pixel (m)

    merge_node_distance: float = 0.25           # merge nodes closer than this (m)

    use_boundary_sampling: bool = True
    use_free_space_sampling: bool = True
    prune_graph: bool = True

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
    """
    GPU-native waypoint graph generator.
    
    Takes in occupancy grid as a CUDA tensor and returns the graph in the same layout as given by
    GlobalGraphGenerator
    
        pos : (N, 3)    float32
        node_types: (N,)    int32           all are free space here so 1.     
    
    
    Edges are formed later when the local nodes are added to the global graph through 
    GlobalGraphGenerator.add_local_graph.

    Occupancy grid conventions
    """
