"""
Fractal Hierarchical Learning for Agentic Perception (Fractal-HLIP)

A novel RL system combining fractal environments with hierarchical attention mechanisms
inspired by HLIP for multi-scale perception and learning.
"""

from .environment import FractalDepthEnvironment
from .observation import construct_multi_scale_observation, visualize_observation
from .encoder import HierarchicalAttentionEncoder
from .agent import FractalHLIPAgent
from .networks import DQNNetwork
from .utils import ReplayBuffer, set_random_seeds, print_network_info, MetricsLogger

__version__ = "0.1.0"
__author__ = "Fractal-HLIP Research Team"

__all__ = [
    "FractalDepthEnvironment",
    "construct_multi_scale_observation", 
    "visualize_observation",
    "HierarchicalAttentionEncoder",
    "FractalHLIPAgent",
    "DQNNetwork",
    "ReplayBuffer",
    "set_random_seeds",
    "print_network_info",
    "MetricsLogger"
] 