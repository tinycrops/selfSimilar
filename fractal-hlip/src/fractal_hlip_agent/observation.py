"""
Multi-Scale Observation Construction for Fractal-HLIP

This module constructs multi-scale observations from the FractalDepthEnvironment,
providing the agent with local, current-depth, and parent-depth views for
hierarchical attention processing.
"""

import numpy as np
from typing import Dict, Tuple, List, Any
from .environment import FractalDepthEnvironment, PortalInfo

def construct_multi_scale_observation(
    env: FractalDepthEnvironment, 
    current_agent_state: Tuple[int, int, int],
    local_patch_size: int = 5,
    depth_map_size: int = 8
) -> Dict[str, np.ndarray]:
    """
    Construct multi-scale observation from the environment state.
    
    Args:
        env: The fractal environment instance
        current_agent_state: Current (x, y, depth) state
        local_patch_size: Size of local view patch around agent
        depth_map_size: Size of downscaled depth maps
        
    Returns:
        Dictionary containing multi-scale observation arrays:
        - 'local_view': Local patch around agent
        - 'current_depth_map': Downscaled view of entire current depth
        - 'parent_depth_map': Downscaled view of parent depth (if exists)
        - 'depth_context': Encoded depth information
    """
    agent_x, agent_y, agent_depth = current_agent_state
    obs_info = env.get_observation_info()
    
    # Construct local view
    local_view = _construct_local_view(
        obs_info, agent_x, agent_y, local_patch_size
    )
    
    # Construct current depth map
    current_depth_map = _construct_depth_map(
        obs_info, agent_x, agent_y, agent_depth, depth_map_size
    )
    
    # Construct parent depth map if not at surface level
    if agent_depth > 0 and len(obs_info['entry_portal_path']) > 0:
        parent_depth_map = _construct_parent_depth_map(
            obs_info, depth_map_size
        )
    else:
        # No parent - use empty map with 4 channels
        parent_depth_map = np.zeros((4, depth_map_size, depth_map_size), dtype=np.float32)
    
    # Construct depth context encoding
    depth_context = _construct_depth_context(
        agent_depth, len(obs_info['entry_portal_path']), env.max_depth
    )
    
    return {
        'local_view': local_view.astype(np.float32),
        'current_depth_map': current_depth_map.astype(np.float32),
        'parent_depth_map': parent_depth_map.astype(np.float32),
        'depth_context': depth_context.astype(np.float32)
    }

def _construct_local_view(
    obs_info: Dict[str, Any], 
    agent_x: int, 
    agent_y: int, 
    patch_size: int
) -> np.ndarray:
    """
    Construct a local view patch around the agent.
    
    Args:
        obs_info: Environment observation info
        agent_x, agent_y: Agent position
        patch_size: Size of the square patch
        
    Returns:
        Local view array with channel encoding:
        - Channel 0: Empty space (1.0) or obstacle (0.0)
        - Channel 1: Portal presence (1.0 if portal, 0.0 otherwise)
        - Channel 2: Goal presence (1.0 if goal, 0.0 otherwise)  
        - Channel 3: Agent presence (1.0 at agent position, 0.0 otherwise)
    """
    base_size = obs_info['base_size']
    base_obstacles = obs_info['base_obstacles']
    base_portals = obs_info['base_portals']
    base_goal = obs_info['base_goal']
    
    # Create multi-channel local view
    local_view = np.zeros((4, patch_size, patch_size), dtype=np.float32)
    
    # Calculate patch boundaries
    half_patch = patch_size // 2
    start_x = agent_x - half_patch
    start_y = agent_y - half_patch
    
    for i in range(patch_size):
        for j in range(patch_size):
            world_x = start_x + i
            world_y = start_y + j
            
            # Check bounds
            if 0 <= world_x < base_size and 0 <= world_y < base_size:
                # Channel 0: Empty space (inverted obstacles)
                local_view[0, i, j] = 0.0 if base_obstacles[world_x, world_y] else 1.0
                
                # Channel 1: Portals
                if (world_x, world_y) in base_portals:
                    local_view[1, i, j] = 1.0
                
                # Channel 2: Goal
                if (world_x, world_y) == base_goal:
                    local_view[2, i, j] = 1.0
                
                # Channel 3: Agent
                if world_x == agent_x and world_y == agent_y:
                    local_view[3, i, j] = 1.0
            else:
                # Out of bounds - treat as obstacle
                local_view[0, i, j] = 0.0
    
    return local_view

def _construct_depth_map(
    obs_info: Dict[str, Any],
    agent_x: int,
    agent_y: int, 
    agent_depth: int,
    map_size: int
) -> np.ndarray:
    """
    Construct a downscaled map of the entire current depth level.
    
    Args:
        obs_info: Environment observation info
        agent_x, agent_y: Agent position
        agent_depth: Current depth
        map_size: Size of the output map
        
    Returns:
        Downscaled depth map with same channel encoding as local view
    """
    base_size = obs_info['base_size']
    base_obstacles = obs_info['base_obstacles']
    base_portals = obs_info['base_portals']
    base_goal = obs_info['base_goal']
    
    # Create multi-channel depth map
    depth_map = np.zeros((4, map_size, map_size), dtype=np.float32)
    
    # Scale factor for downsampling
    scale_factor = base_size / map_size
    
    for i in range(map_size):
        for j in range(map_size):
            # Calculate corresponding region in base grid
            start_x = int(i * scale_factor)
            end_x = int((i + 1) * scale_factor)
            start_y = int(j * scale_factor)
            end_y = int((j + 1) * scale_factor)
            
            # Ensure bounds
            end_x = min(end_x, base_size)
            end_y = min(end_y, base_size)
            
            # Sample region
            region_obstacles = base_obstacles[start_x:end_x, start_y:end_y]
            
            # Channel 0: Empty space (proportion of non-obstacle cells)
            if region_obstacles.size > 0:
                depth_map[0, i, j] = 1.0 - np.mean(region_obstacles)
            
            # Channel 1: Portals (check if any portal in region)
            for px, py in base_portals:
                if start_x <= px < end_x and start_y <= py < end_y:
                    depth_map[1, i, j] = 1.0
                    break
            
            # Channel 2: Goal (check if goal in region)
            gx, gy = base_goal
            if start_x <= gx < end_x and start_y <= gy < end_y:
                depth_map[2, i, j] = 1.0
            
            # Channel 3: Agent (check if agent in region)
            if start_x <= agent_x < end_x and start_y <= agent_y < end_y:
                depth_map[3, i, j] = 1.0
    
    return depth_map

def _construct_parent_depth_map(
    obs_info: Dict[str, Any],
    map_size: int
) -> np.ndarray:
    """
    Construct a downscaled map showing context from parent depth level.
    
    Args:
        obs_info: Environment observation info
        map_size: Size of the output map
        
    Returns:
        Parent depth context map focusing on the area around the entry portal
    """
    entry_portal_path = obs_info['entry_portal_path']
    base_obstacles = obs_info['base_obstacles']
    base_portals = obs_info['base_portals']
    base_goal = obs_info['base_goal']
    base_size = obs_info['base_size']
    
    # Get the most recent parent portal (where we came from)
    if len(entry_portal_path) == 0:
        return np.zeros((4, map_size, map_size), dtype=np.float32)
    
    parent_portal = entry_portal_path[-1]
    parent_x, parent_y = parent_portal.x, parent_portal.y
    
    # Create parent context map focused around the entry portal
    parent_map = np.zeros((4, map_size, map_size), dtype=np.float32)
    
    # Define region around parent portal
    context_radius = base_size // 4  # Show quarter of the parent level
    start_x = max(0, parent_x - context_radius)
    end_x = min(base_size, parent_x + context_radius)
    start_y = max(0, parent_y - context_radius)  
    end_y = min(base_size, parent_y + context_radius)
    
    # Scale this region to fit in map_size
    region_width = end_x - start_x
    region_height = end_y - start_y
    
    for i in range(map_size):
        for j in range(map_size):
            # Map from output coordinates to parent region coordinates
            src_x = start_x + int(i * region_width / map_size)
            src_y = start_y + int(j * region_height / map_size)
            
            if src_x < base_size and src_y < base_size:
                # Channel 0: Empty space
                parent_map[0, i, j] = 0.0 if base_obstacles[src_x, src_y] else 1.0
                
                # Channel 1: Portals
                if (src_x, src_y) in base_portals:
                    parent_map[1, i, j] = 1.0
                
                # Channel 2: Goal
                if (src_x, src_y) == base_goal:
                    parent_map[2, i, j] = 1.0
                
                # Channel 3: Entry portal (highlight where we came from)
                if src_x == parent_x and src_y == parent_y:
                    parent_map[3, i, j] = 1.0
    
    return parent_map

def _construct_depth_context(
    current_depth: int, 
    portal_path_length: int, 
    max_depth: int
) -> np.ndarray:
    """
    Construct an encoding of depth context information.
    
    Args:
        current_depth: Current depth level
        portal_path_length: Length of portal entry path
        max_depth: Maximum possible depth
        
    Returns:
        1D array encoding depth context information
    """
    context = np.zeros(max_depth + 3, dtype=np.float32)
    
    # One-hot encode current depth
    if current_depth < max_depth + 1:
        context[current_depth] = 1.0
    
    # Encode normalized depth
    context[max_depth + 1] = current_depth / max_depth
    
    # Encode portal path length (how deep we've gone)
    context[max_depth + 2] = portal_path_length / max_depth
    
    return context

def visualize_observation(observation: Dict[str, np.ndarray]) -> str:
    """
    Create a string visualization of the multi-scale observation.
    
    Args:
        observation: Multi-scale observation dictionary
        
    Returns:
        String representation for debugging/analysis
    """
    output = "=== Multi-Scale Observation ===\n\n"
    
    # Local view
    local = observation['local_view']
    output += "Local View (Channels: Empty, Portal, Goal, Agent):\n"
    for c in range(local.shape[0]):
        output += f"Channel {c}:\n"
        for i in range(local.shape[1]):
            row = ""
            for j in range(local.shape[2]):
                val = local[c, i, j]
                if val > 0.5:
                    row += "█"
                elif val > 0.0:
                    row += "▓"
                else:
                    row += "░"
            output += row + "\n"
        output += "\n"
    
    # Depth context
    depth_ctx = observation['depth_context']
    output += f"Depth Context: {depth_ctx}\n\n"
    
    output += "Current Depth Map and Parent Depth Map have similar channel structure.\n"
    
    return output 