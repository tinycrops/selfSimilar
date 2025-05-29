"""
FractalDepthEnvironment: A self-similar environment with nested depth levels.

The agent can navigate within levels and transition between depth levels through portals.
The environment maintains self-similarity across scales with configurable complexity.
"""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PortalInfo:
    """Information about a portal entrance/exit."""
    x: int
    y: int
    depth: int

class FractalDepthEnvironment:
    """
    A fractal environment where agents can navigate across self-similar nested scales.
    
    Key Features:
    - Self-similar structure at each depth level
    - Portal system for transitioning between depths
    - Configurable obstacles, goals, and portal positions
    - Reward structure encouraging multi-scale exploration
    """
    
    def __init__(
        self, 
        base_size: int = 16,
        num_portals_per_level: int = 4,
        max_depth: int = 3,
        step_cost: float = -0.01,
        portal_reward: float = 0.1,
        goal_reward: float = 1.0,
        sub_goal_reward: float = 0.5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the fractal environment.
        
        Args:
            base_size: Size of each level grid (base_size x base_size)
            num_portals_per_level: Number of portals at each level
            max_depth: Maximum depth level (0 = surface level)
            step_cost: Negative reward for each step (encourages efficiency)
            portal_reward: Reward for entering/exiting portals
            goal_reward: Reward for reaching main goal (depth 0)
            sub_goal_reward: Reward for reaching goals at depth > 0
            random_seed: Seed for reproducible environments
        """
        self.base_size = base_size
        self.num_portals_per_level = num_portals_per_level
        self.max_depth = max_depth
        self.step_cost = step_cost
        self.portal_reward = portal_reward
        self.goal_reward = goal_reward
        self.sub_goal_reward = sub_goal_reward
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate base layout (self-similar across all levels)
        self._generate_base_layout()
        
        # Agent state
        self.reset()
    
    def _generate_base_layout(self):
        """Generate the base layout that will be replicated at each depth level."""
        # Initialize empty grid
        self.base_obstacles = np.zeros((self.base_size, self.base_size), dtype=bool)
        
        # Add some obstacles (simple pattern for now)
        # Create a maze-like structure
        for i in range(2, self.base_size - 2, 4):
            for j in range(2, self.base_size - 2, 4):
                # Create cross-shaped obstacles
                self.base_obstacles[i-1:i+2, j] = True
                self.base_obstacles[i, j-1:j+2] = True
        
        # Ensure borders are clear for portal placement
        self.base_obstacles[0, :] = False
        self.base_obstacles[-1, :] = False
        self.base_obstacles[:, 0] = False
        self.base_obstacles[:, -1] = False
        self.base_obstacles[1, :] = False
        self.base_obstacles[-2, :] = False
        self.base_obstacles[:, 1] = False
        self.base_obstacles[:, -2] = False
        
        # Place portals around the perimeter
        self.base_portal_coords = []
        portal_positions = [
            (0, self.base_size // 4),           # Top
            (0, 3 * self.base_size // 4),       # Top  
            (self.base_size - 1, self.base_size // 4),     # Bottom
            (self.base_size - 1, 3 * self.base_size // 4), # Bottom
        ]
        
        for i, (x, y) in enumerate(portal_positions[:self.num_portals_per_level]):
            self.base_portal_coords.append((x, y))
            self.base_obstacles[x, y] = False  # Ensure portals are accessible
        
        # Place goal at center
        self.base_goal = (self.base_size // 2, self.base_size // 2)
        self.base_obstacles[self.base_goal] = False
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space_size = 4
        self.action_to_delta = {
            0: (-1, 0),  # up
            1: (0, 1),   # right  
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
    
    def reset(self) -> Tuple[int, int, int]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state (x, y, depth)
        """
        # Agent starts at (1, 1) at depth 0
        self.agent_x = 1
        self.agent_y = 1
        self.current_depth = 0
        
        # Track portal entry path for returning to parent levels
        self.entry_portal_path: List[PortalInfo] = []
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        return self.get_state()
    
    def get_state(self) -> Tuple[int, int, int]:
        """Get current agent state."""
        return (self.agent_x, self.agent_y, self.current_depth)
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (within bounds and not obstacle)."""
        if x < 0 or x >= self.base_size or y < 0 or y >= self.base_size:
            return False
        return not self.base_obstacles[x, y]
    
    def is_portal(self, x: int, y: int) -> bool:
        """Check if position contains a portal."""
        return (x, y) in self.base_portal_coords
    
    def is_goal(self, x: int, y: int) -> bool:
        """Check if position contains the goal."""
        return (x, y) == self.base_goal
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            next_state: New (x, y, depth) state
            reward: Immediate reward
            done: Whether episode is complete
            info: Additional information about the step
        """
        self.episode_steps += 1
        prev_depth = self.current_depth
        reward = self.step_cost
        done = False
        action_type = 'move'
        
        # Calculate new position
        dx, dy = self.action_to_delta[action]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        # Check bounds and handle edge cases
        if new_x < 0 or new_x >= self.base_size or new_y < 0 or new_y >= self.base_size:
            # Hit boundary - check for zoom out
            if self.current_depth > 0:
                # Zoom out to parent level
                action_type = 'zoom_out'
                reward += self.portal_reward
                
                # Return to parent portal position
                parent_portal = self.entry_portal_path.pop()
                self.agent_x = parent_portal.x
                self.agent_y = parent_portal.y
                self.current_depth = parent_portal.depth
            else:
                # At surface level, just block movement
                new_x = self.agent_x
                new_y = self.agent_y
                
        elif not self.is_valid_position(new_x, new_y):
            # Hit obstacle, block movement
            new_x = self.agent_x
            new_y = self.agent_y
            
        else:
            # Valid move
            self.agent_x = new_x
            self.agent_y = new_y
            
            # Check for portal entry
            if self.is_portal(new_x, new_y) and self.current_depth < self.max_depth:
                action_type = 'zoom_in'
                reward += self.portal_reward
                
                # Save current position for return path
                self.entry_portal_path.append(PortalInfo(new_x, new_y, self.current_depth))
                
                # Zoom into new level
                self.current_depth += 1
                self.agent_x = 1  # Start at (1,1) in new level
                self.agent_y = 1
            
            # Check for goal
            elif self.is_goal(new_x, new_y):
                if self.current_depth == 0:
                    # Main goal reached
                    reward += self.goal_reward
                    done = True
                else:
                    # Sub-goal reached
                    reward += self.sub_goal_reward
        
        self.episode_reward += reward
        
        # Create info dictionary
        info = {
            'action_type': action_type,
            'prev_depth': prev_depth,
            'new_depth': self.current_depth,
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'portal_path_length': len(self.entry_portal_path)
        }
        
        return self.get_state(), reward, done, info
    
    def render(self, mode: str = 'ascii') -> Optional[str]:
        """
        Render the current environment state.
        
        Args:
            mode: Rendering mode ('ascii' for text output)
            
        Returns:
            String representation if mode='ascii', None otherwise
        """
        if mode == 'ascii':
            grid = np.full((self.base_size, self.base_size), '.', dtype=str)
            
            # Mark obstacles
            grid[self.base_obstacles] = '#'
            
            # Mark portals
            for px, py in self.base_portal_coords:
                grid[px, py] = 'P'
            
            # Mark goal
            gx, gy = self.base_goal
            grid[gx, gy] = 'G'
            
            # Mark agent
            grid[self.agent_x, self.agent_y] = 'A'
            
            # Create output string
            output = f"Depth: {self.current_depth}, Steps: {self.episode_steps}, Reward: {self.episode_reward:.2f}\n"
            output += f"Portal Path: {len(self.entry_portal_path)} levels deep\n"
            output += "+" + "-" * self.base_size + "+\n"
            
            for row in grid:
                output += "|" + "".join(row) + "|\n"
                
            output += "+" + "-" * self.base_size + "+\n"
            output += "Legend: A=Agent, G=Goal, P=Portal, #=Obstacle, .=Empty\n"
            
            return output
        
        return None
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get information needed for constructing multi-scale observations.
        
        Returns:
            Dictionary with environment information for observation construction
        """
        return {
            'agent_pos': (self.agent_x, self.agent_y),
            'current_depth': self.current_depth,
            'base_obstacles': self.base_obstacles.copy(),
            'base_portals': self.base_portal_coords.copy(),
            'base_goal': self.base_goal,
            'entry_portal_path': self.entry_portal_path.copy(),
            'base_size': self.base_size
        } 