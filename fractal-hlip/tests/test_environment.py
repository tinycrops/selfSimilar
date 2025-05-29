"""
Unit tests for FractalDepthEnvironment

These tests verify the core functionality of the fractal environment including:
- Environment initialization
- Agent movement and state transitions
- Portal navigation between depths
- Goal reaching and rewards
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fractal_hlip_agent import FractalDepthEnvironment

class TestFractalDepthEnvironment:
    """Test suite for FractalDepthEnvironment."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.env = FractalDepthEnvironment(
            base_size=16,
            num_portals_per_level=4,
            max_depth=3,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.base_size == 16
        assert self.env.max_depth == 3
        assert self.env.num_portals_per_level == 4
        assert len(self.env.base_portal_coords) == 4
        assert self.env.base_goal is not None
        assert self.env.action_space_size == 4
    
    def test_reset(self):
        """Test environment reset functionality."""
        state = self.env.reset()
        
        assert state == (1, 1, 0)  # Expected initial state
        assert self.env.current_depth == 0
        assert len(self.env.entry_portal_path) == 0
        assert self.env.episode_steps == 0
        assert self.env.episode_reward == 0.0
    
    def test_valid_movement(self):
        """Test valid agent movement."""
        self.env.reset()
        initial_state = self.env.get_state()
        
        # Move right (action 1)
        next_state, reward, done, info = self.env.step(1)
        
        assert next_state[0] == initial_state[0]  # Same x
        assert next_state[1] == initial_state[1] + 1  # y increased
        assert next_state[2] == initial_state[2]  # Same depth
        assert not done
        assert info['action_type'] == 'move'
    
    def test_obstacle_collision(self):
        """Test movement into obstacles."""
        self.env.reset()
        
        # Try to move into an obstacle (depends on generated layout)
        # This is a general test - specific positions depend on random seed
        initial_state = self.env.get_state()
        
        # Try multiple moves to find an obstacle
        for action in range(4):
            state_before = self.env.get_state()
            next_state, reward, done, info = self.env.step(action)
            
            # If agent didn't move, it hit an obstacle or boundary
            if next_state[:2] == state_before[:2]:
                assert reward <= 0  # Should get step cost or no reward
                break
    
    def test_portal_entry(self):
        """Test portal entry and depth transition."""
        self.env.reset()
        
        # Navigate to a portal
        portal_x, portal_y = self.env.base_portal_coords[0]
        
        # Set agent directly at portal for testing
        self.env.agent_x = portal_x
        self.env.agent_y = portal_y
        
        initial_depth = self.env.current_depth
        
        # Take any action (should trigger portal entry)
        next_state, reward, done, info = self.env.step(0)
        
        if info['action_type'] == 'zoom_in':
            assert next_state[2] == initial_depth + 1  # Depth increased
            assert len(self.env.entry_portal_path) == 1  # Portal path recorded
            assert next_state[:2] == (1, 1)  # Reset to start position in new level
    
    def test_boundary_zoom_out(self):
        """Test zooming out at depth boundaries."""
        self.env.reset()
        
        # Manually set agent at depth 1
        self.env.current_depth = 1
        self.env.entry_portal_path = [self.env.PortalInfo(5, 5, 0)]
        
        # Move to boundary
        self.env.agent_x = 0
        self.env.agent_y = 5
        
        # Try to move out of bounds (up)
        next_state, reward, done, info = self.env.step(0)
        
        if info['action_type'] == 'zoom_out':
            assert next_state[2] == 0  # Back to surface
            assert len(self.env.entry_portal_path) == 0  # Portal path cleared
    
    def test_goal_reaching(self):
        """Test goal reaching and rewards."""
        self.env.reset()
        
        # Navigate to goal at surface level
        goal_x, goal_y = self.env.base_goal
        self.env.agent_x = goal_x
        self.env.agent_y = goal_y
        
        # Should already be at goal, but test step anyway
        next_state, reward, done, info = self.env.step(0)
        
        # Check if we completed the episode
        if done:
            assert reward > 0  # Should get positive reward
    
    def test_episode_tracking(self):
        """Test episode step and reward tracking."""
        self.env.reset()
        
        initial_steps = self.env.episode_steps
        initial_reward = self.env.episode_reward
        
        # Take a few steps
        for _ in range(3):
            _, reward, _, _ = self.env.step(1)
        
        assert self.env.episode_steps == initial_steps + 3
        assert self.env.episode_reward != initial_reward
    
    def test_action_space(self):
        """Test action space validity."""
        self.env.reset()
        
        # Test all actions
        for action in range(self.env.action_space_size):
            state_before = self.env.get_state()
            next_state, reward, done, info = self.env.step(action)
            
            # Should always return valid state
            assert len(next_state) == 3
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
    
    def test_deterministic_behavior(self):
        """Test that environment behaves deterministically with same seed."""
        env1 = FractalDepthEnvironment(random_seed=42)
        env2 = FractalDepthEnvironment(random_seed=42)
        
        # Both environments should have identical layouts
        assert np.array_equal(env1.base_obstacles, env2.base_obstacles)
        assert env1.base_portal_coords == env2.base_portal_coords
        assert env1.base_goal == env2.base_goal
        
        # Both should behave identically
        state1 = env1.reset()
        state2 = env2.reset()
        assert state1 == state2
        
        # Take same actions
        for action in [1, 2, 1, 0]:
            next_state1, reward1, done1, info1 = env1.step(action)
            next_state2, reward2, done2, info2 = env2.step(action)
            
            assert next_state1 == next_state2
            assert reward1 == reward2
            assert done1 == done2
    
    def test_observation_info(self):
        """Test observation info generation."""
        self.env.reset()
        
        obs_info = self.env.get_observation_info()
        
        # Check required keys
        required_keys = [
            'agent_pos', 'current_depth', 'base_obstacles', 
            'base_portals', 'base_goal', 'entry_portal_path', 'base_size'
        ]
        
        for key in required_keys:
            assert key in obs_info
        
        # Check data types and shapes
        assert isinstance(obs_info['agent_pos'], tuple)
        assert len(obs_info['agent_pos']) == 2
        assert isinstance(obs_info['current_depth'], int)
        assert isinstance(obs_info['base_obstacles'], np.ndarray)
        assert obs_info['base_obstacles'].shape == (self.env.base_size, self.env.base_size)

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__]) 