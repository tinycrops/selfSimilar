"""
Utilities for Fractal-HLIP Agent

This module contains utility classes and functions including experience replay buffer,
logging utilities, and other helper functions for the RL agent.
"""

import torch
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Experience tuple for storing transitions
Experience = namedtuple('Experience', [
    'observation', 'action', 'reward', 'next_observation', 'done'
])

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    Handles multi-scale observations represented as dictionaries.
    """
    
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ):
        """
        Add a new experience to the buffer.
        
        Args:
            observation: Current multi-scale observation
            action: Action taken
            reward: Reward received
            next_observation: Next multi-scale observation
            done: Whether episode ended
        """
        experience = Experience(observation, action, reward, next_observation, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary containing batched experiences:
            - 'observations': Dict of observation arrays
            - 'actions': Action array
            - 'rewards': Reward array
            - 'next_observations': Dict of next observation arrays
            - 'dones': Done flags array
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} experiences, but {batch_size} requested")
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Separate the experiences
        observations = [e.observation for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_observations = [e.next_observation for e in experiences]
        dones = [e.done for e in experiences]
        
        # Batch the observations (convert list of dicts to dict of batched arrays)
        batched_obs = self._batch_observations(observations)
        batched_next_obs = self._batch_observations(next_observations)
        
        return {
            'observations': batched_obs,
            'actions': np.array(actions),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_observations': batched_next_obs,
            'dones': np.array(dones, dtype=bool)
        }
    
    def _batch_observations(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Convert list of observation dictionaries to dictionary of batched arrays.
        
        Args:
            obs_list: List of observation dictionaries
            
        Returns:
            Dictionary with batched observation arrays
        """
        batched = {}
        
        # Get keys from first observation
        keys = obs_list[0].keys()
        
        for key in keys:
            # Stack arrays for this key
            arrays = [obs[key] for obs in obs_list]
            batched[key] = np.stack(arrays, axis=0)
        
        return batched
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= min_size

class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler for DQN-style agents.
    """
    
    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.01,
        decay_steps: int = 10000,
        decay_type: str = 'linear'
    ):
        """
        Initialize epsilon scheduler.
        
        Args:
            start_epsilon: Initial exploration rate
            end_epsilon: Final exploration rate
            decay_steps: Number of steps to decay over
            decay_type: Type of decay ('linear' or 'exponential')
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0
        
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.current_step >= self.decay_steps:
            return self.end_epsilon
        
        if self.decay_type == 'linear':
            # Linear decay
            epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (
                self.current_step / self.decay_steps
            )
        elif self.decay_type == 'exponential':
            # Exponential decay
            decay_rate = (self.end_epsilon / self.start_epsilon) ** (1 / self.decay_steps)
            epsilon = self.start_epsilon * (decay_rate ** self.current_step)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return max(epsilon, self.end_epsilon)
    
    def step(self):
        """Update epsilon for next step."""
        self.current_step += 1

class MetricsLogger:
    """
    Logger for tracking training metrics and statistics.
    """
    
    def __init__(self):
        """Initialize metrics logger."""
        self.metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_max_depth': [],
            'loss_values': [],
            'epsilon_values': [],
            'q_value_means': [],
            'attention_stats': []
        }
        
    def log_episode(
        self,
        reward: float,
        steps: int,
        max_depth: int,
        epsilon: float = None,
        loss: float = None,
        q_values: torch.Tensor = None,
        attention_weights: Dict[str, torch.Tensor] = None
    ):
        """
        Log metrics for a completed episode.
        
        Args:
            reward: Total episode reward
            steps: Number of steps in episode
            max_depth: Maximum depth reached in episode
            epsilon: Current exploration rate
            loss: Training loss (if available)
            q_values: Q-values from last step (if available)
            attention_weights: Attention weights for analysis (if available)
        """
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_steps'].append(steps)
        self.metrics['episode_max_depth'].append(max_depth)
        
        if epsilon is not None:
            self.metrics['epsilon_values'].append(epsilon)
            
        if loss is not None:
            self.metrics['loss_values'].append(loss)
            
        if q_values is not None:
            mean_q = float(torch.mean(q_values).item())
            self.metrics['q_value_means'].append(mean_q)
            
        if attention_weights is not None:
            # Store simplified attention statistics
            attention_stats = {}
            for key, weights in attention_weights.items():
                if torch.is_tensor(weights):
                    attention_stats[key] = {
                        'mean': float(torch.mean(weights).item()),
                        'std': float(torch.std(weights).item())
                    }
            self.metrics['attention_stats'].append(attention_stats)
    
    def get_recent_mean(self, metric: str, window: int = 100) -> float:
        """
        Get mean of recent values for a metric.
        
        Args:
            metric: Name of metric
            window: Number of recent values to average
            
        Returns:
            Mean of recent values
        """
        if metric not in self.metrics:
            return 0.0
        
        values = self.metrics[metric]
        if len(values) == 0:
            return 0.0
        
        recent_values = values[-window:]
        return np.mean(recent_values)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves for analysis.
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        if self.metrics['episode_rewards']:
            axes[0, 0].plot(self.metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Episode steps
        if self.metrics['episode_steps']:
            axes[0, 1].plot(self.metrics['episode_steps'])
            axes[0, 1].set_title('Episode Steps')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Max depth reached
        if self.metrics['episode_max_depth']:
            axes[1, 0].plot(self.metrics['episode_max_depth'])
            axes[1, 0].set_title('Max Depth Reached')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Depth')
        
        # Loss values
        if self.metrics['loss_values']:
            axes[1, 1].plot(self.metrics['loss_values'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def dict_to_tensor(obs_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Convert observation dictionary from numpy arrays to PyTorch tensors.
    
    Args:
        obs_dict: Dictionary of numpy observation arrays
        device: PyTorch device to place tensors on
        
    Returns:
        Dictionary of PyTorch tensors
    """
    tensor_dict = {}
    for key, array in obs_dict.items():
        tensor_dict[key] = torch.from_numpy(array).float().to(device)
    return tensor_dict

def tensor_to_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Convert observation dictionary from PyTorch tensors to numpy arrays.
    
    Args:
        tensor_dict: Dictionary of PyTorch tensors
        
    Returns:
        Dictionary of numpy arrays
    """
    numpy_dict = {}
    for key, tensor in tensor_dict.items():
        numpy_dict[key] = tensor.detach().cpu().numpy()
    return numpy_dict

def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_network_info(network: torch.nn.Module, name: str = "Network"):
    """
    Print information about a neural network.
    
    Args:
        network: PyTorch neural network
        name: Name to display
    """
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\n{name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture:")
    for i, (name, module) in enumerate(network.named_modules()):
        if len(list(module.children())) == 0:  # Leaf modules only
            print(f"    {name}: {module}")

def compute_environment_statistics(env, num_episodes: int = 100) -> Dict[str, Any]:
    """
    Compute statistics about the environment for analysis.
    
    Args:
        env: Environment instance
        num_episodes: Number of episodes to run for statistics
        
    Returns:
        Dictionary of environment statistics
    """
    episode_lengths = []
    max_depths = []
    portal_entries = []
    goal_reaches = []
    
    for _ in range(num_episodes):
        env.reset()
        steps = 0
        max_depth = 0
        portal_count = 0
        reached_goal = False
        
        done = False
        while not done and steps < 1000:  # Safety limit
            action = np.random.randint(env.action_space_size)
            _, _, done, info = env.step(action)
            steps += 1
            
            if info['action_type'] == 'zoom_in':
                portal_count += 1
            
            max_depth = max(max_depth, info['new_depth'])
            
            if done:
                reached_goal = True
        
        episode_lengths.append(steps)
        max_depths.append(max_depth)
        portal_entries.append(portal_count)
        goal_reaches.append(reached_goal)
    
    return {
        'mean_episode_length': np.mean(episode_lengths),
        'mean_max_depth': np.mean(max_depths),
        'mean_portal_entries': np.mean(portal_entries),
        'goal_reach_rate': np.mean(goal_reaches),
        'episode_length_std': np.std(episode_lengths),
        'max_depth_distribution': np.bincount(max_depths)
    } 