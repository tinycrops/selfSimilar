"""
Fractal-HLIP Agent

This module implements the main FractalHLIPAgent class that combines the hierarchical
attention encoder with reinforcement learning algorithms to learn navigation and
problem-solving in the fractal environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import copy

from .environment import FractalDepthEnvironment
from .observation import construct_multi_scale_observation
from .encoder import HierarchicalAttentionEncoder
from .networks import DQNNetwork, DuelingDQNNetwork
from .utils import ReplayBuffer, EpsilonScheduler, MetricsLogger, dict_to_tensor

class FractalHLIPAgent:
    """
    Fractal-HLIP Agent that uses hierarchical attention for multi-scale perception
    and deep reinforcement learning for navigation in fractal environments.
    """
    
    def __init__(
        self,
        env: FractalDepthEnvironment,
        device: Optional[torch.device] = None,
        # Encoder parameters
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        # Network parameters
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_dueling: bool = True,
        # RL parameters
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        replay_buffer_size: int = 10000,
        target_update_freq: int = 1000,
        # Exploration parameters
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.01,
        epsilon_decay_steps: int = 10000,
        # Training parameters
        min_replay_size: int = 1000,
        gradient_clip: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the Fractal-HLIP Agent.
        
        Args:
            env: Fractal depth environment
            device: PyTorch device for computation
            embed_dim: Embedding dimension for attention encoder
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            hidden_dims: Hidden layer dimensions for RL networks
            use_dueling: Whether to use dueling DQN architecture
            learning_rate: Learning rate for optimization
            gamma: Discount factor for future rewards
            batch_size: Batch size for training
            replay_buffer_size: Size of experience replay buffer
            target_update_freq: Frequency of target network updates
            start_epsilon: Initial exploration rate
            end_epsilon: Final exploration rate
            epsilon_decay_steps: Steps over which to decay epsilon
            min_replay_size: Minimum replay buffer size before training
            gradient_clip: Gradient clipping threshold
            seed: Random seed for reproducibility
        """
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.gradient_clip = gradient_clip
        
        # Initialize hierarchical attention encoder
        self.encoder = HierarchicalAttentionEncoder(
            local_patch_size=5,
            depth_map_size=8,
            max_depth=env.max_depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)
        
        # Initialize Q-networks
        if use_dueling:
            self.q_network = DuelingDQNNetwork(
                feature_dim=embed_dim,
                action_space_size=env.action_space_size,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_q_network = DuelingDQNNetwork(
                feature_dim=embed_dim,
                action_space_size=env.action_space_size,
                hidden_dims=hidden_dims
            ).to(self.device)
        else:
            self.q_network = DQNNetwork(
                feature_dim=embed_dim,
                action_space_size=env.action_space_size,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_q_network = DQNNetwork(
                feature_dim=embed_dim,
                action_space_size=env.action_space_size,
                hidden_dims=hidden_dims
            ).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        # Initialize optimizer
        all_params = list(self.encoder.parameters()) + list(self.q_network.parameters())
        self.optimizer = optim.Adam(all_params, lr=learning_rate)
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size, seed=seed)
        
        # Initialize exploration scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
            decay_steps=epsilon_decay_steps
        )
        
        # Initialize metrics logger
        self.logger = MetricsLogger()
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.total_steps = 0
        
    def choose_action(
        self, 
        observation: Dict[str, np.ndarray], 
        epsilon: Optional[float] = None
    ) -> int:
        """
        Choose an action using epsilon-greedy policy with hierarchical attention.
        
        Args:
            observation: Multi-scale observation dictionary
            epsilon: Exploration rate (if None, uses current scheduled epsilon)
            
        Returns:
            Action index
        """
        if epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space_size)
        
        # Greedy action selection
        with torch.no_grad():
            # Convert observation to tensors
            obs_tensor = dict_to_tensor(observation, self.device)
            
            # Add batch dimension
            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.unsqueeze(0)
            
            # Forward pass through encoder and Q-network
            features = self.encoder(obs_tensor)
            q_values = self.q_network(features)
            
            # Select action with highest Q-value
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def learn(self) -> Optional[float]:
        """
        Perform one learning step using experience replay.
        
        Returns:
            Loss value if learning occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.min_replay_size):
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        obs_tensors = dict_to_tensor(batch['observations'], self.device)
        next_obs_tensors = dict_to_tensor(batch['next_observations'], self.device)
        actions = torch.from_numpy(batch['actions']).long().to(self.device)
        rewards = torch.from_numpy(batch['rewards']).float().to(self.device)
        dones = torch.from_numpy(batch['dones']).bool().to(self.device)
        
        # Current Q-values
        current_features = self.encoder(obs_tensors)
        current_q_values = self.q_network(current_features)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_features = self.encoder(next_obs_tensors)
            next_q_values = self.target_q_network(next_features)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            self.gradient_clip
        )
        
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.step_count += 1
        
        return loss.item()
    
    def train_episode(self, max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
        """
        Train the agent for one episode.
        
        Args:
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            Episode statistics dictionary
        """
        # Reset environment
        state = self.env.reset()
        
        # Get initial observation
        observation = construct_multi_scale_observation(self.env, state)
        
        # Episode tracking
        episode_reward = 0.0
        episode_steps = 0
        max_depth_reached = 0
        losses = []
        
        done = False
        while not done and episode_steps < max_steps:
            # Choose action
            epsilon = self.epsilon_scheduler.get_epsilon()
            action = self.choose_action(observation, epsilon)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            next_observation = construct_multi_scale_observation(self.env, next_state)
            
            # Store experience in replay buffer
            self.replay_buffer.push(
                observation, action, reward, next_observation, done
            )
            
            # Learn from experience
            loss = self.learn()
            if loss is not None:
                losses.append(loss)
            
            # Update state
            observation = next_observation
            episode_reward += reward
            episode_steps += 1
            max_depth_reached = max(max_depth_reached, info['new_depth'])
            
            # Update epsilon
            self.epsilon_scheduler.step()
            
            # Render if requested
            if render:
                print(self.env.render())
                print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
                print(f"Info: {info}")
                print("-" * 50)
        
        # Log episode metrics
        mean_loss = np.mean(losses) if losses else 0.0
        
        # Get attention weights for analysis (optional)
        attention_weights = None
        if len(losses) > 0:  # Only if we trained
            with torch.no_grad():
                obs_tensor = dict_to_tensor(observation, self.device)
                for key, tensor in obs_tensor.items():
                    obs_tensor[key] = tensor.unsqueeze(0)
                attention_weights = self.encoder.get_attention_weights(obs_tensor)
        
        # Log to metrics logger
        self.logger.log_episode(
            reward=episode_reward,
            steps=episode_steps,
            max_depth=max_depth_reached,
            epsilon=epsilon,
            loss=mean_loss,
            attention_weights=attention_weights
        )
        
        self.episode_count += 1
        self.total_steps += episode_steps
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'max_depth_reached': max_depth_reached,
            'mean_loss': mean_loss,
            'epsilon': epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the agent's performance without learning.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            
        Returns:
            Evaluation statistics
        """
        # Disable exploration
        old_epsilon = self.epsilon_scheduler.get_epsilon()
        eval_epsilon = 0.0  # Greedy policy
        
        episode_rewards = []
        episode_steps = []
        max_depths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            observation = construct_multi_scale_observation(self.env, state)
            
            episode_reward = 0.0
            steps = 0
            max_depth = 0
            done = False
            
            while not done and steps < 1000:
                action = self.choose_action(observation, eval_epsilon)
                next_state, reward, done, info = self.env.step(action)
                next_observation = construct_multi_scale_observation(self.env, next_state)
                
                observation = next_observation
                episode_reward += reward
                steps += 1
                max_depth = max(max_depth, info['new_depth'])
                
                if render:
                    print(f"Episode {episode + 1}, Step {steps}")
                    print(self.env.render())
                    print(f"Action: {action}, Reward: {reward:.2f}")
                    print("-" * 30)
            
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            max_depths.append(max_depth)
            
            # Check if goal was reached (positive final reward typically indicates success)
            if episode_reward > 0:
                success_rate += 1
        
        success_rate /= num_episodes
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'std_steps': np.std(episode_steps),
            'mean_max_depth': np.mean(max_depths),
            'success_rate': success_rate,
            'episode_rewards': episode_rewards
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon_scheduler_step': self.epsilon_scheduler.current_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon_scheduler.current_step = checkpoint['epsilon_scheduler_step']
        
        print(f"Model loaded from {filepath}")
    
    def get_attention_analysis(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Get detailed attention analysis for a given observation.
        
        Args:
            observation: Multi-scale observation dictionary
            
        Returns:
            Attention analysis dictionary
        """
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = dict_to_tensor(observation, self.device)
            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.unsqueeze(0)
            
            # Get attention weights
            attention_weights = self.encoder.get_attention_weights(obs_tensor)
            
            # Get Q-values
            features = self.encoder(obs_tensor)
            q_values = self.q_network(features)
            
            return {
                'attention_weights': attention_weights,
                'q_values': q_values.squeeze(0).cpu().numpy(),
                'selected_action': q_values.argmax(dim=1).item(),
                'feature_norm': torch.norm(features, dim=1).item()
            } 