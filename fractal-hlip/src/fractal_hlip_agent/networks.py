"""
Neural Networks for Fractal-HLIP Agent

This module defines the neural network architectures used by the RL agent,
including DQN and Actor-Critic networks that integrate with the hierarchical
attention encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DQNNetwork(nn.Module):
    """
    Deep Q-Network that uses hierarchical attention features for Q-value estimation.
    """
    
    def __init__(
        self, 
        feature_dim: int,
        action_space_size: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        """
        Initialize DQN network.
        
        Args:
            feature_dim: Dimension of input features from encoder
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_space_size = action_space_size
        
        # Build MLP layers
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer for Q-values
        layers.append(nn.Linear(input_dim, action_space_size))
        
        self.q_network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values.
        
        Args:
            features: Feature tensor from hierarchical encoder (batch, feature_dim)
            
        Returns:
            Q-values for each action (batch, action_space_size)
        """
        return self.q_network(features)

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_space_size: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        """
        Initialize Dueling DQN network.
        
        Args:
            feature_dim: Dimension of input features from encoder
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_space_size = action_space_size
        
        # Shared feature processing
        shared_layers = []
        input_dim = feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], action_space_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values using dueling architecture.
        
        Args:
            features: Feature tensor from hierarchical encoder (batch, feature_dim)
            
        Returns:
            Q-values for each action (batch, action_space_size)
        """
        # Shared processing
        shared_features = self.shared_layers(features)
        
        # Value and advantage streams
        value = self.value_stream(shared_features)  # (batch, 1)
        advantage = self.advantage_stream(shared_features)  # (batch, action_space_size)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ActorNetwork(nn.Module):
    """
    Actor network for Actor-Critic methods (policy network).
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_space_size: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        """
        Initialize Actor network.
        
        Args:
            feature_dim: Dimension of input features from encoder
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_space_size = action_space_size
        
        # Build MLP layers
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer for action probabilities
        layers.append(nn.Linear(input_dim, action_space_size))
        
        self.policy_network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.
        
        Args:
            features: Feature tensor from hierarchical encoder (batch, feature_dim)
            
        Returns:
            Action probabilities (batch, action_space_size)
        """
        logits = self.policy_network(features)
        return F.softmax(logits, dim=-1)
    
    def get_action_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get raw action logits (before softmax).
        
        Args:
            features: Feature tensor from hierarchical encoder
            
        Returns:
            Action logits (batch, action_space_size)
        """
        return self.policy_network(features)

class CriticNetwork(nn.Module):
    """
    Critic network for Actor-Critic methods (value function).
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        """
        Initialize Critic network.
        
        Args:
            feature_dim: Dimension of input features from encoder
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Build MLP layers
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(input_dim, 1))
        
        self.value_network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute state value.
        
        Args:
            features: Feature tensor from hierarchical encoder (batch, feature_dim)
            
        Returns:
            State value (batch, 1)
        """
        return self.value_network(features)

class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature processing.
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_space_size: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        """
        Initialize combined Actor-Critic network.
        
        Args:
            feature_dim: Dimension of input features from encoder
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_space_size = action_space_size
        
        # Shared feature processing
        shared_layers = []
        input_dim = feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], action_space_size)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute both policy and value.
        
        Args:
            features: Feature tensor from hierarchical encoder (batch, feature_dim)
            
        Returns:
            Tuple of (action_probabilities, state_value)
        """
        # Shared processing
        shared_features = self.shared_layers(features)
        
        # Actor and critic outputs
        action_logits = self.actor_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic_head(shared_features)
        
        return action_probs, state_value
    
    def get_action_logits_and_value(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw action logits and value (useful for computing losses).
        
        Args:
            features: Feature tensor from hierarchical encoder
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        shared_features = self.shared_layers(features)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_logits, state_value 