"""
HLIP-inspired Hierarchical Attention Encoder for Fractal-HLIP

This module implements a neural network encoder that processes multi-scale observations
using hierarchical attention mechanisms inspired by HLIP (Hierarchical Language-Image Pre-training).
The encoder processes local, current-depth, and parent-depth views to create a unified
representation for the RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class PatchEmbedding(nn.Module):
    """
    Convert spatial feature maps into sequence of patch embeddings.
    Similar to Vision Transformer patch embedding but for our grid-based observations.
    """
    
    def __init__(self, channels: int, patch_size: int, embed_dim: int):
        """
        Initialize patch embedding layer.
        
        Args:
            channels: Number of input channels
            patch_size: Size of spatial patches
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        
        # Convolutional layer to create patch embeddings
        self.conv = nn.Conv2d(
            channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        max_patches = 8 * 8  # Assume max 8x8 patch grid
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert spatial input to patch sequence.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # Extract patches
        patches = self.conv(x)  # (batch, embed_dim, h_patches, w_patches)
        batch_size, embed_dim, h_patches, w_patches = patches.shape
        
        # Flatten spatial dimensions
        patches = patches.view(batch_size, embed_dim, h_patches * w_patches)
        patches = patches.transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add positional embedding
        num_patches = patches.shape[1]
        if num_patches <= self.pos_embed.shape[1]:
            patches = patches + self.pos_embed[:, :num_patches, :]
        
        return patches

class LocalFeatureExtractor(nn.Module):
    """
    Extract features from local view using CNN layers.
    """
    
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Project to feature dimension
        self.projection = nn.Linear(64, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from local view.
        
        Args:
            x: Local view tensor (batch, channels, height, width)
            
        Returns:
            Feature vector (batch, feature_dim)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.projection(features)

class CrossScaleAttention(nn.Module):
    """
    Multi-head attention mechanism for attending across different observational scales.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-scale attention.
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)  
            value: Value tensor (batch, seq_len, embed_dim)
            
        Returns:
            Attended features (batch, seq_len, embed_dim)
        """
        # Multi-head attention
        attended, _ = self.multihead_attn(query, key, value)
        attended = self.norm1(query + attended)
        
        # Feedforward
        ff_out = self.ff(attended)
        output = self.norm2(attended + ff_out)
        
        return output

class HierarchicalAttentionEncoder(nn.Module):
    """
    HLIP-inspired hierarchical attention encoder for processing multi-scale observations.
    
    Architecture:
    1. Level 1: Extract features from each observational scale separately
    2. Level 2: Apply self-attention within spatial scales (for depth maps)
    3. Level 3: Apply cross-scale attention to integrate information across scales
    """
    
    def __init__(
        self,
        local_patch_size: int = 5,
        depth_map_size: int = 8,
        max_depth: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the hierarchical attention encoder.
        
        Args:
            local_patch_size: Size of local view patches
            depth_map_size: Size of depth maps
            max_depth: Maximum environment depth
            embed_dim: Embedding dimension for attention
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.local_patch_size = local_patch_size
        self.depth_map_size = depth_map_size
        self.max_depth = max_depth
        self.embed_dim = embed_dim
        
        # Level 1: Feature extractors for each scale
        self.local_extractor = LocalFeatureExtractor(4, embed_dim)
        
        # Patch embeddings for depth maps (treating them as image patches)
        self.current_depth_embedding = PatchEmbedding(4, 2, embed_dim)  # 2x2 patches from 8x8 maps
        self.parent_depth_embedding = PatchEmbedding(4, 2, embed_dim)
        
        # Depth context encoder
        depth_context_dim = max_depth + 3
        self.depth_context_encoder = nn.Sequential(
            nn.Linear(depth_context_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Level 2: Self-attention for spatial understanding within scales
        self.spatial_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Level 3: Cross-scale attention for integrating across scales
        self.cross_scale_attention = CrossScaleAttention(embed_dim, num_heads, dropout)
        
        # Final projection to create unified representation
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Scale type embeddings to distinguish different observational scales
        self.scale_embeddings = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hierarchical attention encoder.
        
        Args:
            observation: Dictionary containing multi-scale observation tensors:
                - 'local_view': (batch, 4, local_patch_size, local_patch_size)
                - 'current_depth_map': (batch, 4, depth_map_size, depth_map_size)
                - 'parent_depth_map': (batch, 4, depth_map_size, depth_map_size)
                - 'depth_context': (batch, max_depth + 3)
                
        Returns:
            Unified feature representation (batch, embed_dim)
        """
        batch_size = observation['local_view'].shape[0]
        
        # Level 1: Extract features from each scale
        # Local view - direct feature extraction
        local_features = self.local_extractor(observation['local_view'])  # (batch, embed_dim)
        local_features = local_features.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Current depth map - patch embeddings
        current_depth_patches = self.current_depth_embedding(observation['current_depth_map'])
        
        # Parent depth map - patch embeddings  
        parent_depth_patches = self.parent_depth_embedding(observation['parent_depth_map'])
        
        # Depth context - encode to feature space
        depth_context_features = self.depth_context_encoder(observation['depth_context'])
        depth_context_features = depth_context_features.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Level 2: Apply self-attention within spatial scales
        # Process current depth patches
        current_depth_attended = current_depth_patches
        for layer in self.spatial_attention_layers:
            current_depth_attended = layer(current_depth_attended)
        
        # Process parent depth patches
        parent_depth_attended = parent_depth_patches
        for layer in self.spatial_attention_layers:
            parent_depth_attended = layer(parent_depth_attended)
        
        # Aggregate spatial features (mean pooling over patches)
        current_depth_agg = current_depth_attended.mean(dim=1, keepdim=True)  # (batch, 1, embed_dim)
        parent_depth_agg = parent_depth_attended.mean(dim=1, keepdim=True)    # (batch, 1, embed_dim)
        
        # Level 3: Cross-scale attention integration
        # Create sequence of scale features
        scale_features = torch.cat([
            local_features,        # Scale 0: Local view
            current_depth_agg,     # Scale 1: Current depth
            parent_depth_agg,      # Scale 2: Parent depth  
            depth_context_features # Scale 3: Depth context
        ], dim=1)  # (batch, 4, embed_dim)
        
        # Add scale type embeddings
        scale_features = scale_features + self.scale_embeddings.unsqueeze(0)
        
        # Apply cross-scale attention (self-attention across scales)
        integrated_features = self.cross_scale_attention(
            scale_features, scale_features, scale_features
        )
        
        # Final aggregation and projection
        # Use weighted combination of scale features
        final_features = integrated_features.mean(dim=1)  # (batch, embed_dim)
        output = self.final_projection(final_features)
        
        return output
    
    def get_attention_weights(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for analysis and visualization.
        
        Args:
            observation: Multi-scale observation dictionary
            
        Returns:
            Dictionary containing attention weights from different levels
        """
        # This is a simplified version for analysis
        # In practice, you would need to modify the forward pass to capture attention weights
        with torch.no_grad():
            batch_size = observation['local_view'].shape[0]
            
            # Extract features (simplified)
            local_features = self.local_extractor(observation['local_view']).unsqueeze(1)
            current_depth_patches = self.current_depth_embedding(observation['current_depth_map'])
            parent_depth_patches = self.parent_depth_embedding(observation['parent_depth_map'])
            depth_context_features = self.depth_context_encoder(observation['depth_context']).unsqueeze(1)
            
            # Aggregate depth features
            current_depth_agg = current_depth_patches.mean(dim=1, keepdim=True)
            parent_depth_agg = parent_depth_patches.mean(dim=1, keepdim=True)
            
            # Create scale features
            scale_features = torch.cat([
                local_features, current_depth_agg, parent_depth_agg, depth_context_features
            ], dim=1)
            
            # Compute attention scores (simplified)
            query = scale_features
            key = scale_features
            scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.embed_dim)
            attention_weights = F.softmax(scores, dim=-1)
            
            return {
                'cross_scale_attention': attention_weights,
                'scale_feature_norms': torch.norm(scale_features, dim=-1)
            } 