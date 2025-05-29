#!/usr/bin/env python3
"""
Fractal-HLIP Interactive Exploration

This script provides an interactive environment for exploring the Fractal-HLIP system, including:
- Environment visualization
- Agent behavior analysis
- Attention pattern visualization
- Quick experiments and testing
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append(str(Path.cwd().parent / "src"))

from fractal_hlip_agent import (
    FractalDepthEnvironment,
    FractalHLIPAgent,
    construct_multi_scale_observation,
    visualize_observation,
    set_random_seeds
)

# Set style for plots
plt.style.use('default')


class FractalHLIPExplorer:
    """Interactive explorer for the Fractal-HLIP system."""
    
    def __init__(self, base_size: int = 16, num_portals_per_level: int = 4, 
                 max_depth: int = 3, random_seed: int = 42):
        """Initialize the explorer with environment and agent."""
        self.random_seed = random_seed
        set_random_seeds(random_seed)
        
        # Create environment
        self.env = FractalDepthEnvironment(
            base_size=base_size,
            num_portals_per_level=num_portals_per_level,
            max_depth=max_depth,
            random_seed=random_seed
        )
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Agent will be created when needed
        self.agent = None
        
        print("Fractal-HLIP Explorer Initialized!")
        print(f"Environment: {base_size}x{base_size}, depth {max_depth}, {num_portals_per_level} portals/level")
        print(f"Device: {self.device}")
    
    def explore_environment(self):
        """Explore and visualize the fractal environment."""
        print("\n" + "="*50)
        print("ENVIRONMENT EXPLORATION")
        print("="*50)
        
        print(f"Base size: {self.env.base_size}x{self.env.base_size}")
        print(f"Max depth: {self.env.max_depth}")
        print(f"Portals per level: {self.env.num_portals_per_level}")
        print(f"Portal locations: {self.env.base_portal_coords}")
        print(f"Goal location: {self.env.base_goal}")
        
        # Reset and show initial state
        state = self.env.reset()
        print("\nInitial Environment State:")
        print(self.env.render())
        
        return state
    
    def manual_navigation_demo(self):
        """Demonstrate manual navigation through the environment."""
        print("\n" + "="*50)
        print("MANUAL NAVIGATION DEMO")
        print("="*50)
        
        self.env.reset()
        print("Starting position:")
        print(self.env.render())
        
        # Demo sequence: try to reach a portal
        demo_actions = ['right', 'right', 'down', 'down', 'down']
        
        for i, action_name in enumerate(demo_actions):
            print(f"\nStep {i+1}: Moving {action_name}")
            next_state, reward, done, info = self.manual_step(action_name)
            
            if done:
                print("Episode completed!")
                break
            
            time.sleep(0.5)  # Small delay for visualization
    
    def manual_step(self, action_name: str) -> Tuple[Any, float, bool, Dict]:
        """Take a manual step in the environment."""
        action_map = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        
        if action_name not in action_map:
            print(f"Unknown action: {action_name}")
            return None, 0, False, {}
        
        action = action_map[action_name]
        next_state, reward, done, info = self.env.step(action)
        
        print(f"Action: {action_name} ({action})")
        print(f"Reward: {reward:.3f}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print("Environment State:")
        print(self.env.render())
        
        return next_state, reward, done, info
    
    def analyze_multi_scale_observations(self):
        """Analyze multi-scale observations at different states."""
        print("\n" + "="*50)
        print("MULTI-SCALE OBSERVATION ANALYSIS")
        print("="*50)
        
        scenarios = [
            ("Surface Level", 0, 1, 1),
            ("Near Portal", 0, 0, 4),
            ("Depth 1", 1, 8, 8),
            ("Near Goal", 0, 8, 7)
        ]
        
        observations = {}
        
        for name, depth, x, y in scenarios:
            print(f"\n=== {name} ===")
            obs = self.analyze_observation_at_state(depth, x, y)
            observations[name] = obs
            
            # Show detailed visualization for first scenario
            if name == "Surface Level":
                print("\nDetailed Observation Visualization:")
                print(visualize_observation(obs))
        
        return observations
    
    def analyze_observation_at_state(self, depth: int = 0, x: Optional[int] = None, 
                                   y: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Analyze multi-scale observation at a specific state."""
        # Set state
        if x is not None and y is not None:
            self.env.agent_x = x
            self.env.agent_y = y
        self.env.current_depth = depth
        
        # Get current state and observation
        state = self.env.get_state()
        observation = construct_multi_scale_observation(self.env, state)
        
        print(f"State: {state}")
        print("Observation shapes:")
        for key, array in observation.items():
            print(f"  {key}: {array.shape}")
        
        return observation
    
    def plot_observation_components(self, observation: Dict[str, np.ndarray], 
                                  title: str = "Multi-Scale Observation", show: bool = True):
        """Plot the different components of a multi-scale observation."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Local view channels
        local_view = observation['local_view']
        channel_names = ['Empty Space', 'Portals', 'Goal', 'Agent']
        
        for i in range(4):
            row, col = i // 2, i % 2
            if row < 2 and col < 2:
                im = axes[row, col].imshow(local_view[i], cmap='viridis')
                axes[row, col].set_title(f'Local View: {channel_names[i]}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
        
        # Current depth map (show first channel)
        im1 = axes[0, 2].imshow(observation['current_depth_map'][0], cmap='plasma')
        axes[0, 2].set_title('Current Depth Map')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # Parent depth map (show first channel)
        im2 = axes[1, 2].imshow(observation['parent_depth_map'][0], cmap='plasma')
        axes[1, 2].set_title('Parent Depth Map')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        # Show depth context
        print(f"Depth Context Vector: {observation['depth_context']}")
        
        return fig, axes
    
    def create_agent(self, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 1):
        """Create the Fractal-HLIP agent."""
        print("\n" + "="*50)
        print("AGENT CREATION")
        print("="*50)
        
        self.agent = FractalHLIPAgent(
            env=self.env,
            device=self.device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            seed=self.random_seed
        )
        
        print("Fractal-HLIP Agent Created!")
        print(f"Encoder parameters: {sum(p.numel() for p in self.agent.encoder.parameters()):,}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.agent.q_network.parameters()):,}")
        
        return self.agent
    
    def quick_training(self, num_episodes: int = 100, render_freq: int = 20, 
                      plot_freq: int = 10) -> Tuple[List[float], List[int]]:
        """Run a quick training session with live updates."""
        if self.agent is None:
            self.create_agent()
        
        print("\n" + "="*50)
        print("QUICK TRAINING SESSION")
        print("="*50)
        
        episode_rewards = []
        episode_steps = []
        
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Train one episode
            stats = self.agent.train_episode(max_steps=200, render=(episode % render_freq == 0))
            
            episode_rewards.append(stats['episode_reward'])
            episode_steps.append(stats['episode_steps'])
            
            # Update plot every plot_freq episodes
            if episode % plot_freq == 0:
                self.plot_training_progress(episode_rewards, episode_steps, episode)
                
                # Print recent performance
                recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                print(f"Episode {episode}: Recent avg reward = {np.mean(recent_rewards):.3f}")
                print(f"Current epsilon: {self.agent.epsilon_scheduler.get_epsilon():.3f}")
        
        print(f"\nTraining completed! Final average reward: {np.mean(episode_rewards[-10:]):.3f}")
        return episode_rewards, episode_steps
    
    def plot_training_progress(self, episode_rewards: List[float], episode_steps: List[int], 
                             current_episode: int):
        """Plot training progress."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.7)
        # Moving average
        if len(episode_rewards) >= 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
            plt.legend()
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(episode_steps, alpha=0.7)
        if len(episode_steps) >= 10:
            moving_avg = np.convolve(episode_steps, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(episode_steps)), moving_avg, 'r-', linewidth=2, label='Moving Average')
            plt.legend()
        plt.title('Episode Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_attention_patterns(self):
        """Analyze agent's attention patterns across different scenarios."""
        if self.agent is None:
            print("Agent not created yet. Creating agent first...")
            self.create_agent()
        
        print("\n" + "="*50)
        print("ATTENTION PATTERN ANALYSIS")
        print("="*50)
        
        scenarios = [
            ("Surface level", 0, 1, 1),
            ("Near portal", 0, 0, 4),
            ("Depth 1", 1, 8, 8),
            ("Near goal", 0, 8, 7)
        ]
        
        attention_data = {}
        
        for name, depth, x, y in scenarios:
            print(f"\n=== {name} ===")
            
            # Set environment state
            self.env.reset()
            self.env.current_depth = depth
            self.env.agent_x = x
            self.env.agent_y = y
            
            # Get observation and analysis
            state = self.env.get_state()
            observation = construct_multi_scale_observation(self.env, state)
            analysis = self.agent.get_attention_analysis(observation)
            
            attention_data[name] = analysis
            
            print(f"State: {state}")
            print(f"Q-values: {analysis['q_values']}")
            action_names = ['up', 'right', 'down', 'left']
            print(f"Selected action: {analysis['selected_action']} ({action_names[analysis['selected_action']]})")
            print(f"Feature norm: {analysis['feature_norm']:.3f}")
            
            # Show attention weights if available
            if 'attention_weights' in analysis and 'cross_scale_attention' in analysis['attention_weights']:
                attn_matrix = analysis['attention_weights']['cross_scale_attention'][0].cpu().numpy()
                print("Cross-scale attention (rows=queries, cols=keys):")
                print("  Scales: [Local, Current, Parent, Context]")
                
                for i, row in enumerate(attn_matrix):
                    print(f"  Scale {i}: [{', '.join(f'{x:.3f}' for x in row)}]")
        
        return attention_data
    
    def plot_attention_heatmap(self, scenario_name: str, depth: int, x: int, y: int):
        """Plot attention weights as a heatmap."""
        if self.agent is None:
            print("Agent not created yet. Creating agent first...")
            self.create_agent()
        
        # Set state and get analysis
        self.env.reset()
        self.env.current_depth = depth
        self.env.agent_x = x
        self.env.agent_y = y
        
        state = self.env.get_state()
        observation = construct_multi_scale_observation(self.env, state)
        analysis = self.agent.get_attention_analysis(observation)
        
        if 'attention_weights' in analysis and 'cross_scale_attention' in analysis['attention_weights']:
            attn_matrix = analysis['attention_weights']['cross_scale_attention'][0].cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            im = plt.imshow(attn_matrix, cmap='Blues', interpolation='nearest')
            plt.colorbar(im, label='Attention Weight')
            
            # Add labels
            scale_labels = ['Local', 'Current', 'Parent', 'Context']
            plt.xticks(range(4), scale_labels)
            plt.yticks(range(4), scale_labels)
            plt.xlabel('Key Scales')
            plt.ylabel('Query Scales')
            plt.title(f'Cross-Scale Attention: {scenario_name}')
            
            # Add text annotations
            for i in range(4):
                for j in range(4):
                    plt.text(j, i, f'{attn_matrix[i,j]:.3f}', 
                            ha='center', va='center', 
                            color='white' if attn_matrix[i,j] > 0.5 else 'black')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No attention weights available for visualization.")
    
    def visualize_episode(self, max_steps: int = 50, delay: float = 0.5, epsilon: float = 0.1):
        """Run and visualize an episode step by step."""
        if self.agent is None:
            print("Agent not created yet. Creating agent first...")
            self.create_agent()
        
        print("\n" + "="*50)
        print("EPISODE VISUALIZATION")
        print("="*50)
        
        state = self.env.reset()
        observation = construct_multi_scale_observation(self.env, state)
        
        episode_reward = 0
        step = 0
        
        print("Starting episode visualization...")
        print("Press Ctrl+C to stop early")
        
        try:
            while step < max_steps:
                # Choose action
                action = self.agent.choose_action(observation, epsilon=epsilon)
                action_name = ['up', 'right', 'down', 'left'][action]
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                next_observation = construct_multi_scale_observation(self.env, next_state)
                
                episode_reward += reward
                step += 1
                
                # Display
                print(f"\nStep {step}: Action = {action_name} ({action})")
                print(f"Reward = {reward:.3f}, Total = {episode_reward:.3f}")
                print(f"Info: {info}")
                print("Environment:")
                print(self.env.render())
                
                if done:
                    print("\n*** EPISODE COMPLETED ***")
                    break
                
                # Update for next step
                observation = next_observation
                
                # Delay for visualization
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by user.")
        
        return episode_reward, step
    
    def depth_analysis_experiment(self, num_episodes: int = 20) -> Dict[int, Dict[str, float]]:
        """Analyze agent performance across different starting depths."""
        if self.agent is None:
            print("Agent not created yet. Creating agent first...")
            self.create_agent()
        
        print("\n" + "="*50)
        print("DEPTH ANALYSIS EXPERIMENT")
        print("="*50)
        
        depth_performance = {}
        
        for depth in range(self.env.max_depth + 1):
            print(f"Testing at depth {depth}...")
            
            rewards = []
            steps = []
            
            for episode in range(num_episodes):
                # Reset and set depth
                self.env.reset()
                self.env.current_depth = depth
                
                episode_reward = 0
                episode_steps = 0
                
                done = False
                while not done and episode_steps < 100:  # Limit steps
                    state = self.env.get_state()
                    observation = construct_multi_scale_observation(self.env, state)
                    action = self.agent.choose_action(observation, epsilon=0.0)  # Greedy
                    
                    _, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    episode_steps += 1
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
            
            depth_performance[depth] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_steps': np.mean(steps),
                'std_steps': np.std(steps)
            }
        
        # Plot results
        self.plot_depth_analysis(depth_performance)
        
        # Print summary
        print("\nDepth Analysis Results:")
        for depth, stats in depth_performance.items():
            print(f"Depth {depth}: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f} reward, "
                  f"{stats['mean_steps']:.1f} ± {stats['std_steps']:.1f} steps")
        
        return depth_performance
    
    def plot_depth_analysis(self, depth_performance: Dict[int, Dict[str, float]]):
        """Plot depth analysis results."""
        depths = list(depth_performance.keys())
        mean_rewards = [depth_performance[d]['mean_reward'] for d in depths]
        std_rewards = [depth_performance[d]['std_reward'] for d in depths]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(depths, mean_rewards, yerr=std_rewards, marker='o', capsize=5)
        plt.xlabel('Starting Depth')
        plt.ylabel('Mean Episode Reward')
        plt.title('Agent Performance vs Starting Depth')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_agent(self, model_path: Optional[str] = None):
        """Save the trained agent."""
        if self.agent is None:
            print("No agent to save!")
            return
        
        if model_path is None:
            # Create models directory if it doesn't exist
            os.makedirs('../models', exist_ok=True)
            model_path = '../models/fractal_hlip_interactive_model.pth'
        
        self.agent.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    def load_agent(self, model_path: str):
        """Load a trained agent."""
        if self.agent is None:
            self.create_agent()
        
        self.agent.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def run_full_exploration(self):
        """Run a complete exploration session."""
        print("Starting Full Fractal-HLIP Exploration Session...")
        
        # 1. Environment exploration
        self.explore_environment()
        
        # 2. Manual navigation demo
        self.manual_navigation_demo()
        
        # 3. Multi-scale observation analysis
        observations = self.analyze_multi_scale_observations()
        
        # Plot observation components for surface level
        if 'Surface Level' in observations:
            self.plot_observation_components(observations['Surface Level'], 
                                           "Surface Level Multi-Scale Observation")
        
        # 4. Create and train agent
        self.create_agent()
        rewards, steps = self.quick_training(num_episodes=50)
        
        # 5. Attention analysis
        self.analyze_attention_patterns()
        
        # Plot attention heatmaps
        self.plot_attention_heatmap("Surface Near Portal", 0, 0, 4)
        self.plot_attention_heatmap("Deep Level", 1, 8, 8)
        
        # 6. Episode visualization
        print("\nRunning episode visualization...")
        self.visualize_episode(max_steps=20, delay=1.0)
        
        # 7. Depth analysis experiment
        depth_results = self.depth_analysis_experiment()
        
        # 8. Save the model
        self.save_agent()
        
        print("\n" + "="*50)
        print("EXPLORATION SESSION COMPLETED!")
        print("="*50)
        print("Summary:")
        print(f"- Trained for {len(rewards)} episodes")
        print(f"- Final average reward: {np.mean(rewards[-10:]):.3f}")
        print(f"- Depth analysis completed across {len(depth_results)} depth levels")
        print("- Model saved for future use")


def main():
    """Main function to run the interactive exploration."""
    print("Fractal-HLIP Interactive Exploration")
    print("====================================")
    
    # Create explorer
    explorer = FractalHLIPExplorer(
        base_size=16,
        num_portals_per_level=4,
        max_depth=3,
        random_seed=42
    )
    
    # Run full exploration
    explorer.run_full_exploration()


if __name__ == "__main__":
    main()
