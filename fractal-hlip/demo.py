#!/usr/bin/env python3
"""
Fractal-HLIP Demonstration Script

This script provides a quick demonstration of the Fractal-HLIP system:
1. Creates a fractal environment
2. Trains a Fractal-HLIP agent
3. Shows agent behavior and attention patterns
4. Compares with a simple baseline

Run with: python demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from fractal_hlip_agent import (
    FractalDepthEnvironment,
    FractalHLIPAgent,
    construct_multi_scale_observation,
    visualize_observation,
    set_random_seeds,
    print_network_info
)

def create_environment():
    """Create and display the fractal environment."""
    print("="*60)
    print("FRACTAL-HLIP DEMONSTRATION")
    print("="*60)
    
    print("\n1. Creating Fractal Environment...")
    
    env = FractalDepthEnvironment(
        base_size=16,
        num_portals_per_level=4,
        max_depth=3,
        random_seed=42
    )
    
    print(f"   ✓ Environment created: {env.base_size}x{env.base_size} grid")
    print(f"   ✓ Maximum depth: {env.max_depth}")
    print(f"   ✓ Portals per level: {env.num_portals_per_level}")
    print(f"   ✓ Portal locations: {env.base_portal_coords}")
    print(f"   ✓ Goal location: {env.base_goal}")
    
    # Show initial state
    state = env.reset()
    print(f"\n   Initial state: {state}")
    print("\n   Environment layout:")
    print(env.render())
    
    return env

def demonstrate_multi_scale_observation(env):
    """Demonstrate multi-scale observation construction."""
    print("\n2. Multi-Scale Observation System...")
    
    # Get observation at surface level
    state = env.get_state()
    observation = construct_multi_scale_observation(env, state)
    
    print(f"   ✓ Observation components:")
    for key, array in observation.items():
        print(f"     - {key}: {array.shape}")
    
    print(f"\n   Multi-scale observation structure:")
    print(visualize_observation(observation)[:500] + "...")  # Truncate for demo
    
    return observation

def create_and_train_agent(env, episodes=200):
    """Create and train a Fractal-HLIP agent."""
    print(f"\n3. Training Fractal-HLIP Agent ({episodes} episodes)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Create agent with smaller architecture for quick demo
    agent = FractalHLIPAgent(
        env=env,
        device=device,
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        seed=42
    )
    
    print("   ✓ Agent created with hierarchical attention encoder")
    print_network_info(agent.encoder, "   Hierarchical Encoder")
    
    # Training loop
    episode_rewards = []
    episode_steps = []
    
    print(f"\n   Training progress:")
    
    for episode in range(episodes):
        stats = agent.train_episode(max_steps=100)
        episode_rewards.append(stats['episode_reward'])
        episode_steps.append(stats['episode_steps'])
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            print(f"     Episode {episode+1:3d}: avg reward = {np.mean(recent_rewards):6.3f}, "
                  f"epsilon = {stats['epsilon']:.3f}")
    
    print(f"   ✓ Training completed!")
    print(f"     Final average reward: {np.mean(episode_rewards[-50:]):.3f}")
    
    return agent, episode_rewards, episode_steps

def demonstrate_agent_behavior(agent, env):
    """Demonstrate trained agent behavior."""
    print("\n4. Agent Behavior Analysis...")
    
    # Run a test episode
    env.reset()
    observation = construct_multi_scale_observation(env, env.get_state())
    
    print("   Sample agent decisions:")
    
    for step in range(5):
        # Get agent's analysis
        analysis = agent.get_attention_analysis(observation)
        action = analysis['selected_action']
        action_names = ['up', 'right', 'down', 'left']
        
        print(f"     Step {step+1}: Action = {action_names[action]} "
              f"(Q-values: {analysis['q_values'][:2]}...)")
        
        # Take step
        next_state, reward, done, info = env.step(action)
        observation = construct_multi_scale_observation(env, next_state)
        
        if done:
            print(f"       Episode completed! Total reward: {reward}")
            break
    
    print("   ✓ Agent behavior analysis completed")

def demonstrate_attention_patterns(agent, env):
    """Demonstrate attention pattern analysis."""
    print("\n5. Attention Pattern Analysis...")
    
    scenarios = [
        ("Surface level", 0, 1, 1),
        ("Near portal", 0, 0, 4),
        ("Deep level", 1, 8, 8)
    ]
    
    for name, depth, x, y in scenarios:
        env.reset()
        env.current_depth = depth
        env.agent_x = x
        env.agent_y = y
        
        state = env.get_state()
        observation = construct_multi_scale_observation(env, state)
        analysis = agent.get_attention_analysis(observation)
        
        print(f"   {name}:")
        print(f"     State: {state}")
        print(f"     Selected action: {analysis['selected_action']}")
        print(f"     Feature magnitude: {analysis['feature_norm']:.3f}")
        
        # Show attention if available
        if 'attention_weights' in analysis and 'cross_scale_attention' in analysis['attention_weights']:
            attn = analysis['attention_weights']['cross_scale_attention'][0].cpu().numpy()
            print(f"     Cross-scale attention (local→others): {attn[0]}")
    
    print("   ✓ Attention analysis completed")

def plot_training_results(episode_rewards, episode_steps):
    """Plot training results."""
    print("\n6. Training Results Visualization...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Smooth rewards
        window = 20
        if len(episode_rewards) >= window:
            smooth_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smooth_rewards, 'b-', linewidth=2, label='Smoothed')
        
        ax1.plot(episode_rewards, 'b-', alpha=0.3, label='Raw')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Episode steps
        if len(episode_steps) >= window:
            smooth_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
            ax2.plot(smooth_steps, 'r-', linewidth=2, label='Smoothed')
        
        ax2.plot(episode_steps, 'r-', alpha=0.3, label='Raw')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Steps')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('fractal_hlip_demo_results.png', dpi=150, bbox_inches='tight')
        print("   ✓ Training plots saved to 'fractal_hlip_demo_results.png'")
        
    except Exception as e:
        print(f"   ⚠ Could not create plots: {e}")

def run_performance_comparison(env, agent, episodes=20):
    """Compare agent performance with random baseline."""
    print("\n7. Performance Comparison...")
    
    # Test trained agent
    print("   Testing trained Fractal-HLIP agent...")
    hlip_rewards = []
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            state = env.get_state()
            observation = construct_multi_scale_observation(env, state)
            action = agent.choose_action(observation, epsilon=0.0)  # Greedy
            
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        hlip_rewards.append(episode_reward)
    
    # Test random agent
    print("   Testing random baseline agent...")
    random_rewards = []
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = np.random.randint(env.action_space_size)
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        random_rewards.append(episode_reward)
    
    # Compare results
    hlip_mean = np.mean(hlip_rewards)
    hlip_std = np.std(hlip_rewards)
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    
    print(f"   Results ({episodes} episodes each):")
    print(f"     Fractal-HLIP: {hlip_mean:6.3f} ± {hlip_std:5.3f}")
    print(f"     Random:       {random_mean:6.3f} ± {random_std:5.3f}")
    print(f"     Improvement:  {hlip_mean - random_mean:6.3f} ({((hlip_mean/random_mean-1)*100):+.1f}%)")
    
    return hlip_rewards, random_rewards

def main():
    """Main demonstration function."""
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Run demonstration
    env = create_environment()
    observation = demonstrate_multi_scale_observation(env)
    agent, rewards, steps = create_and_train_agent(env, episodes=200)
    demonstrate_agent_behavior(agent, env)
    demonstrate_attention_patterns(agent, env)
    plot_training_results(rewards, steps)
    hlip_rewards, random_rewards = run_performance_comparison(env, agent)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED!")
    print("="*60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Fractal environment with self-similar structure")
    print("  ✓ Multi-scale observation system")
    print("  ✓ Hierarchical attention encoder (HLIP-inspired)")
    print("  ✓ Deep reinforcement learning with DQN")
    print("  ✓ Cross-scale attention analysis")
    print("  ✓ Performance comparison with baseline")
    print()
    print("Next Steps:")
    print("  • Run full experiments: python experiments/run_training.py")
    print("  • Analyze results: python experiments/analyze_results.py")
    print("  • Interactive exploration: jupyter notebook notebooks/exploration.ipynb")
    print()
    print("The Fractal-HLIP system shows promise for learning in self-similar environments!")

if __name__ == '__main__':
    main() 