"""
Training Script for Fractal-HLIP Experiments

This script runs comparative experiments between:
1. Baseline agent with simple perception (local + current depth only)
2. Fractal-HLIP agent with hierarchical attention across multiple scales

The goal is to test the hypothesis that hierarchical attention improves learning
in fractal environments.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fractal_hlip_agent import (
    FractalDepthEnvironment,
    FractalHLIPAgent,
    construct_multi_scale_observation,
    HierarchicalAttentionEncoder,
    DQNNetwork,
    MetricsLogger,
    set_random_seeds,
    print_network_info
)

class BaselineAgent:
    """
    Baseline agent that uses only local view and current depth map (no hierarchical attention).
    """
    
    def __init__(self, env, device=None, seed=None):
        """Initialize baseline agent with simple perception."""
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if seed is not None:
            set_random_seeds(seed)
        
        # Simple CNN encoder for local view + current depth map
        self.encoder = torch.nn.Sequential(
            # Process local view (4 channels, 5x5)
            torch.nn.Conv2d(4, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            # Combine with flattened current depth map (4 * 8 * 8 = 256)
            torch.nn.Linear(32 + 256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        ).to(self.device)
        
        # Q-network
        self.q_network = DQNNetwork(
            feature_dim=128,
            action_space_size=env.action_space_size
        ).to(self.device)
        
        # Target network
        self.target_q_network = DQNNetwork(
            feature_dim=128,
            action_space_size=env.action_space_size
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Rest of RL setup (similar to FractalHLIPAgent)
        from fractal_hlip_agent.utils import ReplayBuffer, EpsilonScheduler
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            lr=1e-4
        )
        self.replay_buffer = ReplayBuffer(capacity=10000, seed=seed)
        self.epsilon_scheduler = EpsilonScheduler()
        self.logger = MetricsLogger()
        
        self.step_count = 0
        self.episode_count = 0
        
    def encode_observation(self, observation):
        """Encode observation using simple baseline approach."""
        # Use only local view and current depth map
        local_view = torch.from_numpy(observation['local_view']).float().unsqueeze(0).to(self.device)
        current_depth_map = torch.from_numpy(observation['current_depth_map']).float().to(self.device)
        
        # Process local view through CNN
        local_features = self.encoder[:4](local_view)  # Through conv, relu, pool, flatten
        
        # Flatten current depth map
        depth_features = current_depth_map.flatten().unsqueeze(0)
        
        # Concatenate and process
        combined = torch.cat([local_features, depth_features], dim=1)
        features = self.encoder[4:](combined)  # Through remaining linear layers
        
        return features
    
    def choose_action(self, observation, epsilon=None):
        """Choose action using simple baseline encoding."""
        if epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space_size)
        
        with torch.no_grad():
            features = self.encode_observation(observation)
            q_values = self.q_network(features)
            return q_values.argmax(dim=1).item()
    
    def train_episode(self, max_steps=1000, render=False):
        """Train baseline agent for one episode."""
        state = self.env.reset()
        observation = construct_multi_scale_observation(self.env, state)
        
        episode_reward = 0.0
        episode_steps = 0
        max_depth_reached = 0
        
        done = False
        while not done and episode_steps < max_steps:
            epsilon = self.epsilon_scheduler.get_epsilon()
            action = self.choose_action(observation, epsilon)
            
            next_state, reward, done, info = self.env.step(action)
            next_observation = construct_multi_scale_observation(self.env, next_state)
            
            # Store in replay buffer
            self.replay_buffer.push(observation, action, reward, next_observation, done)
            
            # Learn (simplified version)
            if len(self.replay_buffer) >= 1000:
                self._learn()
            
            observation = next_observation
            episode_reward += reward
            episode_steps += 1
            max_depth_reached = max(max_depth_reached, info['new_depth'])
            
            self.epsilon_scheduler.step()
            
            # Render if requested (for debugging/visualization)
            if render:
                print(f"Step {episode_steps}: Action {action}, Reward {reward:.2f}")
                print(self.env.render())
        
        self.logger.log_episode(episode_reward, episode_steps, max_depth_reached, epsilon)
        self.episode_count += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'max_depth_reached': max_depth_reached,
            'epsilon': epsilon
        }
    
    def _learn(self):
        """Simple learning step for baseline."""
        batch = self.replay_buffer.sample(32)
        
        # Convert observations
        obs_features = []
        next_obs_features = []
        
        for i in range(len(batch['actions'])):
            obs = {k: v[i] for k, v in batch['observations'].items()}
            next_obs = {k: v[i] for k, v in batch['next_observations'].items()}
            
            obs_features.append(self.encode_observation(obs))
            next_obs_features.append(self.encode_observation(next_obs))
        
        obs_features = torch.cat(obs_features, dim=0)
        next_obs_features = torch.cat(next_obs_features, dim=0)
        
        actions = torch.from_numpy(batch['actions']).long().to(self.device)
        rewards = torch.from_numpy(batch['rewards']).float().to(self.device)
        dones = torch.from_numpy(batch['dones']).bool().to(self.device)
        
        # Q-learning update
        current_q = self.q_network(obs_features).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_q_network(next_obs_features).max(dim=1)[0]
            target_q = rewards + (0.99 * next_q * ~dones)
        
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.step_count % 1000 == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.step_count += 1

def run_experiment(
    agent_type: str,
    num_episodes: int = 5000,
    eval_freq: int = 500,
    seed: int = 42,
    render_freq: int = 1000,
    save_path: str = None
):
    """
    Run training experiment for specified agent type.
    
    Args:
        agent_type: 'baseline' or 'fractal_hlip'
        num_episodes: Number of training episodes
        eval_freq: Frequency of evaluation episodes
        seed: Random seed
        render_freq: Frequency of rendering episodes
        save_path: Path to save results
    """
    print(f"\n{'='*60}")
    print(f"Running {agent_type.upper()} Agent Experiment")
    print(f"{'='*60}")
    
    # Set random seed
    set_random_seeds(seed)
    
    # Create environment
    env = FractalDepthEnvironment(
        base_size=16,
        num_portals_per_level=4,
        max_depth=3,
        random_seed=seed
    )
    
    print(f"Environment created: {env.base_size}x{env.base_size} grid, max depth {env.max_depth}")
    
    # Create agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if agent_type == 'baseline':
        agent = BaselineAgent(env, device=device, seed=seed)
        print("Created Baseline Agent (simple perception)")
    elif agent_type == 'fractal_hlip':
        agent = FractalHLIPAgent(env, device=device, seed=seed)
        print("Created Fractal-HLIP Agent (hierarchical attention)")
        print_network_info(agent.encoder, "Hierarchical Attention Encoder")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Print network info
    if hasattr(agent, 'q_network'):
        print_network_info(agent.q_network, "Q-Network")
    
    # Training loop
    print(f"\nStarting training for {num_episodes} episodes...")
    
    training_stats = []
    eval_stats = []
    
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Training episode
        stats = agent.train_episode(max_steps=1000, render=(episode % render_freq == 0))
        training_stats.append(stats)
        
        # Evaluation episodes
        if episode % eval_freq == 0 and episode > 0:
            print(f"\nEpisode {episode}: Evaluating...")
            
            if hasattr(agent, 'evaluate'):
                eval_result = agent.evaluate(num_episodes=10)
            else:
                # Simple evaluation for baseline
                eval_rewards = []
                for _ in range(10):
                    eval_stats_ep = agent.train_episode(max_steps=1000)  # No learning during eval
                    eval_rewards.append(eval_stats_ep['episode_reward'])
                eval_result = {'mean_reward': np.mean(eval_rewards)}
            
            eval_stats.append({
                'episode': episode,
                'mean_reward': eval_result['mean_reward']
            })
            
            # Print progress
            recent_rewards = [s['episode_reward'] for s in training_stats[-100:]]
            print(f"  Recent 100 episodes - Mean reward: {np.mean(recent_rewards):.3f}")
            print(f"  Evaluation - Mean reward: {eval_result['mean_reward']:.3f}")
            
            if hasattr(agent, 'epsilon_scheduler'):
                print(f"  Current epsilon: {agent.epsilon_scheduler.get_epsilon():.3f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    # Save results
    if save_path:
        results = {
            'agent_type': agent_type,
            'training_stats': training_stats,
            'eval_stats': eval_stats,
            'total_time': total_time,
            'num_episodes': num_episodes,
            'seed': seed
        }
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {save_path}")
        
        # Save model if possible
        if hasattr(agent, 'save_model'):
            model_path = save_path.replace('.pkl', '_model.pth')
            agent.save_model(model_path)
    
    return training_stats, eval_stats, agent

def compare_agents(num_episodes: int = 5000, seeds: list = [42, 123, 456]):
    """
    Run comparative experiments between baseline and Fractal-HLIP agents.
    
    Args:
        num_episodes: Number of training episodes per agent
        seeds: List of random seeds for multiple runs
    """
    print("\n" + "="*80)
    print("COMPARATIVE EXPERIMENT: Baseline vs Fractal-HLIP")
    print("="*80)
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n--- Running experiments with seed {seed} ---")
        
        # Baseline agent
        baseline_stats, baseline_eval, baseline_agent = run_experiment(
            agent_type='baseline',
            num_episodes=num_episodes,
            seed=seed,
            save_path=f'results/baseline_seed_{seed}.pkl'
        )
        
        # Fractal-HLIP agent
        hlip_stats, hlip_eval, hlip_agent = run_experiment(
            agent_type='fractal_hlip',
            num_episodes=num_episodes,
            seed=seed,
            save_path=f'results/fractal_hlip_seed_{seed}.pkl'
        )
        
        all_results[seed] = {
            'baseline': {'training': baseline_stats, 'eval': baseline_eval},
            'fractal_hlip': {'training': hlip_stats, 'eval': hlip_eval}
        }
    
    # Plot comparison
    plot_comparison(all_results, save_path='results/comparison_plot.png')
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Calculate average performance
    for agent_type in ['baseline', 'fractal_hlip']:
        final_rewards = []
        for seed in seeds:
            stats = all_results[seed][agent_type]['training']
            final_100_rewards = [s['episode_reward'] for s in stats[-100:]]
            final_rewards.extend(final_100_rewards)
        
        print(f"{agent_type.upper()}:")
        print(f"  Final 100 episodes mean reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")

def plot_comparison(results, save_path=None):
    """Plot comparison between agents."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training curves
    plt.subplot(1, 3, 1)
    for seed in results.keys():
        baseline_rewards = [s['episode_reward'] for s in results[seed]['baseline']['training']]
        hlip_rewards = [s['episode_reward'] for s in results[seed]['fractal_hlip']['training']]
        
        # Smooth curves
        window = 100
        baseline_smooth = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
        hlip_smooth = np.convolve(hlip_rewards, np.ones(window)/window, mode='valid')
        
        plt.plot(baseline_smooth, alpha=0.7, color='blue', label='Baseline' if seed == list(results.keys())[0] else "")
        plt.plot(hlip_smooth, alpha=0.7, color='red', label='Fractal-HLIP' if seed == list(results.keys())[0] else "")
    
    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation performance
    plt.subplot(1, 3, 2)
    for seed in results.keys():
        baseline_eval = results[seed]['baseline']['eval']
        hlip_eval = results[seed]['fractal_hlip']['eval']
        
        if baseline_eval and hlip_eval:
            baseline_episodes = [e['episode'] for e in baseline_eval]
            baseline_rewards = [e['mean_reward'] for e in baseline_eval]
            hlip_episodes = [e['episode'] for e in hlip_eval]
            hlip_rewards = [e['mean_reward'] for e in hlip_eval]
            
            plt.plot(baseline_episodes, baseline_rewards, 'o-', alpha=0.7, color='blue', 
                    label='Baseline' if seed == list(results.keys())[0] else "")
            plt.plot(hlip_episodes, hlip_rewards, 'o-', alpha=0.7, color='red',
                    label='Fractal-HLIP' if seed == list(results.keys())[0] else "")
    
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final performance distribution
    plt.subplot(1, 3, 3)
    baseline_final = []
    hlip_final = []
    
    for seed in results.keys():
        baseline_stats = results[seed]['baseline']['training']
        hlip_stats = results[seed]['fractal_hlip']['training']
        
        baseline_final.extend([s['episode_reward'] for s in baseline_stats[-100:]])
        hlip_final.extend([s['episode_reward'] for s in hlip_stats[-100:]])
    
    plt.hist(baseline_final, bins=30, alpha=0.7, label='Baseline', color='blue')
    plt.hist(hlip_final, bins=30, alpha=0.7, label='Fractal-HLIP', color='red')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Final Performance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Train Fractal-HLIP agents')
    parser.add_argument('--agent', choices=['baseline', 'fractal_hlip', 'compare'], 
                       default='compare', help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456], 
                       help='Random seeds for multiple runs')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.agent == 'compare':
        compare_agents(num_episodes=args.episodes, seeds=args.seeds)
    else:
        for seed in args.seeds:
            save_path = f'{args.save_dir}/{args.agent}_seed_{seed}.pkl'
            run_experiment(args.agent, args.episodes, seed=seed, save_path=save_path)

if __name__ == '__main__':
    main() 