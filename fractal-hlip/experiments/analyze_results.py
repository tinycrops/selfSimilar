"""
Analysis Script for Fractal-HLIP Experiments

This script loads saved training results and provides detailed analysis including:
- Learning curve comparisons
- Attention pattern analysis
- Statistical significance testing
- Performance across different depth levels
"""

import sys
import os
import pickle
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

from fractal_hlip_agent import (
    FractalDepthEnvironment,
    FractalHLIPAgent,
    construct_multi_scale_observation,
    visualize_observation
)

def load_results(results_dir: str) -> dict:
    """
    Load all experiment results from a directory.
    
    Args:
        results_dir: Directory containing .pkl result files
        
    Returns:
        Dictionary organizing results by agent type and seed
    """
    results = {}
    
    for file_path in Path(results_dir).glob("*.pkl"):
        if "model" in file_path.name:  # Skip model files
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        agent_type = data['agent_type']
        seed = data['seed']
        
        if agent_type not in results:
            results[agent_type] = {}
        
        results[agent_type][seed] = data
        
    print(f"Loaded results for {len(results)} agent types:")
    for agent_type, seeds in results.items():
        print(f"  {agent_type}: {len(seeds)} seeds ({list(seeds.keys())})")
    
    return results

def plot_learning_curves(results: dict, save_path: str = None):
    """
    Plot detailed learning curves with confidence intervals.
    
    Args:
        results: Results dictionary from load_results()
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Training rewards
    plt.subplot(2, 3, 1)
    
    for agent_type, agent_results in results.items():
        all_curves = []
        
        for seed, data in agent_results.items():
            rewards = [stat['episode_reward'] for stat in data['training_stats']]
            # Smooth the curve
            window = 100
            if len(rewards) >= window:
                smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                all_curves.append(smooth_rewards)
        
        if all_curves:
            # Calculate mean and confidence interval
            min_length = min(len(curve) for curve in all_curves)
            curves_array = np.array([curve[:min_length] for curve in all_curves])
            
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            episodes = np.arange(len(mean_curve))
            
            color = 'blue' if agent_type == 'baseline' else 'red'
            label = 'Baseline' if agent_type == 'baseline' else 'Fractal-HLIP'
            
            plt.plot(episodes, mean_curve, color=color, label=label, linewidth=2)
            plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, 
                           color=color, alpha=0.2)
    
    plt.xlabel('Episode (smoothed)')
    plt.ylabel('Reward')
    plt.title('Training Reward Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Episode length
    plt.subplot(2, 3, 2)
    
    for agent_type, agent_results in results.items():
        all_lengths = []
        
        for seed, data in agent_results.items():
            lengths = [stat['episode_steps'] for stat in data['training_stats']]
            window = 100
            if len(lengths) >= window:
                smooth_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
                all_lengths.append(smooth_lengths)
        
        if all_lengths:
            min_length = min(len(curve) for curve in all_lengths)
            curves_array = np.array([curve[:min_length] for curve in all_lengths])
            
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            episodes = np.arange(len(mean_curve))
            
            color = 'blue' if agent_type == 'baseline' else 'red'
            label = 'Baseline' if agent_type == 'baseline' else 'Fractal-HLIP'
            
            plt.plot(episodes, mean_curve, color=color, label=label, linewidth=2)
            plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.2)
    
    plt.xlabel('Episode (smoothed)')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Max depth reached
    plt.subplot(2, 3, 3)
    
    for agent_type, agent_results in results.items():
        all_depths = []
        
        for seed, data in agent_results.items():
            depths = [stat['max_depth_reached'] for stat in data['training_stats']]
            window = 100
            if len(depths) >= window:
                smooth_depths = np.convolve(depths, np.ones(window)/window, mode='valid')
                all_depths.append(smooth_depths)
        
        if all_depths:
            min_length = min(len(curve) for curve in all_depths)
            curves_array = np.array([curve[:min_length] for curve in all_depths])
            
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            episodes = np.arange(len(mean_curve))
            
            color = 'blue' if agent_type == 'baseline' else 'red'
            label = 'Baseline' if agent_type == 'baseline' else 'Fractal-HLIP'
            
            plt.plot(episodes, mean_curve, color=color, label=label, linewidth=2)
            plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.2)
    
    plt.xlabel('Episode (smoothed)')
    plt.ylabel('Max Depth Reached')
    plt.title('Exploration Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Final performance distribution
    plt.subplot(2, 3, 4)
    
    final_rewards = {}
    for agent_type, agent_results in results.items():
        rewards = []
        for seed, data in agent_results.items():
            # Take last 100 episodes
            episode_rewards = [stat['episode_reward'] for stat in data['training_stats'][-100:]]
            rewards.extend(episode_rewards)
        final_rewards[agent_type] = rewards
    
    # Box plot
    box_data = []
    labels = []
    for agent_type, rewards in final_rewards.items():
        box_data.append(rewards)
        labels.append('Baseline' if agent_type == 'baseline' else 'Fractal-HLIP')
    
    plt.boxplot(box_data, labels=labels)
    plt.ylabel('Episode Reward')
    plt.title('Final Performance Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Evaluation curves
    plt.subplot(2, 3, 5)
    
    for agent_type, agent_results in results.items():
        eval_episodes = []
        eval_rewards = []
        
        for seed, data in agent_results.items():
            if 'eval_stats' in data and data['eval_stats']:
                episodes = [stat['episode'] for stat in data['eval_stats']]
                rewards = [stat['mean_reward'] for stat in data['eval_stats']]
                
                if not eval_episodes:
                    eval_episodes = episodes
                    eval_rewards = [rewards]
                else:
                    eval_rewards.append(rewards)
        
        if eval_rewards and eval_episodes:
            eval_array = np.array(eval_rewards)
            mean_rewards = np.mean(eval_array, axis=0)
            std_rewards = np.std(eval_array, axis=0)
            
            color = 'blue' if agent_type == 'baseline' else 'red'
            label = 'Baseline' if agent_type == 'baseline' else 'Fractal-HLIP'
            
            plt.plot(eval_episodes, mean_rewards, 'o-', color=color, label=label, linewidth=2)
            plt.fill_between(eval_episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                           color=color, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Learning efficiency (episodes to reach threshold)
    plt.subplot(2, 3, 6)
    
    threshold = -0.5  # Example threshold for "decent" performance
    efficiency_data = {}
    
    for agent_type, agent_results in results.items():
        episodes_to_threshold = []
        
        for seed, data in agent_results.items():
            rewards = [stat['episode_reward'] for stat in data['training_stats']]
            
            # Find first episode where moving average exceeds threshold
            window = 100
            for i in range(window, len(rewards)):
                avg_reward = np.mean(rewards[i-window:i])
                if avg_reward >= threshold:
                    episodes_to_threshold.append(i)
                    break
            else:
                episodes_to_threshold.append(len(rewards))  # Never reached threshold
        
        efficiency_data[agent_type] = episodes_to_threshold
    
    # Bar plot
    agent_types = list(efficiency_data.keys())
    means = [np.mean(efficiency_data[agent]) for agent in agent_types]
    stds = [np.std(efficiency_data[agent]) for agent in agent_types]
    labels = ['Baseline' if agent == 'baseline' else 'Fractal-HLIP' for agent in agent_types]
    
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, 
           color=['blue' if 'Baseline' in label else 'red' for label in labels],
           alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel(f'Episodes to reach {threshold:.1f} reward')
    plt.title('Learning Efficiency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")
    else:
        plt.show()

def statistical_analysis(results: dict):
    """
    Perform statistical analysis comparing agent performance.
    
    Args:
        results: Results dictionary from load_results()
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Collect final performance data
    final_performance = {}
    
    for agent_type, agent_results in results.items():
        rewards = []
        steps = []
        depths = []
        
        for seed, data in agent_results.items():
            # Take last 100 episodes for each seed
            final_episodes = data['training_stats'][-100:]
            
            rewards.extend([stat['episode_reward'] for stat in final_episodes])
            steps.extend([stat['episode_steps'] for stat in final_episodes])
            depths.extend([stat['max_depth_reached'] for stat in final_episodes])
        
        final_performance[agent_type] = {
            'rewards': rewards,
            'steps': steps,
            'depths': depths
        }
    
    # Compare performance metrics
    if len(results) == 2:
        agent_types = list(results.keys())
        agent1, agent2 = agent_types[0], agent_types[1]
        
        print(f"\nComparing {agent1} vs {agent2}:")
        print("-" * 40)
        
        for metric in ['rewards', 'steps', 'depths']:
            data1 = final_performance[agent1][metric]
            data2 = final_performance[agent2][metric]
            
            # Descriptive statistics
            mean1, std1 = np.mean(data1), np.std(data1)
            mean2, std2 = np.mean(data2), np.std(data2)
            
            print(f"\n{metric.capitalize()}:")
            print(f"  {agent1}: {mean1:.3f} ± {std1:.3f}")
            print(f"  {agent2}: {mean2:.3f} ± {std2:.3f}")
            
            # Statistical test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            print(f"  Mann-Whitney U test: p = {p_value:.6f}")
            if p_value < 0.001:
                print(f"  *** Highly significant difference (p < 0.001)")
            elif p_value < 0.01:
                print(f"  ** Significant difference (p < 0.01)")
            elif p_value < 0.05:
                print(f"  * Significant difference (p < 0.05)")
            else:
                print(f"  No significant difference (p >= 0.05)")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            if pooled_std > 0:
                cohens_d = (mean2 - mean1) / pooled_std
                print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                
                if abs(cohens_d) >= 0.8:
                    effect_desc = "large"
                elif abs(cohens_d) >= 0.5:
                    effect_desc = "medium"
                elif abs(cohens_d) >= 0.2:
                    effect_desc = "small"
                else:
                    effect_desc = "negligible"
                
                print(f"  Effect size interpretation: {effect_desc}")

def analyze_attention_patterns(model_path: str, env_path: str = None):
    """
    Analyze attention patterns of a trained Fractal-HLIP agent.
    
    Args:
        model_path: Path to saved model
        env_path: Path to environment config (optional)
    """
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    # Create environment
    env = FractalDepthEnvironment(base_size=16, num_portals_per_level=4, max_depth=3)
    
    # Create agent and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = FractalHLIPAgent(env, device=device)
    
    try:
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return
    
    # Generate sample observations at different depths
    print("\nAnalyzing attention patterns across different scenarios...")
    
    scenarios = []
    
    # Scenario 1: Surface level, near portal
    env.reset()
    env.agent_x, env.agent_y = 0, 4  # Near a portal
    observation1 = construct_multi_scale_observation(env, env.get_state())
    scenarios.append(("Surface near portal", observation1))
    
    # Scenario 2: Depth 1, exploring
    env.reset()
    env.current_depth = 1
    env.agent_x, env.agent_y = 8, 8  # Center of level
    observation2 = construct_multi_scale_observation(env, env.get_state())
    scenarios.append(("Depth 1 exploring", observation2))
    
    # Scenario 3: Deep level, near goal
    env.reset()
    env.current_depth = 2
    env.agent_x, env.agent_y = env.base_goal[0], env.base_goal[1] - 1  # Near goal
    observation3 = construct_multi_scale_observation(env, env.get_state())
    scenarios.append(("Deep level near goal", observation3))
    
    for scenario_name, observation in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Get attention analysis
        analysis = agent.get_attention_analysis(observation)
        
        print(f"Q-values: {analysis['q_values']}")
        print(f"Selected action: {analysis['selected_action']}")
        print(f"Feature norm: {analysis['feature_norm']:.3f}")
        
        # Print attention weights
        if 'attention_weights' in analysis:
            attention = analysis['attention_weights']
            if 'cross_scale_attention' in attention:
                attn_matrix = attention['cross_scale_attention'][0].cpu().numpy()  # First batch item
                print("Cross-scale attention matrix:")
                print("  Scales: [Local, Current Depth, Parent Depth, Depth Context]")
                for i, row in enumerate(attn_matrix):
                    print(f"  Scale {i}: {row}")
        
        # Visualize observation
        print("\nObservation structure:")
        print(visualize_observation(observation))

def generate_comprehensive_report(results_dir: str, output_path: str = "fractal_hlip_report.html"):
    """
    Generate a comprehensive HTML report of the experiment results.
    
    Args:
        results_dir: Directory containing results
        output_path: Path for output HTML file
    """
    results = load_results(results_dir)
    
    # Create plots and save them
    plot_learning_curves(results, "learning_curves.png")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fractal-HLIP Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Fractal-HLIP Experiment Report</h1>
        
        <div class="summary">
            <h2>Experiment Summary</h2>
            <p>This report analyzes the performance of Fractal-HLIP agents compared to baseline agents
            in navigating fractal environments with hierarchical structure.</p>
            
            <h3>Key Findings:</h3>
            <ul>
                <li>Agents tested: {', '.join(results.keys())}</li>
                <li>Seeds per agent: {len(list(results.values())[0]) if results else 0}</li>
                <li>Episodes per run: {list(results.values())[0][list(list(results.values())[0].keys())[0]]['num_episodes'] if results else 'N/A'}</li>
            </ul>
        </div>
        
        <div class="plot">
            <h2>Learning Curves</h2>
            <img src="learning_curves.png" alt="Learning Curves" style="max-width: 100%;">
        </div>
        
        <h2>Performance Summary</h2>
        <p>Detailed statistical analysis and attention pattern studies can be found in the 
        experimental logs.</p>
        
        <p><em>Report generated by Fractal-HLIP analysis script.</em></p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Comprehensive report saved to {output_path}")

def main():
    """Main analysis function with command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Fractal-HLIP experiment results')
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Directory containing result files')
    parser.add_argument('--plots', action='store_true', help='Generate learning curve plots')
    parser.add_argument('--stats', action='store_true', help='Perform statistical analysis')
    parser.add_argument('--attention', type=str, help='Path to model for attention analysis')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive HTML report')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    if args.all:
        args.plots = True
        args.stats = True
        args.report = True
    
    # Load results
    if args.plots or args.stats or args.report:
        if not os.path.exists(args.results_dir):
            print(f"Results directory {args.results_dir} not found!")
            return
        
        results = load_results(args.results_dir)
        
        if not results:
            print("No results found!")
            return
    
    # Generate plots
    if args.plots:
        plot_learning_curves(results)
    
    # Statistical analysis
    if args.stats:
        statistical_analysis(results)
    
    # Attention analysis
    if args.attention:
        if os.path.exists(args.attention):
            analyze_attention_patterns(args.attention)
        else:
            print(f"Model file {args.attention} not found!")
    
    # Comprehensive report
    if args.report:
        generate_comprehensive_report(args.results_dir)

if __name__ == '__main__':
    main() 