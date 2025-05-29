# Fractal Hierarchical Learning for Agentic Perception (Fractal-HLIP)

## Overview

This project implements a novel Reinforcement Learning system that combines:

1. **FractalDepthEnvironment**: A self-similar, nested environment where agents can navigate across literal fractal scales
2. **FractalHLIPAgent**: An RL agent with hierarchical attention mechanisms inspired by HLIP (Hierarchical Language-Image Pre-training) for processing multi-scale observations

## Core Hypothesis

*Does an agent that processes multi-scale observations from a fractal environment using hierarchical attention develop a more profound, transferable understanding ("agentic knowledge" or "awareness") of its world, leading to superior problem-solving and generalization across fractal scales?*

## Key Features

- **Self-Similar Environment**: Navigate through nested depth levels with consistent structure
- **Multi-Scale Perception**: Agent simultaneously perceives local, current-depth, and parent-depth views
- **Hierarchical Attention**: HLIP-inspired neural architecture for cross-scale information integration
- **Deep Reinforcement Learning**: DQN-based agent with experience replay and target networks
- **Attention Analysis**: Tools for visualizing and understanding attention patterns
- **Comparative Framework**: Baseline agents for performance comparison

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NumPy, Matplotlib, SciPy
- Jupyter (for notebooks)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fractal-hlip
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python demo.py
   ```

## Quick Start

### 1. Run the Demo

The fastest way to see the system in action:

```bash
python demo.py
```

This will:
- Create a fractal environment
- Train a Fractal-HLIP agent for 200 episodes
- Show attention patterns and behavior analysis
- Compare performance with a random baseline

### 2. Interactive Exploration

For hands-on exploration:

```bash
jupyter notebook notebooks/exploration.ipynb
```

This notebook provides:
- Step-by-step environment navigation
- Multi-scale observation visualization
- Real-time training monitoring
- Attention pattern analysis

### 3. Full Experiments

Run comparative experiments between Fractal-HLIP and baseline agents:

```bash
# Compare both agents across multiple seeds
python experiments/run_training.py --agent compare --episodes 5000

# Train only Fractal-HLIP agent
python experiments/run_training.py --agent fractal_hlip --episodes 5000

# Train only baseline agent  
python experiments/run_training.py --agent baseline --episodes 5000
```

### 4. Analyze Results

Generate detailed analysis and plots:

```bash
# Generate all analyses
python experiments/analyze_results.py --all

# Just learning curves
python experiments/analyze_results.py --plots

# Statistical analysis
python experiments/analyze_results.py --stats

# Attention analysis (requires trained model)
python experiments/analyze_results.py --attention results/fractal_hlip_seed_42_model.pth
```

## Project Structure

```
fractal-hlip/
├── src/fractal_hlip_agent/        # Core implementation
│   ├── environment.py             # FractalDepthEnvironment
│   ├── observation.py             # Multi-scale observation system
│   ├── encoder.py                 # Hierarchical attention encoder
│   ├── networks.py                # RL neural networks
│   ├── agent.py                   # FractalHLIPAgent
│   └── utils.py                   # Utilities and replay buffer
├── experiments/                    # Training and analysis scripts
│   ├── run_training.py            # Main training script
│   └── analyze_results.py         # Results analysis
├── notebooks/                     # Interactive exploration
│   └── exploration.ipynb          # Jupyter notebook
├── tests/                         # Unit tests
├── demo.py                        # Quick demonstration
└── README.md                      # This file
```

## Core Components

### FractalDepthEnvironment

A self-similar grid world where:
- Agent navigates through portals to access nested depth levels
- Each level has identical structure (obstacles, portals, goals)
- Portal entry increases depth; boundary exit decreases depth
- Rewards encourage exploration and goal-reaching across scales

```python
env = FractalDepthEnvironment(
    base_size=16,           # 16x16 grid at each level
    num_portals_per_level=4, # 4 portals per level
    max_depth=3,            # Up to 3 depth levels
    random_seed=42          # Reproducible layouts
)
```

### Multi-Scale Observation System

Constructs rich observations containing:
- **Local View**: 5x5 patch around agent with 4 channels (empty, portals, goal, agent)
- **Current Depth Map**: 8x8 downscaled view of entire current level
- **Parent Depth Map**: Context from parent level (if exists)
- **Depth Context**: Encoded depth information and portal path

```python
observation = construct_multi_scale_observation(env, agent_state)
# Returns: {'local_view': (4,5,5), 'current_depth_map': (4,8,8), 
#          'parent_depth_map': (4,8,8), 'depth_context': (6,)}
```

### Hierarchical Attention Encoder

HLIP-inspired neural architecture with three levels:
1. **Level 1**: Extract features from each observational scale separately
2. **Level 2**: Apply self-attention within spatial scales
3. **Level 3**: Apply cross-scale attention to integrate information across scales

```python
encoder = HierarchicalAttentionEncoder(
    embed_dim=128,      # Feature dimension
    num_heads=8,        # Attention heads
    num_layers=2,       # Attention layers
    dropout=0.1         # Dropout rate
)
```

### FractalHLIPAgent

Complete RL agent integrating:
- Hierarchical attention encoder
- DQN or Dueling DQN networks
- Experience replay buffer
- Epsilon-greedy exploration
- Target network updates

```python
agent = FractalHLIPAgent(
    env=env,
    embed_dim=128,
    learning_rate=1e-4,
    batch_size=32,
    use_dueling=True
)
```

## Key Experiments

### Experiment 1: Baseline Comparison

Compare Fractal-HLIP agent against a baseline agent that uses only local view and current depth map (no hierarchical attention).

**Hypothesis**: Hierarchical attention improves learning efficiency and final performance.

### Experiment 2: Depth Generalization

Test agent performance when starting at different depth levels.

**Hypothesis**: Hierarchical attention enables better generalization across scales.

### Experiment 3: Attention Pattern Analysis

Analyze attention weights to understand how the agent integrates multi-scale information.

**Research Questions**: 
- Which scales does the agent attend to in different situations?
- How do attention patterns evolve during training?
- Do attention patterns reveal emergent hierarchical strategies?

## Usage Examples

### Basic Training Loop

```python
from fractal_hlip_agent import FractalDepthEnvironment, FractalHLIPAgent

# Create environment and agent
env = FractalDepthEnvironment(random_seed=42)
agent = FractalHLIPAgent(env, seed=42)

# Train for 1000 episodes
for episode in range(1000):
    stats = agent.train_episode(max_steps=200)
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {stats['episode_reward']:.3f}")

# Evaluate performance
eval_stats = agent.evaluate(num_episodes=50)
print(f"Evaluation: {eval_stats['mean_reward']:.3f} ± {eval_stats['std_reward']:.3f}")
```

### Attention Analysis

```python
# Get observation
state = env.get_state()
observation = construct_multi_scale_observation(env, state)

# Analyze agent's attention
analysis = agent.get_attention_analysis(observation)
print(f"Selected action: {analysis['selected_action']}")
print(f"Q-values: {analysis['q_values']}")
print(f"Cross-scale attention: {analysis['attention_weights']['cross_scale_attention']}")
```

### Visualization

```python
# Visualize multi-scale observation
print(visualize_observation(observation))

# Plot training curves
agent.logger.plot_training_curves('training_results.png')

# Render environment
print(env.render())
```

## Advanced Usage

### Custom Environment Configuration

```python
env = FractalDepthEnvironment(
    base_size=32,              # Larger grid
    num_portals_per_level=6,   # More portals
    max_depth=5,               # Deeper nesting
    step_cost=-0.02,           # Different reward structure
    portal_reward=0.2,
    goal_reward=2.0,
    random_seed=123
)
```

### Custom Agent Architecture

```python
agent = FractalHLIPAgent(
    env=env,
    embed_dim=256,             # Larger feature space
    num_heads=16,              # More attention heads
    num_layers=4,              # Deeper attention
    hidden_dims=(512, 512),    # Larger Q-network
    learning_rate=5e-5,        # Different learning rate
    use_dueling=True,          # Dueling DQN
    gradient_clip=0.5          # Gradient clipping
)
```

### Training with Custom Schedules

```python
# Custom epsilon schedule
from fractal_hlip_agent.utils import EpsilonScheduler

custom_epsilon = EpsilonScheduler(
    start_epsilon=0.9,
    end_epsilon=0.05,
    decay_steps=20000,
    decay_type='exponential'
)

agent.epsilon_scheduler = custom_epsilon
```

## Research Applications

This system enables research into:

1. **Hierarchical Perception**: How agents integrate information across multiple scales
2. **Fractal Cognition**: Whether self-similar environments promote transferable learning
3. **Attention Mechanisms**: Role of attention in multi-scale reasoning
4. **Emergent Strategies**: What navigation strategies emerge from hierarchical training
5. **Scale Invariance**: How agents generalize across different scales

## Performance Metrics

The system tracks multiple metrics:

- **Episode Rewards**: Total reward per episode
- **Episode Steps**: Number of steps to completion
- **Depth Exploration**: Maximum depth reached per episode
- **Success Rate**: Fraction of episodes reaching the goal
- **Learning Efficiency**: Episodes needed to reach performance thresholds
- **Attention Patterns**: Cross-scale attention weights and distributions

## Contributing

We welcome contributions! Areas of particular interest:

- New environment variations (different fractal structures)
- Alternative attention mechanisms
- Additional baseline agents
- Visualization improvements
- Performance optimizations

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fractal-hlip,
  title={Fractal Hierarchical Learning for Agentic Perception},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]}
}
```

## License

[Specify license here]

## Acknowledgments

- Inspired by HLIP (Hierarchical Language-Image Pre-training)
- Built with PyTorch and modern RL techniques
- Thanks to the broader RL and attention mechanism research communities

---

**Happy exploring the fractal depths of artificial intelligence! 🌀🤖** 