
## Guided Walkthrough: Fractal-HLIP Repository

Hello! Welcome to the `Fractal-HLIP` project. This system was designed to explore whether a Reinforcement Learning agent, equipped with a perception mechanism inspired by HLIP's hierarchical attention, could develop a deeper understanding of a self-similar, fractal environment. The exciting news is that it seems to have worked remarkably well!

This walkthrough will guide you through the main parts of the repository.

### 1. High-Level Project Structure

When you open the `fractal-hlip/` directory, you'll see the following main areas:

```
fractal-hlip/
├── src/                      # Core source code for the agent and environment
│   └── fractal_hlip_agent/
├── experiments/              # Scripts to run training and analyze results
├── notebooks/                # Jupyter notebooks for interactive exploration
├── tests/                    # Unit tests for various components
├── demo.py                   # A script for a quick demonstration of the system
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview and setup instructions
```

*   **`src/fractal_hlip_agent/`**: This is the heart of the project. It contains all the Python modules that define the environment, the agent's perception, the agent's learning logic, and neural networks.
*   **`experiments/`**: Here you'll find scripts to conduct training runs (e.g., comparing the Fractal-HLIP agent to a baseline) and to analyze the data generated from these runs (e.g., plotting learning curves, statistical analysis).
*   **`notebooks/`**: Contains Jupyter notebooks, which are great for interactively playing with the environment, visualizing observations, or debugging parts of the agent.
*   **`tests/`**: Houses unit tests to ensure individual components are working as expected.
*   **`demo.py`**: A great starting point to see the whole system in action quickly.
*   **`README.md`**: Your first stop for project setup and a general overview.
*   **`requirements.txt`**: Lists the Python packages needed to run the project.

### 2. Deep Dive into `src/fractal_hlip_agent/`

This is where the magic happens. Let's go through the key files:

1.  **`environment.py` (`FractalDepthEnvironment`)**
    *   **Purpose**: Defines the world the agent lives in. It's a grid-based environment that is "fractal" or self-similar, meaning the agent can "zoom" into portals to enter new levels that have a similar structure to the parent level, but at a different scale.
    *   **Key Features**:
        *   **State**: Represented as `(x, y, depth)`.
        *   **Base Layout**: `base_obstacles`, `base_goal`, `base_portal_coords` define the repeating structure at each level.
        *   **Actions**: Typically Up, Down, Left, Right.
        *   **`step(action)` method**: Handles:
            *   Movement within the current `depth`.
            *   "Zooming in": Entering a portal increments `depth`, agent position might reset (e.g., to `(0,0)` in the new level), and the entry portal info is tracked.
            *   "Zooming out": Hitting an edge at `depth > 0` decrements `depth`, returning the agent to the parent portal.
        *   **Rewards**: Designed to guide the agent (e.g., step cost, rewards for entering portals, reaching sub-goals at deeper levels, and the main goal at `depth=0`).
    *   **Connection to Agent**: Provides `next_state, reward, done, info` and crucial information via `get_observation_info()` for building the agent's multi-scale perception.

2.  **`observation.py` (`construct_multi_scale_observation`)**
    *   **Purpose**: This module is critical. It takes the agent's current state in the `FractalDepthEnvironment` and constructs the complex, multi-scale observation that the `FractalHLIPAgent` will use as input.
    *   **Key Idea**: Instead of just seeing its immediate surroundings, the agent gets multiple "views" of the world at different scales.
    *   **Observation Structure (typically a dictionary of NumPy arrays/PyTorch tensors)**:
        *   `'local_view'`: A small patch around the agent at its current depth.
        *   `'current_depth_map'`: A downscaled overview of the entire grid at the agent's current depth.
        *   `'parent_depth_map'`: (If `depth > 0`) A downscaled view of the parent level, often centered around the portal the agent came through. This gives context.
        *   `'depth_context'`: A vector encoding the current depth, path length, etc.
    *   **How it works**: It queries the `FractalDepthEnvironment` (using `env.get_observation_info()`) for the necessary raw data (obstacles, portals, goal locations, agent position, depth) and then processes this into the structured dictionary of views.

3.  **`encoder.py` (`HierarchicalAttentionEncoder`)**
    *   **Purpose**: This is the "perceptual brain" of the `FractalHLIPAgent`, inspired by HLIP. It takes the multi-scale observation dictionary and processes it into a single, rich feature vector.
    *   **Architecture (PyTorch `nn.Module`)**:
        *   **Level 1 (Local Feature Extraction)**: Each view (`local_view`, `current_depth_map`, etc.) is typically processed by its own small neural network (e.g., CNNs for grid views, MLPs for vector inputs like `depth_context`) to extract initial features.
        *   **Level 2 (Contextual Attention - Optional/Advanced)**: Self-attention might be applied *within* certain views (like `current_depth_map` if treated as a sequence of patches) to build context-aware representations of that specific scale.
        *   **Level 3 (Cross-Scale Attention/Integration)**: This is the core HLIP-like mechanism. The feature vectors from the different scales (output of Level 1 or 2) are fed into a Transformer-style attention mechanism (e.g., `nn.MultiheadAttention` or a full Transformer Encoder layer). This allows the model to learn to weigh and integrate information *across* the different observational scales.
    *   **Output**: A flat feature vector that represents the agent's "understanding" of its current multi-scale situation. This vector becomes the input to the RL agent's decision-making networks.

4.  **`networks.py` (e.g., `DQNNetwork`, `DuelingDQNNetwork`, `ActorNetwork`, `CriticNetwork`)**
    *   **Purpose**: Defines the standard Reinforcement Learning neural network heads.
    *   **Input**: The feature vector produced by the `HierarchicalAttentionEncoder`.
    *   **Output**:
        *   For DQN: Q-values for each possible action.
        *   For Actor-Critic: A policy (action probabilities) and a state value.
    *   **Implementation**: Typically simple MLPs that sit on top of the encoder's output.

5.  **`agent.py` (`FractalHLIPAgent`)**
    *   **Purpose**: This class brings everything together. It's the RL agent itself.
    *   **Key Components**:
        *   An instance of the `FractalDepthEnvironment`.
        *   An instance of the `HierarchicalAttentionEncoder`.
        *   Instances of the RL networks (from `networks.py`).
        *   RL algorithm logic (e.g., DQN with experience replay, target network updates).
        *   Optimizer, Replay Buffer.
    *   **Core Methods**:
        *   `choose_action(observation_dict, epsilon)`: Gets the multi-scale observation, passes it through the encoder, then the RL network, and selects an action (e.g., epsilon-greedy).
        *   `learn(batch)`: Samples from the replay buffer, calculates losses using the encoder and networks, and performs a gradient update.
        *   `train_episode()`: Manages a single episode of interaction with the environment, including getting observations, choosing actions, stepping the environment, storing experiences, and learning.

6.  **`utils.py`**
    *   **Purpose**: Contains helper classes and functions used across the project.
    *   **Key Components**:
        *   `ReplayBuffer`: Stores `(observation_dict, action, reward, next_observation_dict, done)` tuples.
        *   `EpsilonScheduler`: Manages the exploration-exploitation trade-off (for epsilon-greedy).
        *   `MetricsLogger`: For tracking training progress (rewards, steps, loss, etc.).
        *   Helper functions like `dict_to_tensor` for converting observation dictionaries.

### 3. Running Experiments and Analysis: `experiments/`

*   **`run_training.py`**:
    *   This is the main script you'd run to train your agent(s).
    *   It handles instantiating the environment, the encoder, the agent(s) (Fractal-HLIP and potentially a simpler baseline for comparison).
    *   It contains the main training loop that calls `agent.train_episode()` repeatedly.
    *   It logs metrics and saves trained model weights.
    *   The conversation suggests this script sets up comparative experiments (e.g., Fractal-HLIP vs. Baseline).

*   **`analyze_results.py`**:
    *   After training, this script is used to load the saved logs and models.
    *   It generates plots (e.g., learning curves for reward, steps per episode, max depth reached).
    *   It can perform statistical analysis to compare the performance of different agents.
    *   It might also include code to visualize attention weights from the trained `HierarchicalAttentionEncoder` to understand what the agent is "looking at."

### 4. Interactive Exploration: `notebooks/exploration.ipynb`

*   This Jupyter notebook is an excellent place for:
    *   Interactively stepping through the `FractalDepthEnvironment`.
    *   Visualizing the different components of the `construct_multi_scale_observation` output for various agent states.
    *   Testing individual parts of the encoder or agent logic in isolation.
    *   Running small-scale training experiments or visualizing agent rollouts.

### 5. Tests: `tests/`

*   Contains unit tests for modules like `environment.py`, `encoder.py`, etc.
*   These help ensure that individual pieces of the system behave as expected. For example, `test_environment.py` would check if portals work correctly, if rewards are given appropriately, etc.

### How to Get Started & Run the Code (Based on the Conversation)

1.  **Setup**:
    *   Make sure you have Python and the dependencies from `requirements.txt` installed (especially PyTorch).
    *   The conversation indicated a `demo.py` script was created/used, which is an excellent first thing to run.

2.  **Run the Demo**:
    ```bash
    python demo.py
    ```
    This should give a quick overview of the system creating an environment, an agent, running a few steps, and potentially showing some outputs.

3.  **Run a Full Experiment**:
    ```bash
    python experiments/run_training.py --agent compare --episodes 5000
    ```
    (The specific arguments might vary, but this is the typical way to kick off a training run comparing the HLIP agent with a baseline). This will take time.

4.  **Analyze Results**:
    Once training is done, run:
    ```bash
    python experiments/analyze_results.py --all
    ```
    This should generate plots and statistical summaries, as indicated in the conversation. Check the `results/` directory (or similar) for output files like `learning_curves.png` or `fractal_hlip_report.html`.

5.  **Explore Interactively**:
    Open and run the cells in `notebooks/exploration.ipynb`.

### Tips for Your Friend:

*   **Start with `demo.py`**: It's likely designed to be a gentle introduction.
*   **Then, the `exploration.ipynb`**: This will let you play with the core components interactively.
*   **Trace the Data Flow**: A good way to understand the system is to follow how data flows:
    1.  `FractalDepthEnvironment` produces a raw state.
    2.  `construct_multi_scale_observation` turns this into a dictionary of views.
    3.  `HierarchicalAttentionEncoder` processes this dictionary into a feature vector.
    4.  `FractalHLIPAgent` (and its `DQNNetwork`) uses this feature vector to choose an action.
    5.  The action is applied to the `FractalDepthEnvironment`.
*   **Look at `experiments/run_training.py`**: This script shows how all the main `src` components are instantiated and wired together for a full training run.
*   **Use a Debugger**: If you want to understand the internals, stepping through the code with a debugger (e.g., in VS Code or PyCharm) while running the demo or a small experiment can be very insightful.

This project is quite ambitious and combines several interesting ideas. Given the success described in the conversation, diving into how the `HierarchicalAttentionEncoder` manages to make sense of the fractal world should be particularly rewarding!

Good luck with the exploration!

---