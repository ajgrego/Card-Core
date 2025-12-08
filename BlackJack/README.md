# BlackJack

This directory contains all components for the Blackjack reinforcement learning project, including agents, simulators, data generation tools, and analysis scripts.

## Directory Structure

### agents/

Contains the reinforcement learning agents and training infrastructure.

- `blackjack_rl_agent.py` - Monte Carlo RL agent with optional Thorp initialization. Implements `BlackjackRLAgent` (trainable) and `ThorpOnlyAgent` (baseline heuristic).
- `thorps_strategy.py` - Implementation of Edward O. Thorp's optimal basic strategy from "Beat the Dealer" (1962). Includes lookup tables for hard totals, soft totals, and pair splitting decisions.
- `train_all_models.py` - Script to train and compare three model variants: Thorp-initialized, no-heuristic (learns from scratch), and heuristic-only (pure Thorp strategy).

### data/

Simulation datasets used for training and evaluation.

- `development/` - 10,000 hands for development and debugging. Also contains pre-computed Q and V tables (`Q_full.csv`, `V_full.csv`).
- `training/` - 100,000 hands for model training.
- `validation/` - 20,000 hands for model evaluation.

Each CSV file contains hand records with fields including initial hand, dealer upcard, true count, actions taken, and win/loss outcomes.

### heuristics/

- `blackjackHeuristic.py` - Q-value based learned heuristic that computes expected values from simulation data and selects optimal actions based on state-action value estimates.

### outputs/

Trained model artifacts and results.

- `models/` - Serialized model files (`.pkl`) for trained agents.
- `results/` - JSON files containing model comparison metrics and alignment statistics.

### plots/

Generated visualizations for analysis and reporting.

- `alignment_comparison.png` - Bar chart comparing strategy alignment across model variants.
- `disagreement_heatmap.png` - Heatmap showing where the RL agent deviates from Thorp's strategy.
- `action_distribution.png` - Distribution of actions in the test set.
- `q_value_scatter.png` - Comparison of initial vs learned Q-values.
- Additional diagnostic plots for policy changes and visit counts.

### rl_seperated/

Isolated reinforcement learning implementation for independent development.

- `blackjack_simulator.py` - Blackjack game simulator with card counting support.
- `blackjack_rl.py` - Q-learning implementation for training directly against the simulator.
- `blackjackHeuristic.py` - Heuristic agent for this module.
- `run_heuristic.py` - Script to evaluate the learned heuristic policy.

### scripts/

Data generation and analysis utilities.

- `blackjack_simulator.py` - Full-featured blackjack simulator based on dennis-ho's implementation. Supports configurable deck count, penetration, and outputs detailed hand logs.
- `blackjack_analysis.py` - Generates figures and statistics for model comparison. Produces alignment charts, disagreement heatmaps, and summary statistics.
- `generate_all_data.sh` - Shell script to generate all three datasets (development, training, validation).

## Usage

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib (for analysis and plotting)

### Generating Training Data

To regenerate the simulation datasets:

```bash
cd BlackJack/scripts
chmod +x generate_all_data.sh
./generate_all_data.sh
```

This creates three datasets with 8 decks and 6.5 deck penetration:
- Development: 10,000 hands
- Training: 100,000 hands
- Validation: 20,000 hands

You can also run the simulator directly with custom parameters:

```bash
python3 blackjack_simulator.py --hands 50000 --decks 6 --pen 5.0 --output_path custom_data.csv
```

### Training Models

To train all three model variants and generate comparison metrics:

```bash
cd BlackJack/agents
python3 train_all_models.py
```

This will:
1. Train a Thorp-initialized Monte Carlo agent
2. Train a no-heuristic agent (learns from scratch)
3. Evaluate a pure Thorp strategy baseline
4. Save models to `outputs/models/`
5. Save comparison results to `outputs/results/model_comparison.json`

To run just the RL agent standalone:

```bash
python3 blackjack_rl_agent.py
```

### Running Analysis

To generate visualizations and statistics:

```bash
cd BlackJack/scripts
python3 blackjack_analysis.py
```

This loads trained models, evaluates them against the validation set, and outputs:
- Alignment comparison charts
- Disagreement heatmaps
- Action distribution plots
- Summary statistics in `plots/poster_stats.json`

### Evaluating the Q-Learning Agent

To train and evaluate the Q-learning implementation:

```bash
cd BlackJack/rl_seperated
python3 blackjack_rl.py
```

To evaluate the learned heuristic policy:

```bash
python3 run_heuristic.py
```

## Model Variants

The project compares three approaches:

1. **Thorp-Initialized** - Q-values start from Thorp's optimal strategy and are refined through Monte Carlo updates. Benefits from expert knowledge while allowing learning.

2. **No-Heuristic** - Q-values initialize to zero and learn entirely from experience. Demonstrates the challenge of learning without prior knowledge.

3. **Heuristic-Only** - Pure Thorp strategy with no learning. Serves as an optimal baseline for comparison.

## References

- Thorp, E. O. (1962). Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Random House.
- Blackjack simulator based on https://github.com/dennis-ho/blackjack-simulator
