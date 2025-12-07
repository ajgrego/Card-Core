#!/usr/bin/env python3
"""
Train All Model Variants for Comparative Analysis
------------------------------------------------------------------------------------------------------------------------
This script trains and evaluates three blackjack agent variants to demonstrate
the importance of Thorp initialization:

1. Thorp-Initialized Model — Q-values start from Thorp, refined with training
2. No-Heuristic Model — Q-values start at 0, trained from scratch
3. Heuristic-Only Model — Pure Thorp strategy, no learning at all

The comparison demonstrates the "counterfactual problem" where zero-initialized
agents learn to prefer unobserved actions due to negative returns on observed
Thorp-recommended actions.

Usage:
    python train_all_models.py

Author: Anthony Grego
Date: December 2025
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blackjack_rl_agent import BlackjackMCAgent, ThorpOnlyAgent


def main():
    """Train and save all three model variants."""

    print("="*70)
    print("TRAINING ALL MODEL VARIANTS FOR COMPARISON")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)

    train_csv = os.path.join(project_dir, "blackjack_data/training/blackjack_train_100k.csv")
    dev_csv = os.path.join(project_dir, "blackjack_data/development/blackjack_dev_10k.csv")
    output_dir = base_dir  # Save models in blackjack_agents directory

    # Verify data files exist
    if not os.path.exists(train_csv):
        print(f"ERROR: Training data not found: {train_csv}")
        sys.exit(1)
    if not os.path.exists(dev_csv):
        print(f"ERROR: Development data not found: {dev_csv}")
        sys.exit(1)

    print(f"\nData files:")
    print(f"  Training: {train_csv}")
    print(f"  Development: {dev_csv}")
    print(f"  Output: {output_dir}")

    results = {}

    # =========================================================================
    # MODEL 1: Thorp-Initialized MC Agent
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 1: Thorp-Initialized MC Agent")
    print("="*70)
    print("This model initializes Q-values from Thorp's strategy as a Bayesian prior.")
    print("It should achieve high alignment with Thorp (~99%+).\n")

    agent_thorp = BlackjackMCAgent(prior_strength=10)
    agent_thorp.initialize_q_table_from_thorp()
    agent_thorp.model_type = 'thorp_initialized'
    agent_thorp.train_from_csv(train_csv)

    alignment_thorp = agent_thorp.evaluate_thorp_alignment(dev_csv)
    performance_thorp = agent_thorp.evaluate_performance(dev_csv)
    stats_thorp = agent_thorp.get_statistics()

    results['thorp_initialized'] = {
        'alignment': alignment_thorp,
        'performance': performance_thorp,
        'stats': stats_thorp
    }

    model_path_thorp = os.path.join(output_dir, "model_thorp_initialized.pkl")
    agent_thorp.save_model(model_path_thorp)

    # =========================================================================
    # MODEL 2: No-Heuristic MC Agent (Zero Initialization)
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 2: No-Heuristic MC Agent (Zero Initialization)")
    print("="*70)
    print("This model initializes Q-values to zero and uses no Thorp prior.")
    print("It demonstrates the 'counterfactual problem':")
    print("  - Training data only contains Thorp-recommended actions")
    print("  - Those actions have negative returns (house edge)")
    print("  - After training: observed actions get Q < 0")
    print("  - Unobserved actions stay at Q = 0")
    print("  - Agent incorrectly prefers unobserved actions!\n")

    agent_zero = BlackjackMCAgent(prior_strength=0)  # No Bayesian blending
    agent_zero.initialize_q_table_zeros()
    agent_zero.model_type = 'no_heuristic'
    agent_zero.train_from_csv(train_csv)

    alignment_zero = agent_zero.evaluate_thorp_alignment(dev_csv)
    performance_zero = agent_zero.evaluate_performance(dev_csv)
    stats_zero = agent_zero.get_statistics()

    results['no_heuristic'] = {
        'alignment': alignment_zero,
        'performance': performance_zero,
        'stats': stats_zero
    }

    model_path_zero = os.path.join(output_dir, "model_no_heuristic.pkl")
    agent_zero.save_model(model_path_zero)

    # =========================================================================
    # MODEL 3: Heuristic-Only Agent (Pure Thorp Strategy)
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 3: Heuristic-Only Agent (Pure Thorp Strategy)")
    print("="*70)
    print("This agent uses Thorp's strategy directly without any learning.")
    print("It serves as a ceiling/baseline - should achieve 100% alignment.\n")

    agent_heuristic = ThorpOnlyAgent()

    alignment_heuristic = agent_heuristic.evaluate_thorp_alignment(dev_csv)
    performance_heuristic = agent_heuristic.evaluate_performance(dev_csv)
    stats_heuristic = agent_heuristic.get_statistics()

    results['heuristic_only'] = {
        'alignment': alignment_heuristic,
        'performance': performance_heuristic,
        'stats': stats_heuristic
    }

    # No model to save - it's just Thorp's tables
    print("(No model file saved - uses Thorp's strategy tables directly)")

    # =========================================================================
    # COMPARATIVE SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)

    print(f"\n{'Model':<25} {'Alignment':>12} {'Win Rate':>12} {'EV':>12}")
    print("-"*65)

    for name, data in results.items():
        align = data['alignment']['alignment_percentage']
        win_rate = data['performance']['win_rate']
        ev = data['performance']['ev_per_hand']
        print(f"{name:<25} {align:>11.2f}% {win_rate:>11.1f}% {ev:>+11.2f}%")

    print("-"*65)

    # Analysis of the counterfactual problem
    print("\n" + "="*70)
    print("ANALYSIS: THE COUNTERFACTUAL PROBLEM")
    print("="*70)

    thorp_align = results['thorp_initialized']['alignment']['alignment_percentage']
    zero_align = results['no_heuristic']['alignment']['alignment_percentage']
    heuristic_align = results['heuristic_only']['alignment']['alignment_percentage']

    print(f"""
The no-heuristic model achieves only {zero_align:.1f}% alignment with Thorp's strategy,
compared to {thorp_align:.1f}% for the Thorp-initialized model.

This demonstrates the counterfactual problem:
- When training on policy-generated data (from Thorp's strategy), the agent only
  observes actions that Thorp recommends.
- Those actions often result in negative returns due to the house edge (~2%).
- After training, observed actions accumulate negative Q-values.
- Unobserved (non-Thorp) actions retain their initial Q-value of 0.
- The agent incorrectly learns to prefer unobserved actions!

The Thorp-initialized model avoids this by starting with correct prior beliefs:
- Thorp-recommended actions start with Q = +0.5
- Non-recommended actions start with Q = -0.5
- Even after training reduces the Thorp Q-values, they remain higher than
  the non-recommended alternatives.

This is why expert initialization is essential for learning from policy data.
""")

    print("="*70)

    # Save comparative results to JSON for later analysis
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'alignment_pct': data['alignment']['alignment_percentage'],
            'total_decisions': data['alignment']['total_decisions'],
            'win_rate': data['performance']['win_rate'],
            'loss_rate': data['performance']['loss_rate'],
            'push_rate': data['performance']['push_rate'],
            'ev_per_hand': data['performance']['ev_per_hand'],
            'hands_evaluated': data['performance']['hands_evaluated'],
            'model_type': data['stats'].get('model_type', name),
            'trainable': data['stats'].get('trainable', False)
        }

    # Add metadata
    json_results['_metadata'] = {
        'training_data': train_csv,
        'evaluation_data': dev_csv,
        'timestamp': datetime.now().isoformat(),
        'description': 'Comparative analysis of blackjack RL agent variants'
    }

    results_path = os.path.join(output_dir, "model_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
