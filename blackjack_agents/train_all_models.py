"""
Trains All Model Variants for Comparisons
------------------------------------------------------------------------------------------------------------------------
This script trains and evaluates three blackjack agent variants to demonstrate the importance of Thorp initialization:
    1. Thorp-Initialized Model — Q-values start from Thorp, refined with training
    2. No-Heuristic Model — Q-values start at 0, trained from scratch
    3. Heuristic-Only Model — Pure Thorp strategy, no learning at all

THE COUNTERFACTUAL PROBLEM:
When training data is generated using Thorp's strategy:
    - The data only contains Thorp-recommended actions
    - Those actions often have negative returns (house edge ~2%)
    - After training: observed actions accumulate negative Q-values
    - Unobserved actions retain Q=0 (neutral)
    - Without Bayesian initialization, the agent incorrectly prefers unobserved actions

Author: Anthony Grego
Date: December 5, 2025
"""
#imports
import os
import sys
import json
from datetime import datetime
#Adds the current directory to path for imports for blackjack_rl_agent.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blackjack_rl_agent import BlackjackRLAgent, ThorpOnlyAgent

def main():
    #Trains and save all three model variants
    print("-"*80)
    print("Begining to train all blackjack agent model variants for comparison")
    print("-"*80)

    #Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    train_csv = os.path.join(project_dir, "blackjack_data/training/blackjack_train_100k.csv")
    dev_csv = os.path.join(project_dir, "blackjack_data/development/blackjack_dev_10k.csv")
    output_dir = base_dir

    #Verify data files exist
    if not os.path.exists(train_csv):
        print(f"ERROR: Training data not found: {train_csv}")
        sys.exit(1)
    if not os.path.exists(dev_csv):
        print(f"ERROR: Development data not found: {dev_csv}")
        sys.exit(1)
    results = {}

    #----------------------------------------------------------------------------------------
    #MODEL 1: Thorp-Initialized MC Agent
    print("\n" + "-"*80)
    print("MODEL 1: Thorp-Initialized MC Agent")
    print("-"*80)

    agent_thorp = BlackjackRLAgent(prior_strength=10)
    agent_thorp.initialize_q_table_from_thorp()
    agent_thorp.model_type = 'thorp_initialized'
    agent_thorp.train_from_csv(train_csv)
    alignment_thorp = agent_thorp.evaluate_thorp_alignment(dev_csv)
    stats_thorp = agent_thorp.get_statistics()
    results['thorp_initialized'] = {
        'alignment': alignment_thorp,
        'stats': stats_thorp
    }
    model_path_thorp = os.path.join(output_dir, "model_thorp_initialized.pkl")
    agent_thorp.save_model(model_path_thorp)

    #-------------------------------------------------------------------------------------------
    #MODEL 2: No-Heuristic MC Agent; just learning
    #------------------------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODEL 2: No-Heuristic MC Agent; just learning")
    print("-"*80)

    agent_zero = BlackjackRLAgent(prior_strength=0)
    agent_zero.initialize_q_table_zeros()
    agent_zero.model_type = 'no_heuristic'
    agent_zero.train_from_csv(train_csv)
    alignment_zero = agent_zero.evaluate_thorp_alignment(dev_csv)
    stats_zero = agent_zero.get_statistics()
    results['no_heuristic'] = {
        'alignment': alignment_zero,
        'stats': stats_zero
    }
    model_path_zero = os.path.join(output_dir, "model_no_heuristic.pkl")
    agent_zero.save_model(model_path_zero)

    #-------------------------------------------------------------------------------------------
    #MODEL 3: Heuristic-Only Agent; just Thorp Strategy
    #--------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODEL 3: Heuristic-Only Agent (Pure Thorp Strategy)")
    print("-"*80)

    agent_heuristic = ThorpOnlyAgent()
    alignment_heuristic = agent_heuristic.evaluate_thorp_alignment(dev_csv)
    stats_heuristic = agent_heuristic.get_statistics()
    results['heuristic_only'] = {
        'alignment': alignment_heuristic,
        'stats': stats_heuristic
    }
    #no model is saved because it pulls from thorps charts directly

    #--------------------------------------------------------------------------------------------
    #Comparison
    #------------------------------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODEL COMPARISON")
    print("-"*80)
    print("\nAlignment with Thorp's Optimal Strategy:")
    print(f"\n{'Model':<25} {'Alignment':>12} {'Decisions':>12} {'Q-Updates':>12}")
    print("-"*65)
    for name, data in results.items():
        align = data['alignment']['alignment_percentage']
        decisions = data['alignment']['total_decisions']
        updates = data['stats'].get('total_updates', 'N/A')
        if isinstance(updates, int):
            print(f"{name:<25} {align:>11.2f}% {decisions:>12} {updates:>12}")
        else:
            print(f"{name:<25} {align:>11.2f}% {decisions:>12} {updates:>12}")
    print("-"*65)

    #Save comparative results to JSON for later analysis
    json_results = {}
    for name, data in results.items():
        entry = {
            'alignment_pct': data['alignment']['alignment_percentage'],
            'total_decisions': data['alignment']['total_decisions'],
            'model_type': data['stats'].get('model_type', name),
            'trainable': data['stats'].get('trainable', False)
        }
        #Add RL-specific metrics for trainable models
        if data['stats'].get('trainable', False):
            entry['episodes_seen'] = data['stats'].get('episodes_seen', 0)
            entry['total_updates'] = data['stats'].get('total_updates', 0)
        json_results[name] = entry

    results_path = os.path.join(output_dir, "model_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    return results

if __name__ == "__main__":
    main()
