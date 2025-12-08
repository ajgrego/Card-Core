"""
Blackjack Poster Analysis Script
------------------------------------------------------------------------------------------------------------------------
Generates figures and statistics for the CardCore AI Blackjack poster.

Outputs:
    Figures (to blackjack_plots/):
        - alignment_comparison.png: Bar chart comparing alignment across 3 models
        - disagreement_heatmap.png: Where Thorp-initialized agent disagrees with Thorp
        - visit_count_distribution.png: Training data coverage across states
        - q_value_scatter.png: Initial vs final Q-values for Thorp-initialized model
        - action_distribution.png: Pie chart of action percentages in test set

    Statistics (to blackjack_plots/):
        - poster_stats.json: All numerical statistics for the poster

Author: Anthony Grego
"""

import os
import sys
import json
import csv
import ast
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
agents_dir = os.path.join(project_dir, "blackjack_agents")
sys.path.append(agents_dir)

from blackjack_rl_agent import BlackjackRLAgent, ThorpOnlyAgent, BlackjackState
from thorps_strategy import get_thorp_action


def load_models():
    """Load all trained models from pickle files."""
    models_dir = os.path.join(project_dir, "blackjack_outputs/models")

    # Load Thorp-initialized model
    thorp_init_agent = BlackjackRLAgent()
    thorp_init_agent.load_model(os.path.join(models_dir, "model_thorp_initialized.pkl"))

    # Load No-heuristic model
    no_heuristic_agent = BlackjackRLAgent()
    no_heuristic_agent.load_model(os.path.join(models_dir, "model_no_heuristic.pkl"))

    # Heuristic-only agent (no pickle needed)
    heuristic_only_agent = ThorpOnlyAgent()

    return {
        'thorp_initialized': thorp_init_agent,
        'no_heuristic': no_heuristic_agent,
        'heuristic_only': heuristic_only_agent
    }


def load_validation_data():
    """Load validation dataset."""
    val_path = os.path.join(project_dir, "blackjack_data/validation/blackjack_val_20k.csv")
    data = []
    with open(val_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'initial_hand': ast.literal_eval(row['initial_hand']),
                'dealer_up': int(row['dealer_up']),
                'true_count': int(row['true_count']),
                'actions_taken': ast.literal_eval(row.get('actions_taken', '[]')),
                'win': float(row['win'])
            })
    return data


def evaluate_alignment(agent, val_data):
    """Evaluate agent's alignment with Thorp's strategy."""
    total = 0
    matches = 0
    disagreements_by_type = defaultdict(int)  # (agent_action, thorp_action) -> count

    for row in val_data:
        hand = row['initial_hand']
        total_val = sum(hand)
        is_soft = 11 in hand
        dealer_up = row['dealer_up']
        true_count = row['true_count']

        state = BlackjackState(total_val, is_soft, dealer_up, true_count)
        available_actions = ['H', 'S', 'D', 'R']

        if hasattr(agent, 'get_best_action'):
            if isinstance(agent, ThorpOnlyAgent):
                agent_action = agent.get_best_action(state)
            else:
                agent_action = agent.get_best_action(state, available_actions)

        pair_value = hand[0] if len(hand) == 2 and hand[0] == hand[1] else None
        thorp_action = get_thorp_action(
            total_val, is_soft, dealer_up, pair_value,
            can_double=True, can_split=False, can_surrender=True
        )

        total += 1
        if agent_action == thorp_action:
            matches += 1
        else:
            disagreements_by_type[(agent_action, thorp_action)] += 1

    return {
        'total': total,
        'matches': matches,
        'alignment_pct': (matches / total * 100) if total > 0 else 0,
        'disagreements_by_type': dict(disagreements_by_type)
    }


def get_disagreement_matrix(agent, val_data):
    """Build matrix of disagreements by (player_total, dealer_upcard)."""
    # Matrix: rows = player total (4-21), cols = dealer upcard (2-11)
    disagree_counts = np.zeros((18, 10))  # 4-21 = 18 rows, 2-11 = 10 cols
    total_counts = np.zeros((18, 10))

    for row in val_data:
        hand = row['initial_hand']
        player_total = sum(hand)
        is_soft = 11 in hand
        dealer_up = row['dealer_up']
        true_count = row['true_count']

        if player_total < 4 or player_total > 21:
            continue
        if dealer_up < 2 or dealer_up > 11:
            continue

        state = BlackjackState(player_total, is_soft, dealer_up, true_count)
        available_actions = ['H', 'S', 'D', 'R']
        agent_action = agent.get_best_action(state, available_actions)

        pair_value = hand[0] if len(hand) == 2 and hand[0] == hand[1] else None
        thorp_action = get_thorp_action(
            player_total, is_soft, dealer_up, pair_value,
            can_double=True, can_split=False, can_surrender=True
        )

        row_idx = player_total - 4
        col_idx = dealer_up - 2
        total_counts[row_idx, col_idx] += 1
        if agent_action != thorp_action:
            disagree_counts[row_idx, col_idx] += 1

    return disagree_counts, total_counts


def get_visit_count_distribution(agent):
    """Get distribution of visit counts across state-action pairs."""
    visit_counts = []
    for key, count in agent.visit_counts.items():
        if count > 0:
            visit_counts.append(count)
    return visit_counts


def get_q_value_comparison(agent):
    """Get initial vs final Q-values for the agent."""
    initial_q = []
    final_q = []

    for key in agent.init_values:
        init_val = agent.init_values[key]
        final_val = agent.Q.get(key, init_val)
        initial_q.append(init_val)
        final_q.append(final_val)

    return initial_q, final_q


def get_action_distribution(val_data):
    """Get distribution of actions in the test set."""
    action_counts = defaultdict(int)

    for row in val_data:
        actions = row['actions_taken']
        if actions and actions[0]:
            first_action_list = actions[0]
            if isinstance(first_action_list, list) and len(first_action_list) > 0:
                action = first_action_list[0]
            elif isinstance(first_action_list, str):
                action = first_action_list
            else:
                continue

            # Skip 'N' (no action) and get next action if available
            if action == 'N' and len(first_action_list) > 1:
                action = first_action_list[1]

            if action in ['H', 'S', 'D', 'R', 'P']:
                action_counts[action] += 1

    return dict(action_counts)


def plot_alignment_comparison(alignments, output_dir):
    """Create bar chart comparing alignment percentages."""
    models = ['Thorp-Initialized', 'No-Heuristic', 'Heuristic-Only']
    values = [
        alignments['thorp_initialized']['alignment_pct'],
        alignments['no_heuristic']['alignment_pct'],
        alignments['heuristic_only']['alignment_pct']
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Alignment with Thorp Strategy (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title('Model Alignment Comparison', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alignment_comparison.png'), dpi=150)
    plt.close()


def plot_disagreement_heatmap(disagree_counts, total_counts, output_dir):
    """Create heatmap showing where agent disagrees with Thorp."""
    # Calculate disagreement rate
    with np.errstate(divide='ignore', invalid='ignore'):
        disagree_rate = np.where(total_counts > 0,
                                  disagree_counts / total_counts * 100,
                                  np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create custom colormap
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color='lightgray')

    im = ax.imshow(disagree_rate, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Labels
    player_totals = list(range(4, 22))
    dealer_upcards = list(range(2, 12))

    ax.set_xticks(range(10))
    ax.set_xticklabels([str(d) if d < 11 else 'A' for d in dealer_upcards])
    ax.set_yticks(range(18))
    ax.set_yticklabels(player_totals)

    ax.set_xlabel('Dealer Upcard', fontsize=12)
    ax.set_ylabel('Player Total', fontsize=12)
    ax.set_title('Thorp-Initialized Agent Disagreement with Thorp Strategy', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Disagreement Rate (%)', fontsize=11)

    # Add text annotations for cells with data
    for i in range(18):
        for j in range(10):
            if not np.isnan(disagree_rate[i, j]) and total_counts[i, j] > 0:
                text_color = 'white' if disagree_rate[i, j] > 50 else 'black'
                if disagree_rate[i, j] > 0:
                    ax.text(j, i, f'{disagree_rate[i, j]:.0f}%',
                           ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disagreement_heatmap.png'), dpi=150)
    plt.close()


def plot_visit_count_distribution(visit_counts, output_dir):
    """Create histogram of visit counts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(visit_counts) > 0:
        # Use log scale for x-axis bins
        max_count = max(visit_counts)
        bins = np.logspace(0, np.log10(max_count + 1), 50)

        ax.hist(visit_counts, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.set_xlabel('Visit Count (log scale)', fontsize=12)
        ax.set_ylabel('Number of State-Action Pairs', fontsize=12)
        ax.set_title('Training Data Coverage: Visit Count Distribution', fontsize=14, fontweight='bold')

        # Add statistics
        stats_text = f'Total observed pairs: {len(visit_counts):,}\n'
        stats_text += f'Mean visits: {np.mean(visit_counts):.1f}\n'
        stats_text += f'Median visits: {np.median(visit_counts):.0f}\n'
        stats_text += f'Max visits: {max_count:,}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visit_count_distribution.png'), dpi=150)
    plt.close()


def plot_q_value_scatter(initial_q, final_q, output_dir):
    """Create scatter plot of initial vs final Q-values."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample if too many points
    n_points = len(initial_q)
    if n_points > 5000:
        indices = np.random.choice(n_points, 5000, replace=False)
        init_sample = [initial_q[i] for i in indices]
        final_sample = [final_q[i] for i in indices]
    else:
        init_sample = initial_q
        final_sample = final_q

    ax.scatter(init_sample, final_sample, alpha=0.3, s=10, c='#3498db')

    # Add diagonal line (no change)
    min_val = min(min(initial_q), min(final_q))
    max_val = max(max(initial_q), max(final_q))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No change')

    ax.set_xlabel('Initial Q-Value (from Thorp)', fontsize=12)
    ax.set_ylabel('Final Q-Value (after training)', fontsize=12)
    ax.set_title('Q-Value Evolution: Thorp-Initialized Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_value_scatter.png'), dpi=150)
    plt.close()


def plot_action_distribution(action_counts, output_dir):
    """Create pie chart of action distribution."""
    action_names = {
        'H': 'Hit',
        'S': 'Stand',
        'D': 'Double',
        'R': 'Surrender',
        'P': 'Split'
    }

    # Sort by count for consistent ordering
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])

    labels = [action_names.get(a, a) for a, _ in sorted_actions]
    sizes = [c for _, c in sorted_actions]
    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#3498db']

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors[:len(labels)], startangle=90,
                                       explode=[0.02] * len(labels))

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Action Distribution in Validation Set', fontsize=14, fontweight='bold')

    # Add legend with counts
    legend_labels = [f'{l}: {s:,}' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title='Actions', loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=150)
    plt.close()


def calculate_statistics(models, val_data, alignments):
    """Calculate all statistics for poster_stats.json."""
    thorp_agent = models['thorp_initialized']
    no_heuristic_agent = models['no_heuristic']

    # Count training hands from model metadata
    training_hands = thorp_agent.episodes_seen
    eval_hands = len(val_data)

    # Count observed state-action pairs
    observed_pairs = len([k for k in thorp_agent.visit_counts if thorp_agent.visit_counts[k] > 0])

    # Calculate average Q-value shift
    q_shifts = []
    for key in thorp_agent.init_values:
        init_val = thorp_agent.init_values[key]
        final_val = thorp_agent.Q.get(key, init_val)
        q_shifts.append(final_val - init_val)
    avg_q_shift = np.mean(q_shifts) if q_shifts else 0

    # Format disagreements by action type
    disagreement_counts = defaultdict(int)
    for (agent_action, thorp_action), count in alignments['thorp_initialized']['disagreements_by_type'].items():
        key = f"agent_{agent_action}_when_thorp_{thorp_action}"
        disagreement_counts[key] = count

    stats = {
        'total_hands_trained': training_hands,
        'total_hands_evaluated': eval_hands,
        'alignment_percentages': {
            'thorp_initialized': alignments['thorp_initialized']['alignment_pct'],
            'no_heuristic': alignments['no_heuristic']['alignment_pct'],
            'heuristic_only': alignments['heuristic_only']['alignment_pct']
        },
        'observed_state_action_pairs': observed_pairs,
        'disagreements_by_action_type': dict(disagreement_counts),
        'average_q_value_shift': avg_q_shift
    }

    return stats


def main():
    print("=" * 70)
    print("BLACKJACK POSTER ANALYSIS")
    print("=" * 70)

    # Output directory
    output_dir = os.path.join(project_dir, "blackjack_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load models and data
    print("\nLoading models...")
    models = load_models()

    print("Loading validation data...")
    val_data = load_validation_data()
    print(f"  Loaded {len(val_data)} validation hands")

    # Evaluate alignments
    print("\nEvaluating model alignments...")
    alignments = {}
    for name, agent in models.items():
        print(f"  Evaluating {name}...")
        alignments[name] = evaluate_alignment(agent, val_data)
        print(f"    Alignment: {alignments[name]['alignment_pct']:.2f}%")

    # Generate figures
    print("\nGenerating figures...")

    print("  - Alignment comparison bar chart...")
    plot_alignment_comparison(alignments, output_dir)

    print("  - Disagreement heatmap...")
    disagree_counts, total_counts = get_disagreement_matrix(models['thorp_initialized'], val_data)
    plot_disagreement_heatmap(disagree_counts, total_counts, output_dir)

    print("  - Visit count distribution...")
    visit_counts = get_visit_count_distribution(models['thorp_initialized'])
    plot_visit_count_distribution(visit_counts, output_dir)

    print("  - Q-value scatter plot...")
    initial_q, final_q = get_q_value_comparison(models['thorp_initialized'])
    plot_q_value_scatter(initial_q, final_q, output_dir)

    print("  - Action distribution pie chart...")
    action_counts = get_action_distribution(val_data)
    plot_action_distribution(action_counts, output_dir)

    # Calculate and save statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(models, val_data, alignments)

    stats_path = os.path.join(output_dir, "poster_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to {stats_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nHands trained on: {stats['total_hands_trained']:,}")
    print(f"Hands evaluated on: {stats['total_hands_evaluated']:,}")
    print(f"Observed state-action pairs: {stats['observed_state_action_pairs']:,}")
    print(f"\nAlignment with Thorp's Strategy:")
    for model, pct in stats['alignment_percentages'].items():
        print(f"  {model}: {pct:.2f}%")
    print(f"\nAverage Q-value shift: {stats['average_q_value_shift']:.4f}")

    print("\n" + "=" * 70)
    print("Figures saved to: blackjack_plots/")
    print("  - alignment_comparison.png")
    print("  - disagreement_heatmap.png")
    print("  - visit_count_distribution.png")
    print("  - q_value_scatter.png")
    print("  - action_distribution.png")
    print("  - poster_stats.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
