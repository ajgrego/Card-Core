"""
Blackjack Monte Carlo Reinforcement Learning Agent
------------------------------------------------------------------------------------------------------------------------
This module implements a Monte Carlo RL agent for Blackjack that:
1. Uses Thorp's optimal strategy as a Bayesian prior for Q-value initialization
2. Learns from simulation data with support for Hit, Stand, Double, and Surrender
3. Tracks comprehensive performance metrics (win/loss/EV)
4. Exports learned policies to CSV and ASCII tables

The agent uses a first-visit Monte Carlo approach where Q-values are initialized
from Thorp's strategy to accelerate learning and improve convergence.

References:
    - Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
    - Thorp, E. O. (1962). Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Random House.
    - Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.

Author: Anthony Grego
Date: December 2025
"""

import csv
import ast
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import sys
import os

# Import Thorp's strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from thorps_strategy import get_thorp_action, thorp_hard, thorp_soft

# =============================================================================
# CONSTANTS
# =============================================================================

# Actions the agent can learn (excluding splits - too complex due to multiple hands)
LEARNING_ACTIONS = ['H', 'S', 'D', 'R']  # Hit, Stand, Double, Surrender

# Q-value initialization constants for Bayesian prior approach
THORP_INIT_VALUE = 0.5      # Q-value for actions recommended by Thorp
NON_THORP_INIT_VALUE = -0.5  # Q-value for actions NOT recommended by Thorp

# Prior strength controls how much weight the Thorp initialization has
# Higher values make the agent more resistant to deviating from Thorp
DEFAULT_PRIOR_STRENGTH = 10

# True count range for state space
TC_MIN = -5
TC_MAX = 5

# Surrender states according to Thorp's strategy
# Hard 16 vs 9, 10, A and Hard 15 vs 10
THORP_SURRENDER_STATES = {
    (16, False, 9),   # Hard 16 vs 9
    (16, False, 10),  # Hard 16 vs 10
    (16, False, 11),  # Hard 16 vs A
    (15, False, 10),  # Hard 15 vs 10
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BlackjackState:
    """
    Represents a state in Blackjack for the RL agent.

    State includes:
        - player_total: Sum of player's cards (4-21)
        - is_soft: Whether hand contains a usable Ace
        - dealer_up: Dealer's upcard (2-11, where 11=Ace)
        - true_count: Card counting true count (clamped to [-5, +5])
    """
    player_total: int
    is_soft: bool
    dealer_up: int
    true_count: int = 0

    def to_tuple(self) -> Tuple[int, bool, int, int]:
        """Convert state to hashable tuple for use as dictionary key."""
        return (self.player_total, self.is_soft, self.dealer_up, self.true_count)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if not isinstance(other, BlackjackState):
            return False
        return self.to_tuple() == other.to_tuple()

    def __repr__(self):
        soft_str = "soft" if self.is_soft else "hard"
        return f"State(player={self.player_total} {soft_str}, dealer={self.dealer_up}, TC={self.true_count:+d})"


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """
    Track win/loss/push statistics and expected value for blackjack hands.

    This class records comprehensive performance metrics including:
    - Win/loss/push counts and rates
    - Total profit and expected value per hand
    - Per-action outcome tracking
    - True count-based outcome analysis
    """

    def __init__(self):
        self.hands_played = 0
        self.wins = 0          # Positive profit
        self.losses = 0        # Negative profit
        self.pushes = 0        # Zero profit
        self.total_profit = 0.0
        self.blackjacks = 0    # Natural 21s
        self.surrenders = 0    # Hands surrendered

        # Per-action tracking
        self.action_outcomes: Dict[str, Dict[str, Any]] = {
            'H': {'count': 0, 'profit': 0.0, 'wins': 0, 'losses': 0},
            'S': {'count': 0, 'profit': 0.0, 'wins': 0, 'losses': 0},
            'D': {'count': 0, 'profit': 0.0, 'wins': 0, 'losses': 0},
            'R': {'count': 0, 'profit': 0.0, 'wins': 0, 'losses': 0}
        }

        # By true count buckets: {tc: {'hands': n, 'profit': x}}
        self.tc_outcomes: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {'hands': 0, 'profit': 0.0, 'wins': 0, 'losses': 0}
        )

    def record_hand(self, profit: float, action: str, true_count: int,
                    is_blackjack: bool = False, is_surrender: bool = False) -> None:
        """
        Record outcome of a single hand.

        Args:
            profit: Net profit/loss for the hand (positive = win)
            action: First action taken ('H', 'S', 'D', 'R')
            true_count: True count at start of hand
            is_blackjack: Whether player had natural blackjack
            is_surrender: Whether player surrendered
        """
        self.hands_played += 1
        self.total_profit += profit

        # Classify outcome
        if profit > 0:
            self.wins += 1
        elif profit < 0:
            self.losses += 1
        else:
            self.pushes += 1

        # Track special cases
        if is_blackjack:
            self.blackjacks += 1
        if is_surrender:
            self.surrenders += 1

        # Per-action tracking
        if action in self.action_outcomes:
            self.action_outcomes[action]['count'] += 1
            self.action_outcomes[action]['profit'] += profit
            if profit > 0:
                self.action_outcomes[action]['wins'] += 1
            elif profit < 0:
                self.action_outcomes[action]['losses'] += 1

        # True count tracking (clamp to range)
        tc_clamped = max(TC_MIN, min(TC_MAX, true_count))
        self.tc_outcomes[tc_clamped]['hands'] += 1
        self.tc_outcomes[tc_clamped]['profit'] += profit
        if profit > 0:
            self.tc_outcomes[tc_clamped]['wins'] += 1
        elif profit < 0:
            self.tc_outcomes[tc_clamped]['losses'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return comprehensive statistics dictionary.

        Returns:
            Dictionary containing win_rate, loss_rate, push_rate, ev_per_hand,
            ev_by_action, ev_by_true_count, and other metrics.
        """
        stats = {
            'hands_played': self.hands_played,
            'wins': self.wins,
            'losses': self.losses,
            'pushes': self.pushes,
            'blackjacks': self.blackjacks,
            'surrenders': self.surrenders,
            'total_profit': self.total_profit,
            'win_rate': (self.wins / self.hands_played * 100) if self.hands_played > 0 else 0.0,
            'loss_rate': (self.losses / self.hands_played * 100) if self.hands_played > 0 else 0.0,
            'push_rate': (self.pushes / self.hands_played * 100) if self.hands_played > 0 else 0.0,
            'ev_per_hand': (self.total_profit / self.hands_played * 100) if self.hands_played > 0 else 0.0,
        }

        # EV by action
        ev_by_action = {}
        for action, data in self.action_outcomes.items():
            if data['count'] > 0:
                ev_by_action[action] = {
                    'count': data['count'],
                    'ev': (data['profit'] / data['count'] * 100),
                    'win_rate': (data['wins'] / data['count'] * 100)
                }
        stats['ev_by_action'] = ev_by_action

        # EV by true count buckets
        ev_by_tc = {}
        for tc in sorted(self.tc_outcomes.keys()):
            data = self.tc_outcomes[tc]
            if data['hands'] > 0:
                ev_by_tc[tc] = {
                    'hands': data['hands'],
                    'ev': (data['profit'] / data['hands'] * 100),
                    'win_rate': (data['wins'] / data['hands'] * 100)
                }
        stats['ev_by_true_count'] = ev_by_tc

        # Grouped TC stats
        low_tc = {'hands': 0, 'profit': 0.0}
        mid_tc = {'hands': 0, 'profit': 0.0}
        high_tc = {'hands': 0, 'profit': 0.0}

        for tc, data in self.tc_outcomes.items():
            if tc <= -3:
                low_tc['hands'] += data['hands']
                low_tc['profit'] += data['profit']
            elif tc >= 3:
                high_tc['hands'] += data['hands']
                high_tc['profit'] += data['profit']
            else:
                mid_tc['hands'] += data['hands']
                mid_tc['profit'] += data['profit']

        stats['ev_by_tc_group'] = {
            'low (TC <= -3)': (low_tc['profit'] / low_tc['hands'] * 100) if low_tc['hands'] > 0 else 0.0,
            'mid (-2 to +2)': (mid_tc['profit'] / mid_tc['hands'] * 100) if mid_tc['hands'] > 0 else 0.0,
            'high (TC >= +3)': (high_tc['profit'] / high_tc['hands'] * 100) if high_tc['hands'] > 0 else 0.0,
        }

        return stats

    def print_summary(self) -> None:
        """Print formatted performance summary."""
        stats = self.get_statistics()

        print(f"\n{'='*70}")
        print("PERFORMANCE STATISTICS")
        print(f"{'='*70}")
        print(f"Hands evaluated: {stats['hands_played']}")
        print(f"Win rate:  {stats['win_rate']:.1f}%")
        print(f"Loss rate: {stats['loss_rate']:.1f}%")
        print(f"Push rate: {stats['push_rate']:.1f}%")
        print(f"Expected Value: {stats['ev_per_hand']:+.2f}% (per hand)")
        print(f"Blackjacks: {stats['blackjacks']}")
        print(f"Surrenders: {stats['surrenders']}")

        print(f"\nEV by Action:")
        for action in ['H', 'S', 'D', 'R']:
            if action in stats['ev_by_action']:
                data = stats['ev_by_action'][action]
                print(f"  {action}: {data['ev']:+.2f}% (n={data['count']}, win_rate={data['win_rate']:.1f}%)")

        print(f"\nEV by True Count:")
        for group, ev in stats['ev_by_tc_group'].items():
            print(f"  TC {group}: {ev:+.2f}%")

        print(f"{'='*70}\n")

    def reset(self) -> None:
        """Reset all tracking statistics."""
        self.__init__()


# =============================================================================
# MONTE CARLO AGENT
# =============================================================================

class BlackjackMCAgent:
    """
    Monte Carlo Reinforcement Learning agent for Blackjack.

    This agent uses first-visit Monte Carlo to learn Q-values from experience,
    with Thorp's optimal strategy providing Bayesian prior initialization.

    Key features:
    - Bayesian prior: Q-values initialized from Thorp's strategy
    - Prior strength: Controls how much initial evidence the prior represents
    - True count awareness: Separate Q-values for different card counting situations
    - Surrender support: Can learn when surrender is optimal
    """

    def __init__(self,
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 tc_range: Tuple[int, int] = (TC_MIN, TC_MAX)):
        """
        Initialize the Monte Carlo agent.

        Args:
            prior_strength: Number of pseudo-observations for Bayesian prior.
                           Higher values make the agent more resistant to deviating from Thorp.
            tc_range: (min, max) true count values to track separately
        """
        self.prior_strength = prior_strength
        self.tc_min, self.tc_max = tc_range

        # Q-table: {(state_tuple, action): q_value}
        self.Q: Dict[Tuple[Tuple, str], float] = {}

        # Visit counts for each state-action pair (including prior pseudo-counts)
        self.N: Dict[Tuple[Tuple, str], int] = defaultdict(int)

        # State visit counts (for policy sample)
        self.state_visits: Dict[Tuple, int] = defaultdict(int)

        # Training statistics
        self.episodes_trained = 0
        self.episodes_skipped = 0
        self.rows_read = 0

        # Initialize Q-table from Thorp's strategy
        self._initialize_q_table_from_thorp()

        print(f"{'='*70}")
        print("BLACKJACK MONTE CARLO RL AGENT - TRAINING")
        print("With Thorp Strategy Initialization (including Surrender)")
        print(f"{'='*70}")
        print(f"\nInitialized Q-table with {len(self.Q)} state-action pairs from Thorp")
        print(f"Prior strength: {self.prior_strength}")
        print(f"True count range: [{self.tc_min}, {self.tc_max}]")

    def _initialize_q_table_from_thorp(self) -> None:
        """
        Initialize Q-values using Thorp's strategy as a Bayesian prior.

        For each possible state:
        - Actions recommended by Thorp get Q-value = THORP_INIT_VALUE
        - Other actions get Q-value = NON_THORP_INIT_VALUE
        - Visit counts initialized to prior_strength (pseudo-observations)

        This provides a strong starting point while allowing the agent to
        deviate when sufficient evidence suggests Thorp is suboptimal.
        """
        # Player totals: 4-21 for hard, 13-21 for soft
        hard_totals = range(4, 22)
        soft_totals = range(13, 22)
        dealer_upcards = range(2, 12)  # 2-10 plus 11 for Ace
        true_counts = range(self.tc_min, self.tc_max + 1)

        # Initialize hard hand states
        for total in hard_totals:
            for dealer_up in dealer_upcards:
                for tc in true_counts:
                    state = BlackjackState(total, False, dealer_up, tc)
                    state_tuple = state.to_tuple()

                    # Get Thorp's recommended action
                    thorp_action = get_thorp_action(total, False, dealer_up, can_surrender=True)

                    # Check if this is a surrender state
                    is_surrender_state = (total, False, dealer_up) in THORP_SURRENDER_STATES

                    for action in LEARNING_ACTIONS:
                        key = (state_tuple, action)

                        if action == 'R':
                            # Surrender initialization
                            if is_surrender_state:
                                self.Q[key] = THORP_INIT_VALUE
                            else:
                                self.Q[key] = NON_THORP_INIT_VALUE
                        elif action == thorp_action:
                            self.Q[key] = THORP_INIT_VALUE
                        else:
                            self.Q[key] = NON_THORP_INIT_VALUE

                        # Initialize visit count with prior strength
                        self.N[key] = self.prior_strength

        # Initialize soft hand states
        for total in soft_totals:
            for dealer_up in dealer_upcards:
                for tc in true_counts:
                    state = BlackjackState(total, True, dealer_up, tc)
                    state_tuple = state.to_tuple()

                    # Get Thorp's recommended action (no surrender for soft hands)
                    thorp_action = get_thorp_action(total, True, dealer_up, can_surrender=False)

                    for action in LEARNING_ACTIONS:
                        key = (state_tuple, action)

                        if action == 'R':
                            # Never surrender soft hands
                            self.Q[key] = NON_THORP_INIT_VALUE
                        elif action == thorp_action:
                            self.Q[key] = THORP_INIT_VALUE
                        else:
                            self.Q[key] = NON_THORP_INIT_VALUE

                        self.N[key] = self.prior_strength

    def _clamp_true_count(self, tc: int) -> int:
        """Clamp true count to the tracked range."""
        return max(self.tc_min, min(self.tc_max, tc))

    def _parse_episode_from_csv_row(self, row: Dict) -> Optional[Dict]:
        """
        Parse a CSV row into episode data for training.

        Args:
            row: Dictionary from CSV DictReader

        Returns:
            Episode dictionary or None if row should be skipped
        """
        try:
            # Parse initial hand
            initial_hand = ast.literal_eval(row['initial_hand'])
            if not initial_hand or len(initial_hand) < 2:
                return None

            # Calculate player total and soft status
            player_total = sum(initial_hand)
            is_soft = 11 in initial_hand

            # Handle soft ace conversion
            if player_total > 21 and is_soft:
                player_total -= 10
                is_soft = False

            # Skip busted initial hands (shouldn't happen but safety check)
            if player_total > 21 or player_total < 4:
                return None

            # Parse dealer upcard
            dealer_up = int(row['dealer_up'])

            # Parse true count
            true_count = self._clamp_true_count(int(row['true_count']))

            # Parse actions taken
            actions_taken = ast.literal_eval(row.get('actions_taken', '[]'))
            if not actions_taken or not actions_taken[0]:
                return None

            # Get first action (for first-visit MC)
            first_action_list = actions_taken[0]
            if isinstance(first_action_list, list):
                # Skip insurance decisions ('N', 'I')
                first_action = None
                for a in first_action_list:
                    if a in LEARNING_ACTIONS:
                        first_action = a
                        break
                if first_action is None:
                    return None
            elif isinstance(first_action_list, str):
                if first_action_list not in LEARNING_ACTIONS:
                    return None
                first_action = first_action_list
            else:
                return None

            # Skip splits (too complex)
            if first_action == 'P' or 'P' in str(actions_taken):
                return None

            # Parse reward (profit)
            reward = float(row['win'])

            # Check for blackjack and surrender
            player_final = row.get('player_final_value', '')
            is_blackjack = 'BJ' in str(player_final)
            is_surrender = first_action == 'R'

            return {
                'player_total': player_total,
                'is_soft': is_soft,
                'dealer_up': dealer_up,
                'true_count': true_count,
                'first_action': first_action,
                'reward': reward,
                'is_blackjack': is_blackjack,
                'is_surrender': is_surrender
            }

        except (ValueError, KeyError, SyntaxError) as e:
            return None

    def _update_q_value(self, state: BlackjackState, action: str, reward: float) -> None:
        """
        Update Q-value using incremental mean (Monte Carlo update).

        Uses the formula: Q(s,a) = Q(s,a) + (1/N(s,a)) * (G - Q(s,a))
        where G is the return (reward) from this episode.

        Args:
            state: The state visited
            action: The action taken
            reward: The return (total reward) from this episode
        """
        state_tuple = state.to_tuple()
        key = (state_tuple, action)

        # Increment visit count
        self.N[key] += 1

        # Get current Q-value (or initialize if not present)
        if key not in self.Q:
            self.Q[key] = 0.0

        # Incremental mean update
        n = self.N[key]
        self.Q[key] = self.Q[key] + (1.0 / n) * (reward - self.Q[key])

        # Track state visits
        self.state_visits[state_tuple] += 1

    def train_from_csv(self, csv_path: str, max_episodes: Optional[int] = None) -> None:
        """
        Train the agent from a CSV file of simulation data.

        Args:
            csv_path: Path to training CSV file
            max_episodes: Maximum episodes to train on (None = all)
        """
        print(f"Training from CSV: {csv_path}")

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.rows_read += 1

                if max_episodes and self.episodes_trained >= max_episodes:
                    break

                # Parse episode
                episode = self._parse_episode_from_csv_row(row)

                if episode is None:
                    self.episodes_skipped += 1
                    continue

                # Create state and update Q-value
                state = BlackjackState(
                    episode['player_total'],
                    episode['is_soft'],
                    episode['dealer_up'],
                    episode['true_count']
                )

                self._update_q_value(state, episode['first_action'], episode['reward'])
                self.episodes_trained += 1

                if self.episodes_trained % 10000 == 0:
                    print(f"  Processed {self.episodes_trained} episodes...")

        print(f"\nTraining complete!")
        print(f"  Rows read: {self.rows_read}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Episodes skipped: {self.episodes_skipped}")

    def get_best_action(self, state: BlackjackState,
                        available_actions: Optional[List[str]] = None) -> str:
        """
        Get the best action for a state according to current Q-values.

        Args:
            state: Current game state
            available_actions: List of available actions (default: LEARNING_ACTIONS)

        Returns:
            Best action string
        """
        if available_actions is None:
            available_actions = LEARNING_ACTIONS

        state_tuple = state.to_tuple()
        best_action = 'S'  # Default
        best_q = float('-inf')

        for action in available_actions:
            key = (state_tuple, action)
            q = self.Q.get(key, NON_THORP_INIT_VALUE)
            if q > best_q:
                best_q = q
                best_action = action

        return best_action

    def get_q_value(self, state: BlackjackState, action: str) -> float:
        """Get Q-value for a state-action pair."""
        key = (state.to_tuple(), action)
        return self.Q.get(key, NON_THORP_INIT_VALUE)

    def evaluate_thorp_alignment(self, csv_path: str,
                                  tracker: Optional[PerformanceTracker] = None) -> Dict:
        """
        Evaluate agent's alignment with Thorp's strategy and track performance.

        Args:
            csv_path: Path to evaluation CSV file
            tracker: Optional PerformanceTracker to record outcomes

        Returns:
            Dictionary with alignment statistics
        """
        print(f"\nEvaluating on: {csv_path}")

        total_decisions = 0
        matches = 0
        action_breakdown = {a: {'total': 0, 'matches': 0} for a in LEARNING_ACTIONS}

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                episode = self._parse_episode_from_csv_row(row)
                if episode is None:
                    continue

                state = BlackjackState(
                    episode['player_total'],
                    episode['is_soft'],
                    episode['dealer_up'],
                    episode['true_count']
                )

                # Get agent's best action
                agent_action = self.get_best_action(state, LEARNING_ACTIONS)

                # Get Thorp's recommendation
                thorp_action = get_thorp_action(
                    episode['player_total'],
                    episode['is_soft'],
                    episode['dealer_up'],
                    can_surrender=True
                )

                # Compare
                match = (agent_action == thorp_action)
                total_decisions += 1
                if match:
                    matches += 1

                # Track by action type (agent's action)
                action_breakdown[agent_action]['total'] += 1
                if match:
                    action_breakdown[agent_action]['matches'] += 1

                # Record performance if tracker provided
                if tracker is not None:
                    tracker.record_hand(
                        profit=episode['reward'],
                        action=episode['first_action'],
                        true_count=episode['true_count'],
                        is_blackjack=episode['is_blackjack'],
                        is_surrender=episode['is_surrender']
                    )

        alignment_pct = (matches / total_decisions * 100) if total_decisions > 0 else 0.0

        results = {
            'total_decisions': total_decisions,
            'matches': matches,
            'alignment_percentage': alignment_pct,
            'action_breakdown': action_breakdown
        }

        # Print results
        print(f"\n{'='*70}")
        print("THORP STRATEGY ALIGNMENT EVALUATION")
        print(f"{'='*70}")
        print(f"Total Decisions: {total_decisions}")
        print(f"Matching Thorp:  {matches}")
        print(f"Alignment:       {alignment_pct:.2f}%")
        print(f"\nBreakdown by action:")
        for action in LEARNING_ACTIONS:
            stats = action_breakdown[action]
            if stats['total'] > 0:
                pct = stats['matches'] / stats['total'] * 100
                print(f"  {action}: {stats['matches']}/{stats['total']} ({pct:.1f}%)")
            else:
                print(f"  {action}: 0/0 (N/A)")
        print(f"{'='*70}")

        return results

    def evaluate_performance(self, csv_path: str) -> PerformanceTracker:
        """
        Evaluate pure performance (win/loss/EV) without alignment checking.

        Args:
            csv_path: Path to evaluation CSV file

        Returns:
            PerformanceTracker with recorded outcomes
        """
        tracker = PerformanceTracker()

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                episode = self._parse_episode_from_csv_row(row)
                if episode is None:
                    continue

                tracker.record_hand(
                    profit=episode['reward'],
                    action=episode['first_action'],
                    true_count=episode['true_count'],
                    is_blackjack=episode['is_blackjack'],
                    is_surrender=episode['is_surrender']
                )

        return tracker

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the learned policy.

        Returns:
            Dictionary with training stats, Q-table info, and coverage metrics
        """
        # Count states where agent disagrees with Thorp
        disagreements = 0
        total_states = 0

        # Average Q-value by action
        q_sum_by_action = {a: 0.0 for a in LEARNING_ACTIONS}
        q_count_by_action = {a: 0 for a in LEARNING_ACTIONS}

        # Check all initialized states
        checked_states = set()
        for (state_tuple, action), q_val in self.Q.items():
            if state_tuple not in checked_states:
                checked_states.add(state_tuple)
                total_states += 1

                # Get agent's best action for this state
                player_total, is_soft, dealer_up, tc = state_tuple
                state = BlackjackState(player_total, is_soft, dealer_up, tc)
                agent_action = self.get_best_action(state, LEARNING_ACTIONS)

                # Get Thorp's recommendation
                thorp_action = get_thorp_action(player_total, is_soft, dealer_up, can_surrender=True)

                if agent_action != thorp_action:
                    disagreements += 1

            # Track Q-value sums
            if action in q_sum_by_action:
                q_sum_by_action[action] += q_val
                q_count_by_action[action] += 1

        # Calculate averages
        avg_q_by_action = {}
        for action in LEARNING_ACTIONS:
            if q_count_by_action[action] > 0:
                avg_q_by_action[action] = q_sum_by_action[action] / q_count_by_action[action]
            else:
                avg_q_by_action[action] = 0.0

        # Calculate coverage (states with training data beyond prior)
        states_with_training = sum(
            1 for visits in self.state_visits.values() if visits > 0
        )

        # Calculate theoretical state space size
        # Hard: 18 totals (4-21) x 10 dealers x 11 TCs = 1980
        # Soft: 9 totals (13-21) x 10 dealers x 11 TCs = 990
        theoretical_states = (18 * 10 * 11) + (9 * 10 * 11)
        coverage_pct = (states_with_training / theoretical_states * 100) if theoretical_states > 0 else 0

        return {
            'episodes_trained': self.episodes_trained,
            'episodes_skipped': self.episodes_skipped,
            'rows_read': self.rows_read,
            'total_states_in_q_table': total_states,
            'states_with_training_data': states_with_training,
            'theoretical_state_space': theoretical_states,
            'coverage_percentage': coverage_pct,
            'states_disagreeing_with_thorp': disagreements,
            'disagreement_percentage': (disagreements / total_states * 100) if total_states > 0 else 0,
            'average_q_by_action': avg_q_by_action,
            'unique_state_action_pairs': len(self.Q),
            'prior_strength': self.prior_strength
        }

    def print_policy_sample(self, num_samples: int = 10) -> None:
        """
        Print a sample of learned policy decisions for most-visited states.

        Args:
            num_samples: Number of top states to show
        """
        print(f"\n{'='*70}")
        print(f"LEARNED POLICY SAMPLE (Top {num_samples} states by visit count)")
        print(f"{'='*70}")

        if not self.state_visits:
            print("No training data yet.")
            return

        # Get most visited states
        top_states = sorted(
            self.state_visits.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_samples]

        for state_tuple, visits in top_states:
            player_total, is_soft, dealer_up, tc = state_tuple
            state = BlackjackState(player_total, is_soft, dealer_up, tc)

            # Get agent's best action
            agent_action = self.get_best_action(state, LEARNING_ACTIONS)

            # Get Thorp's recommendation
            thorp_action = get_thorp_action(player_total, is_soft, dealer_up, can_surrender=True)

            # Get Q-values
            q_h = self.get_q_value(state, 'H')
            q_s = self.get_q_value(state, 'S')
            q_d = self.get_q_value(state, 'D')
            q_r = self.get_q_value(state, 'R')

            match_str = "MATCH" if agent_action == thorp_action else "DIFF"
            soft_str = "soft" if is_soft else "hard"

            print(f"\nState: {player_total} ({soft_str}) vs Dealer {dealer_up}, TC={tc:+d}")
            print(f"  Visits: {visits}")
            print(f"  Agent: {agent_action}  |  Thorp: {thorp_action}  [{match_str}]")
            print(f"  Q-values: H={q_h:+.3f}, S={q_s:+.3f}, D={q_d:+.3f}, R={q_r:+.3f}")

        print(f"{'='*70}\n")

    def export_policy_to_csv(self, filepath: str) -> None:
        """
        Export complete learned policy to CSV.

        Columns:
        - player_total: 4-21
        - is_soft: True/False
        - dealer_up: 2-11
        - true_count: -5 to +5
        - best_action: H/S/D/R
        - Q_hit, Q_stand, Q_double, Q_surrender: Q-values
        - thorp_action: Thorp's recommendation
        - matches_thorp: True/False
        - visit_count: Training visits for this state
        """
        print(f"\nExporting policy to: {filepath}")

        fieldnames = [
            'player_total', 'is_soft', 'dealer_up', 'true_count',
            'best_action', 'Q_hit', 'Q_stand', 'Q_double', 'Q_surrender',
            'thorp_action', 'matches_thorp', 'visit_count'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Export all states
            exported_states = set()

            for (state_tuple, _), _ in self.Q.items():
                if state_tuple in exported_states:
                    continue
                exported_states.add(state_tuple)

                player_total, is_soft, dealer_up, tc = state_tuple
                state = BlackjackState(player_total, is_soft, dealer_up, tc)

                # Get values
                best_action = self.get_best_action(state, LEARNING_ACTIONS)
                thorp_action = get_thorp_action(player_total, is_soft, dealer_up, can_surrender=True)
                visit_count = self.state_visits.get(state_tuple, 0)

                row = {
                    'player_total': player_total,
                    'is_soft': is_soft,
                    'dealer_up': dealer_up,
                    'true_count': tc,
                    'best_action': best_action,
                    'Q_hit': f"{self.get_q_value(state, 'H'):.4f}",
                    'Q_stand': f"{self.get_q_value(state, 'S'):.4f}",
                    'Q_double': f"{self.get_q_value(state, 'D'):.4f}",
                    'Q_surrender': f"{self.get_q_value(state, 'R'):.4f}",
                    'thorp_action': thorp_action,
                    'matches_thorp': best_action == thorp_action,
                    'visit_count': visit_count
                }
                writer.writerow(row)

        print(f"  Exported {len(exported_states)} states")

    def export_policy_tables(self) -> None:
        """
        Print Thorp-style ASCII tables showing agent's policy for TC=0.

        Shows:
        - Hard totals table (4-21 vs dealer 2-A)
        - Soft totals table (13-21 vs dealer 2-A)
        - Indicators where agent differs from Thorp
        """
        print(f"\n{'='*70}")
        print("AGENT POLICY TABLES (TC=0)")
        print("Legend: Action shown, * indicates difference from Thorp")
        print(f"{'='*70}")

        # Hard totals table
        print("\nHARD TOTALS:")
        print("Player |  2    3    4    5    6    7    8    9   10    A")
        print("-" * 60)

        for total in range(4, 22):
            row = f"  {total:2d}   |"
            for dealer_up in range(2, 12):
                dealer_display = "A" if dealer_up == 11 else str(dealer_up)
                state = BlackjackState(total, False, dealer_up, 0)

                agent_action = self.get_best_action(state, LEARNING_ACTIONS)
                thorp_action = get_thorp_action(total, False, dealer_up, can_surrender=True)

                if agent_action != thorp_action:
                    row += f" {agent_action}*  "
                else:
                    row += f"  {agent_action}  "
            print(row)

        # Soft totals table
        print("\nSOFT TOTALS:")
        print("Player |  2    3    4    5    6    7    8    9   10    A")
        print("-" * 60)

        for total in range(13, 22):
            ace_kicker = total - 11
            row = f" A,{ace_kicker:2d}  |"
            for dealer_up in range(2, 12):
                state = BlackjackState(total, True, dealer_up, 0)

                agent_action = self.get_best_action(state, LEARNING_ACTIONS)
                thorp_action = get_thorp_action(total, True, dealer_up, can_surrender=True)

                if agent_action != thorp_action:
                    row += f" {agent_action}*  "
                else:
                    row += f"  {agent_action}  "
            print(row)

        print(f"\n{'='*70}")

    def save_model(self, filepath: str) -> None:
        """Save the trained model to a pickle file."""
        model_data = {
            'Q': dict(self.Q),
            'N': dict(self.N),
            'state_visits': dict(self.state_visits),
            'episodes_trained': self.episodes_trained,
            'episodes_skipped': self.episodes_skipped,
            'rows_read': self.rows_read,
            'prior_strength': self.prior_strength,
            'tc_range': (self.tc_min, self.tc_max)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from a pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.Q = model_data['Q']
        self.N = defaultdict(int, model_data['N'])
        self.state_visits = defaultdict(int, model_data['state_visits'])
        self.episodes_trained = model_data['episodes_trained']
        self.episodes_skipped = model_data.get('episodes_skipped', 0)
        self.rows_read = model_data.get('rows_read', 0)
        self.prior_strength = model_data.get('prior_strength', DEFAULT_PRIOR_STRENGTH)

        tc_range = model_data.get('tc_range', (TC_MIN, TC_MAX))
        self.tc_min, self.tc_max = tc_range

        print(f"Model loaded from: {filepath}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Q-table size: {len(self.Q)} entries")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function demonstrating complete RL agent workflow."""

    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, "blackjack_data/training/blackjack_train_100k.csv")
    dev_csv = os.path.join(base_dir, "blackjack_data/development/blackjack_dev_10k.csv")
    model_path = os.path.join(base_dir, "blackjack_agents/trained_mc_model.pkl")
    policy_csv = os.path.join(base_dir, "blackjack_agents/learned_policy.csv")

    # Initialize agent with Bayesian prior
    agent = BlackjackMCAgent(prior_strength=10)

    # Train from CSV
    agent.train_from_csv(train_csv)

    # Print training statistics
    stats = agent.get_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Episodes trained: {stats['episodes_trained']}")
    print(f"  Unique states: {stats['states_with_training_data']}")
    print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
    print(f"  States disagreeing with Thorp: {stats['states_disagreeing_with_thorp']}")
    print(f"  Average Q by action:")
    for action, avg_q in stats['average_q_by_action'].items():
        print(f"    {action}: {avg_q:+.3f}")

    # Print policy sample
    agent.print_policy_sample(num_samples=10)

    # Evaluate on development set
    print("\nEvaluating on development set...")
    tracker = PerformanceTracker()
    alignment_results = agent.evaluate_thorp_alignment(dev_csv, tracker)

    # Print performance statistics
    tracker.print_summary()

    # Export policy tables
    agent.export_policy_tables()

    # Save model
    agent.save_model(model_path)

    # Export policy to CSV
    agent.export_policy_to_csv(policy_csv)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"Model saved to: {model_path}")
    print(f"Policy exported to: {policy_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
