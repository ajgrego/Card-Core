"""
Blackjack Reinforcement Learning Agent
------------------------------------------------------------------------------------------------------------------------
This module implements a Q-learning based RL agent for Blackjack that can:
1. Learn optimal policies from simulation data
2. Integrate with Thorp's optimal basic strategy as a baseline
3. Incorporate card counting (true count) into decision-making
4. Compare performance against heuristic and optimal strategies

References:
    - Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
    - Thorp, E. O. (1962). Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Random House.
    - Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292.

Author: Anthony Grego
Date: December 4 2025
"""

import numpy as np
import pickle
import csv
import ast
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import sys
import os

# Import Thorp's strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from thorps_strategy import get_thorp_action, compare_action_to_thorp


class BlackjackState:
    """
    Represents a state in Blackjack for the RL agent.
    
    State includes:
        - player_total: Sum of player's cards (4-21)
        - is_soft: Whether hand contains a usable Ace
        - dealer_up: Dealer's upcard (2-11, where 11=Ace)
        - true_count: Card counting true count (-20 to +20 typically)
    """
    
    def __init__(self, player_total: int, is_soft: bool, dealer_up: int, true_count: int = 0):
        self.player_total = player_total
        self.is_soft = is_soft
        self.dealer_up = dealer_up
        self.true_count = true_count
    
    def to_tuple(self) -> Tuple[int, bool, int, int]:
        """Convert state to hashable tuple."""
        return (self.player_total, self.is_soft, self.dealer_up, self.true_count)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
    
    def __repr__(self):
        soft_str = "soft" if self.is_soft else "hard"
        return f"State(player={self.player_total} {soft_str}, dealer={self.dealer_up}, TC={self.true_count})"


class BlackjackRLAgent:
    """
    Q-Learning based Reinforcement Learning agent for Blackjack.
    
    This agent learns optimal action-value functions Q(s,a) through experience
    and can integrate with Thorp's optimal strategy for guidance and comparison.
    """
    
    # Possible actions in Blackjack
    ACTIONS = ['H', 'S', 'D', 'P', 'R']  # Hit, Stand, Double, Split, Surrender
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 use_thorp_initialization: bool = True):
        """
        Initialize the RL agent.
        
        Args:
            learning_rate: Alpha parameter for Q-learning (0-1)
            discount_factor: Gamma parameter for future rewards (0-1)
            epsilon: Exploration rate for epsilon-greedy policy (0-1)
            use_thorp_initialization: Initialize Q-values using Thorp's strategy
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.use_thorp_init = use_thorp_initialization
        
        # Q-table: Q[(state, action)] = expected value
        self.Q = defaultdict(float)
        
        # Visit counts for exploration tracking
        self.visit_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        
        # State value function V(s) = max_a Q(s,a)
        self.V = defaultdict(float)
        
        # Statistics for tracking learning progress
        self.total_updates = 0
        self.episodes_seen = 0
        
        print(f"Initialized BlackjackRLAgent")
        print(f"  Learning rate (α): {self.alpha}")
        print(f"  Discount factor (γ): {self.gamma}")
        print(f"  Exploration rate (ε): {self.epsilon}")
        print(f"  Thorp initialization: {self.use_thorp_init}")
    
    
    def get_q_value(self, state: BlackjackState, action: str) -> float:
        """
        Get Q-value for a state-action pair.
        
        If using Thorp initialization and Q-value hasn't been learned yet,
        returns a heuristic value based on whether action matches Thorp's strategy.
        """
        state_tuple = state.to_tuple()
        key = (state_tuple, action)
        
        if key in self.Q:
            return self.Q[key]
        
        # Initialize with Thorp's strategy guidance if enabled
        if self.use_thorp_init:
            thorp_action = get_thorp_action(
                state.player_total, 
                state.is_soft, 
                state.dealer_up
            )
            # Give bonus to actions that match Thorp's strategy
            return 0.5 if action == thorp_action else 0.0
        
        return 0.0
    
    
    def get_best_action(self, state: BlackjackState, available_actions: List[str]) -> str:
        """
        Get the best action according to current Q-values (greedy policy).
        
        Args:
            state: Current game state
            available_actions: List of legal actions in this state
            
        Returns:
            Best action string ('H', 'S', 'D', 'P', or 'R')
        """
        if not available_actions:
            return 'S'  # Default to stand if no actions available
        
        # Get Q-values for all available actions
        q_values = {action: self.get_q_value(state, action) for action in available_actions}
        
        # Return action with highest Q-value
        return max(q_values, key=q_values.get)
    
    
    def select_action(self, state: BlackjackState, available_actions: List[str], 
                     training: bool = True) -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            available_actions: List of legal actions
            training: If True, use epsilon-greedy; if False, use greedy policy
            
        Returns:
            Selected action
        """
        if not available_actions:
            return 'S'
        
        # During testing or with probability (1-epsilon), choose best action
        if not training or np.random.random() > self.epsilon:
            return self.get_best_action(state, available_actions)
        
        # Explore: choose random action
        return np.random.choice(available_actions)
    
    
    def update_q_value(self, state: BlackjackState, action: str, reward: float, 
                      next_state: Optional[BlackjackState] = None,
                      next_available_actions: Optional[List[str]] = None):
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward received
            next_state: Next state (None if terminal)
            next_available_actions: Actions available in next state
        """
        state_tuple = state.to_tuple()
        key = (state_tuple, action)
        
        current_q = self.get_q_value(state, action)
        
        # Calculate max Q-value for next state
        if next_state is None or next_available_actions is None:
            # Terminal state: no future rewards
            max_next_q = 0.0
        else:
            # Non-terminal: get max Q-value over available actions
            next_q_values = [self.get_q_value(next_state, a) for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        
        self.Q[key] = new_q
        self.state_action_counts[key] += 1
        self.total_updates += 1
        
        # Update state value function
        self.V[state_tuple] = max(
            [self.get_q_value(state, a) for a in self.ACTIONS],
            default=0.0
        )
    
    
    def train_from_episode(self, episode_data: Dict):
        """
        Train from a single episode (hand) of Blackjack.
        
        Episode data should contain:
            - initial_hand: Player's starting cards
            - dealer_up: Dealer's upcard
            - true_count: Card counting true count
            - actions_taken: List of [action] lists
            - win: Final reward (positive for win, negative for loss)
        """
        # Extract initial state
        hand = episode_data['initial_hand']
        total = sum(hand)
        is_soft = 11 in hand
        dealer_up = episode_data['dealer_up']
        true_count = episode_data.get('true_count', 0)
        
        initial_state = BlackjackState(total, is_soft, dealer_up, true_count)
        
        # Extract actions taken
        actions = episode_data.get('actions_taken', [])
        if not actions or not actions[0]:
            return
        
        # Get final reward
        final_reward = episode_data['win']
        
        # Extract first action from the list format [[action], [action], ...]
        # The simulator stores actions as lists within lists
        first_action_list = actions[0]
        if isinstance(first_action_list, list) and len(first_action_list) > 0:
            first_action = first_action_list[0]
        elif isinstance(first_action_list, str):
            first_action = first_action_list
        else:
            return  # Invalid action format
        
        # Only train on valid action types
        if first_action not in self.ACTIONS:
            return
        
        # Train on the sequence of state-action pairs
        # For simplicity, we assign the final reward to the first action
        # (This is a Monte Carlo approach)
        self.update_q_value(
            initial_state,
            first_action,
            final_reward,
            next_state=None,  # Terminal state
            next_available_actions=None
        )
        
        self.visit_counts[initial_state.to_tuple()] += 1
        self.episodes_seen += 1
    
    
    def train_from_csv(self, csv_path: str, max_episodes: Optional[int] = None):
        """
        Train the agent from a CSV file of simulation data.
        
        Args:
            csv_path: Path to CSV file with columns:
                - initial_hand
                - dealer_up
                - true_count
                - actions_taken
                - win
            max_episodes: Maximum number of episodes to train on (None = all)
        """
        print(f"\nTraining from CSV: {csv_path}")
        
        episodes_trained = 0
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if max_episodes and episodes_trained >= max_episodes:
                    break
                
                # Parse episode data
                episode = {
                    'initial_hand': ast.literal_eval(row['initial_hand']),
                    'dealer_up': int(row['dealer_up']),
                    'true_count': int(row['true_count']),
                    'actions_taken': ast.literal_eval(row.get('actions_taken', '[]')),
                    'win': float(row['win'])
                }
                
                self.train_from_episode(episode)
                episodes_trained += 1
                
                if episodes_trained % 10000 == 0:
                    print(f"  Processed {episodes_trained} episodes...")
        
        print(f"✓ Training complete: {episodes_trained} episodes")
        print(f"  Total Q-updates: {self.total_updates}")
        print(f"  Unique states visited: {len(self.visit_counts)}")
        print(f"  Unique state-action pairs: {len(self.Q)}")
    
    
    def evaluate_against_thorp(self, test_csv_path: str) -> Dict:
        """
        Evaluate agent's decisions against Thorp's optimal strategy.
        
        Returns dictionary with:
            - total_decisions: Number of decisions evaluated
            - matches: Number of decisions matching Thorp
            - alignment_percentage: Percentage of matching decisions
            - action_breakdown: Breakdown by action type
        """
        print(f"\nEvaluating against Thorp's strategy using: {test_csv_path}")
        
        total_decisions = 0
        matches = 0
        action_breakdown = defaultdict(lambda: {'total': 0, 'matches': 0})
        
        with open(test_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse state
                hand = ast.literal_eval(row['initial_hand'])
                total = sum(hand)
                is_soft = 11 in hand
                dealer_up = int(row['dealer_up'])
                true_count = int(row['true_count'])
                
                state = BlackjackState(total, is_soft, dealer_up, true_count)
                
                # Get RL agent's choice
                available_actions = ['H', 'S', 'D']  # Simplified for comparison
                rl_action = self.get_best_action(state, available_actions)
                
                # Get Thorp's recommendation
                pair_value = hand[0] if len(hand) == 2 and hand[0] == hand[1] else None
                thorp_action = get_thorp_action(total, is_soft, dealer_up, pair_value)
                
                # Compare
                match = (rl_action == thorp_action)
                
                total_decisions += 1
                if match:
                    matches += 1
                
                action_breakdown[rl_action]['total'] += 1
                if match:
                    action_breakdown[rl_action]['matches'] += 1
        
        alignment_pct = (matches / total_decisions * 100) if total_decisions > 0 else 0
        
        results = {
            'total_decisions': total_decisions,
            'matches': matches,
            'alignment_percentage': alignment_pct,
            'action_breakdown': dict(action_breakdown)
        }
        
        print(f"\n{'='*60}")
        print(f"THORP STRATEGY ALIGNMENT EVALUATION")
        print(f"{'='*60}")
        print(f"Total Decisions: {total_decisions}")
        print(f"Matching Thorp: {matches}")
        print(f"Alignment: {alignment_pct:.2f}%")
        print(f"\nAction Breakdown:")
        for action, stats in action_breakdown.items():
            action_pct = (stats['matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {action}: {stats['matches']}/{stats['total']} ({action_pct:.1f}%)")
        print(f"{'='*60}\n")
        
        return results
    
    
    def save_model(self, filepath: str):
        """Save the trained Q-table and parameters to file."""
        model_data = {
            'Q': dict(self.Q),
            'V': dict(self.V),
            'visit_counts': dict(self.visit_counts),
            'state_action_counts': dict(self.state_action_counts),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_seen': self.episodes_seen
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    
    def load_model(self, filepath: str):
        """Load a trained Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.Q = defaultdict(float, model_data['Q'])
        self.V = defaultdict(float, model_data['V'])
        self.visit_counts = defaultdict(int, model_data['visit_counts'])
        self.state_action_counts = defaultdict(int, model_data['state_action_counts'])
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.total_updates = model_data['total_updates']
        self.episodes_seen = model_data['episodes_seen']
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Episodes trained: {self.episodes_seen}")
        print(f"  Q-table size: {len(self.Q)} entries")
    
    
    def get_statistics(self) -> Dict:
        """Get statistics about the learned policy."""
        return {
            'total_updates': self.total_updates,
            'episodes_seen': self.episodes_seen,
            'unique_states': len(self.visit_counts),
            'q_table_size': len(self.Q),
            'avg_visits_per_state': np.mean(list(self.visit_counts.values())) if self.visit_counts else 0,
            'most_visited_state': max(self.visit_counts, key=self.visit_counts.get) if self.visit_counts else None
        }
    
    
    def print_policy_sample(self, num_samples: int = 10):
        """Print a sample of learned policy decisions."""
        print(f"\n{'='*70}")
        print(f"LEARNED POLICY SAMPLE (Top {num_samples} states by visit count)")
        print(f"{'='*70}")
        
        # Get most visited states
        top_states = sorted(self.visit_counts.items(), key=lambda x: x[1], reverse=True)[:num_samples]
        
        for state_tuple, visits in top_states:
            total, is_soft, dealer_up, tc = state_tuple
            state = BlackjackState(total, is_soft, dealer_up, tc)
            
            # Get RL agent's best action
            rl_action = self.get_best_action(state, ['H', 'S', 'D'])
            
            # Get Thorp's recommendation
            thorp_action = get_thorp_action(total, is_soft, dealer_up)
            
            # Get Q-values
            q_h = self.get_q_value(state, 'H')
            q_s = self.get_q_value(state, 'S')
            q_d = self.get_q_value(state, 'D')
            
            match_str = "✓" if rl_action == thorp_action else "✗"
            soft_str = "soft" if is_soft else "hard"
            
            print(f"\nState: {total} ({soft_str}) vs Dealer {dealer_up}, TC={tc:+d}")
            print(f"  Visits: {visits}")
            print(f"  RL Action: {rl_action}  |  Thorp: {thorp_action}  {match_str}")
            print(f"  Q-values: H={q_h:.3f}, S={q_s:.3f}, D={q_d:.3f}")
        
        print(f"{'='*70}\n")


def main():
    """Main function demonstrating RL agent usage."""
    print("\n" + "="*70)
    print("BLACKJACK RL AGENT - TRAINING AND EVALUATION")
    print("="*70)
    
    # Initialize agent
    agent = BlackjackRLAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        use_thorp_initialization=True
    )
    
    # Train from simulation data
    train_csv = "/mnt/project/blackjack_train_100k.csv"
    agent.train_from_csv(train_csv, max_episodes=100000)
    
    # Print statistics
    stats = agent.get_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Episodes seen: {stats['episodes_seen']}")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Unique states: {stats['unique_states']}")
    print(f"  Q-table size: {stats['q_table_size']}")
    print(f"  Avg visits/state: {stats['avg_visits_per_state']:.2f}")
    
    # Show sample policy
    agent.print_policy_sample(num_samples=10)
    
    # Evaluate against Thorp's strategy
    results = agent.evaluate_against_thorp(train_csv)
    
    # Save model
    agent.save_model("/home/claude/blackjack_rl_model.pkl")
    
    print("\n✓ Training and evaluation complete!")


if __name__ == "__main__":
    main()
