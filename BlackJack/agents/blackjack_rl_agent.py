"""
Blackjack Reinforcement Learning Agent
------------------------------------------------------------------------------------------------------------------------
This implements RL agents for Blackjack including:

1. BlackjackRLAgent - Monte Carlo RL agent with Thorp initialization
   - This initializes Q-values from Thorp's strategy

2. ThorpOnlyAgent - Pure heuristic agent using Thorp's strategy
   - No learning, just strategy lookup
   - Serves as a baseline for comparison

References:
    - Thorp, E. O. (1962). Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Random House.

Author: Anthony Grego
"""
#imports
import numpy as np
import pickle
import csv
import ast
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import sys
import os

#Import Thorp's strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from thorps_strategy import get_thorp_action

class BlackjackState:
    """
    Represents a state in Blackjack for the RL agent.
    includes:
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
        #Convert state to hashable tuple
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
    Monte Carlo Reinforcement Learning Agent for Blackjack.

    This agent uses first-visit Monte Carlo learning with optional Bayesian
    prior blending from Thorp's optimal basic strategy. The prior_strength
    parameter controls how much weight is given to Thorp's recommendations
    vs. observed training data.

    The Bayesian blending formula:
        Q(s,a) = (prior_strength * init_value + n * sample_mean) / (prior_strength + n)

    Attributes:
        prior_strength: Weight given to prior belief (higher = more trust in Thorp)
        Q: Q-table storing (state, action) -> expected value
        init_values: Initial Q-values for Bayesian blending
        visit_counts: Number of visits per (state, action) pair
    """
    #Hit, Stand, Double, Split, Surrender
    ACTIONS = ['H', 'S', 'D', 'P', 'R']  

    def __init__(self, prior_strength: float = 10.0):
        """
        Initializes the agent
        Args:
            prior_strength: Weight for Bayesian prior blending (0 = no prior,
                           higher values = more trust in initial values).
        """
        self.prior_strength = prior_strength
        self.Q = defaultdict(float)
        self.init_values = defaultdict(float)
        self.visit_counts = defaultdict(int)
        self.return_sums = defaultdict(float)
        self.episodes_seen = 0
        self.total_updates = 0
        self.model_type = 'rl_agent'

    def initialize_q_table_from_thorp(self) -> int:
        """
        Initialize Q-table with values derived from Thorp's optimal strategy.
        This creates a Bayesian prior where Thorp recommended actions have positive initial Q-values (+0.5) and nonrecommended actions have negative values (-0.5).
        Returns:
            Number of Q-table entries created
        """
        entries_created = 0
        for player_total in range(4, 22):
            for is_soft in [True, False]:
                if is_soft and (player_total < 12 or player_total > 21):
                    continue
                for dealer_up in range(2, 12):
                    for true_count in range(-10, 11):
                        state = BlackjackState(player_total, is_soft, dealer_up, true_count)
                        state_tuple = state.to_tuple()
                        thorp_action = get_thorp_action(
                            player_total=player_total,
                            is_soft=is_soft,
                            dealer_up=dealer_up,
                            pair_value=None,
                            can_double=True,
                            can_split=False,
                            can_surrender=True
                        )
                        for action in self.ACTIONS:
                            key = (state_tuple, action)
                            init_val = 0.5 if action == thorp_action else -0.5
                            self.init_values[key] = init_val
                            self.Q[key] = init_val
                            entries_created += 1
        return entries_created

    def initialize_q_table_zeros(self) -> int:
        """
        Initialize Q-table with zeros for all state-action pairs.

        This creates a baseline to demonstrate the 'counterfactual problem':

        THE COUNTERFACTUAL PROBLEM:
        ----------------------------
        When training data is generated using Thorp's strategy:
            1. The data only contains Thorp-recommended actions
            2. Those actions often have negative returns because of the house edge of about 2%
            3. After training: observed actions accumulate negative Q-values
            4. Unobserved actions stay at Q = 0
            5. The agent incorrectly learns to prefer unobserved actions

        Returns:
            Number of entries created
        """
        entries_created = 0
        for player_total in range(4, 22):
            for is_soft in [True, False]:
                if is_soft and (player_total < 12 or player_total > 21):
                    continue
                for dealer_up in range(2, 12):
                    for true_count in range(-10, 11):
                        state = BlackjackState(player_total, is_soft, dealer_up, true_count)
                        state_tuple = state.to_tuple()
                        for action in self.ACTIONS:
                            key = (state_tuple, action)
                            self.init_values[key] = 0.0
                            self.Q[key] = 0.0
                            entries_created += 1
        return entries_created

    def get_q_value(self, state: BlackjackState, action: str) -> float:
        #Get Q-value using Bayesian blending
        state_tuple = state.to_tuple()
        key = (state_tuple, action)
        n = self.visit_counts[key]
        init_value = self.init_values.get(key, 0.0)
        if n == 0:
            return init_value
        if self.prior_strength == 0:
            return self.return_sums[key] / n
        sample_sum = self.return_sums[key]
        return (self.prior_strength * init_value + sample_sum) / (self.prior_strength + n)

    def get_best_action(self, state: BlackjackState,available_actions: Optional[List[str]] = None) -> str:
        #Get the best action according to current Q-values
        if available_actions is None:
            available_actions = ['H', 'S', 'D', 'R']
        if not available_actions:
            return 'S'
        q_values = {action: self.get_q_value(state, action) for action in available_actions}
        return max(q_values, key=q_values.get)

    def update_q_value(self, state: BlackjackState, action: str, reward: float):
       #update Q-value using Monte Carlo first-visit update approach
        state_tuple = state.to_tuple()
        key = (state_tuple, action)
        self.return_sums[key] += reward
        self.visit_counts[key] += 1
        self.total_updates += 1
        self.Q[key] = self.get_q_value(state, action)

    def train_from_episode(self, episode_data: Dict):
        #Train from a single hand of Blackjack
        hand = episode_data['initial_hand']
        total = sum(hand)
        is_soft = 11 in hand
        dealer_up = episode_data['dealer_up']
        true_count = episode_data.get('true_count', 0)
        initial_state = BlackjackState(total, is_soft, dealer_up, true_count)
        actions = episode_data.get('actions_taken', [])
        if not actions or not actions[0]:
            return
        final_reward = episode_data['win']
        first_action_list = actions[0]
        if isinstance(first_action_list, list) and len(first_action_list) > 0:
            first_action = first_action_list[0]
        elif isinstance(first_action_list, str):
            first_action = first_action_list
        else:
            return
        if first_action == 'N' and len(first_action_list) > 1:
            first_action = first_action_list[1]
        if first_action not in self.ACTIONS:
            return
        self.update_q_value(initial_state, first_action, final_reward)
        self.episodes_seen += 1

    def train_from_csv(self, csv_path: str, max_episodes: Optional[int] = None):
        #Train the agent from a CSV file of simulation data
        print(f"\nTraining from Simulation data")
        episodes_trained = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_episodes and episodes_trained >= max_episodes:
                    break
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
                    print(f"  Processed {episodes_trained} hands...")
        print(f"Training complete: {episodes_trained} hands")
        print(f"Total Q-updates: {self.total_updates}")

    def train_no_heuristic(self, csv_path: str, max_episodes: Optional[int] = None):
        #Train from scratch with no Thorp initialization.
        print("\nConfiguring for no-heuristic training...")
        self.prior_strength = 0.0
        self.initialize_q_table_zeros()
        self.train_from_csv(csv_path, max_episodes)

    def evaluate_thorp_alignment(self, test_csv_path: str) -> Dict:
        #Evaluate agent's decisions against Thorp's optimal strategy
        total_decisions = 0
        matches = 0
        action_breakdown = defaultdict(lambda: {'total': 0, 'matches': 0})
        with open(test_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hand = ast.literal_eval(row['initial_hand'])
                total = sum(hand)
                is_soft = 11 in hand
                dealer_up = int(row['dealer_up'])
                true_count = int(row['true_count'])
                state = BlackjackState(total, is_soft, dealer_up, true_count)
                available_actions = ['H', 'S', 'D', 'R']
                agent_action = self.get_best_action(state, available_actions)
                pair_value = hand[0] if len(hand) == 2 and hand[0] == hand[1] else None
                thorp_action = get_thorp_action(
                    total, is_soft, dealer_up, pair_value,
                    can_double=True, can_split=False, can_surrender=True
                )
                match = (agent_action == thorp_action)
                total_decisions += 1
                if match:
                    matches += 1
                action_breakdown[thorp_action]['total'] += 1
                if match:
                    action_breakdown[thorp_action]['matches'] += 1
        alignment_pct = (matches / total_decisions * 100) if total_decisions > 0 else 0
        results = {'total_decisions': total_decisions,'matches': matches,'alignment_percentage': alignment_pct,'action_breakdown': dict(action_breakdown)}

        print(f"\n{'-'*60}")
        print(f"Thorp's Strategy Alignment")
        print(f"{'-'*60}")
        print(f"Total Decisions: {total_decisions}")
        print(f"Matching Thorp: {matches}")
        print(f"Alignment: {alignment_pct:.2f}%")
        print(f"{'-'*60}")
        return results

    def get_statistics(self) -> Dict:
        #Get statistics about the learned policy
        observed_pairs = len([k for k in self.visit_counts if self.visit_counts[k] > 0])
        return {
            'model_type': self.model_type,
            'prior_strength': self.prior_strength,
            'episodes_seen': self.episodes_seen,
            'total_updates': self.total_updates,
            'observed_state_action_pairs': observed_pairs,
            'trainable': True
        }

    def save_model(self, filepath: str):
        #Save the trained model to file
        model_data = {
            'model_type': self.model_type,
            'prior_strength': self.prior_strength,
            'Q': dict(self.Q),
            'init_values': dict(self.init_values),
            'return_sums': dict(self.return_sums),
            'visit_counts': dict(self.visit_counts),
            'total_updates': self.total_updates,
            'episodes_seen': self.episodes_seen
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        #Load a trained model from file
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model_type = model_data.get('model_type', 'rl_agent')
        self.prior_strength = model_data['prior_strength']
        self.Q = defaultdict(float, model_data['Q'])
        self.init_values = defaultdict(float, model_data['init_values'])
        self.return_sums = defaultdict(float, model_data['return_sums'])
        self.visit_counts = defaultdict(int, model_data['visit_counts'])
        self.total_updates = model_data['total_updates']
        self.episodes_seen = model_data['episodes_seen']

class ThorpOnlyAgent:
    #Agent that uses Thorp's strategy directly without any learning.

    ACTIONS = ['H', 'S', 'D', 'P', 'R']

    def __init__(self):
        #Initialize the heuristic only agent
        self.episodes_seen = 0
        self.model_type = 'heuristic_only'

    def get_best_action(self, state: BlackjackState,available_actions: Optional[List[str]] = None) -> str:
        #Return Thorp's recommended action for this state
        return get_thorp_action(
            player_total=state.player_total,
            is_soft=state.is_soft,
            dealer_up=state.dealer_up,
            pair_value=None,
            can_double=True,
            can_split=False,
            can_surrender=True
        )

    def evaluate_thorp_alignment(self, test_csv_path: str) -> Dict:
        #Evaluate alignment with Thorp's strategy (should be 100% by definition)
        total_decisions = 0
        matches = 0
        with open(test_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hand = ast.literal_eval(row['initial_hand'])
                total = sum(hand)
                is_soft = 11 in hand
                dealer_up = int(row['dealer_up'])
                true_count = int(row['true_count'])
                state = BlackjackState(total, is_soft, dealer_up, true_count)
                agent_action = self.get_best_action(state)
                pair_value = hand[0] if len(hand) == 2 and hand[0] == hand[1] else None
                thorp_action = get_thorp_action(
                    total, is_soft, dealer_up, pair_value,
                    can_double=True, can_split=False, can_surrender=True
                )
                total_decisions += 1
                if agent_action == thorp_action:
                    matches += 1
        alignment_pct = (matches / total_decisions * 100) if total_decisions > 0 else 0
        results = {
            'total_decisions': total_decisions,
            'matches': matches,
            'alignment_percentage': alignment_pct
        }
        print(f"\n{'-'*60}")
        print(f"Thorp's Strategy Alignment")
        print(f"{'-'*60}")
        print(f"Total Decisions: {total_decisions}")
        print(f"Matching Thorp: {matches}")
        print(f"Alignment: {alignment_pct:.2f}%")
        print(f"{'-'*60}")
        return results

    def get_statistics(self) -> Dict:
        #Return basic statistics for heuristic-only agent
        return {
            'model_type': self.model_type,
            'trainable': False
        }

def main():
    print("\n" + "-"*80)
    print("BLACKJACK RL AGENT")
    print("-"*80)
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    train_csv = os.path.join(project_dir, "data/training/blackjack_train_100k.csv")
    dev_csv = os.path.join(project_dir, "data/development/blackjack_dev_10k.csv")
    agent = BlackjackRLAgent(prior_strength=10)
    agent.initialize_q_table_from_thorp()
    agent.train_from_csv(train_csv)
    agent.evaluate_thorp_alignment(dev_csv)

if __name__ == "__main__":
    main()