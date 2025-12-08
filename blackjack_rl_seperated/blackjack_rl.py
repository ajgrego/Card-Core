import random
from collections import defaultdict

from blackjack_simulator import Table 

"""
Q-learning implementation for Blackjack.

Learned off YouTube and reverse engineered with AI assistance.
"""



# Map current table state to a tuple for Q learning
def get_state(table: Table):

    total = table.curr().value()
    is_soft = table.curr().is_soft()
    dealer_up = table.dealer_hand.cards[0]
    true_count = table.true_count()
    return (total, is_soft, dealer_up, true_count)



# ε-greedy action selection
def epsilon_greedy_action(state, available_actions, Q, epsilon):
    if not available_actions:
        return None

    # explore if random < ε
    if random.random() < epsilon:
        return random.choice(available_actions)

    # choose best action based on Q values
    q_vals = [Q[(state, a)] for a in available_actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(available_actions, q_vals) if q == max_q]
    return random.choice(best_actions)


# Q-learning training function
# Assisted by AI (Parts will be identifed with comments)
def train_q_learning(num_episodes=200_000, alpha=0.05,gamma=0.99, epsilon_start=1.0,
    epsilon_end=0.05,decks=8,penetration=6.5,
):

    Q = defaultdict(float)

    table = Table()
    table.new_shoe(decks=decks, penetration=penetration)

    for episode_idx in range(num_episodes):

        # AI Assisted Code
        frac = episode_idx / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        # AI Assisted Code

        if table.shuffle_pending:
            table.new_shoe(decks=decks, penetration=penetration)

        # start a new hand
        table.initial_deal()

        while True:
            if table.curr_idx is None:
                r = table.results()['win']
                break

            s = get_state(table)
            actions = table.available_actions()
            if not actions:
                table.curr_idx = None
                continue

            a = epsilon_greedy_action(s, actions, Q, epsilon)

            # take action
            table.do_action(a)

            if table.curr_idx is None:
                r = table.results()['win']
                target = r 

            # AI Assisted Code 
            else:
                r = 0.0
                s_next = get_state(table)
                next_actions = table.available_actions()
                if next_actions:
                    max_next_q = max(Q[(s_next, a_next)] for a_next in next_actions)
                else:
                    max_next_q = 0.0
                target = r + gamma * max_next_q
            # AI Assisted Code 

            # Q-learning update
            old_q = Q[(s, a)]

            # *Reinforced Learning Update*
            Q[(s, a)] = old_q + alpha * (target - old_q)

    return Q



# select greedy action based on learned Q
def greedy_action(table: Table, Q):
    actions = table.available_actions()
    if not actions:
        return None
    # get current state
    s = get_state(table)
    q_vals = [Q[(s, a)] for a in actions]
    max_q = max(q_vals)
    # select actions with max Q value
    best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
    # select one of the best actions randomly / deals with ties in Q values
    return random.choice(best_actions)


# check the learned policy performance
def evaluate_policy(Q, num_episodes=50_000, decks=8, penetration=6.5):
    table = Table()
    table.new_shoe(decks=decks, penetration=penetration)

    total_win = 0.0

    for _ in range(num_episodes):
        if table.shuffle_pending:
            table.new_shoe(decks=decks, penetration=penetration)

        table.initial_deal()

        while table.curr_idx is not None:
            a = greedy_action(table, Q)
            if a is None:
                break
            table.do_action(a)

        total_win += table.results()['win']

    # return average win per hand
    return total_win / num_episodes




if __name__ == "__main__":
    NUM_TRAIN = 600_000
    print(f"Training Q-learning : {NUM_TRAIN} hands")
    Q = train_q_learning(num_episodes=NUM_TRAIN)

    NUM_EVAL = 100_000
    avg_win = evaluate_policy(Q, num_episodes=NUM_EVAL)
    print(f"Q-learning average win over {NUM_EVAL} hands: {avg_win:.3f}")
