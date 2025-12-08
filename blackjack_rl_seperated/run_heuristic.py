from blackjack_simulator import Table
from blackjackHeuristic import choose_action

"""
Uses the learned heuristic to play blackhands and evaluate overall performance.
"""


def heuristic_policy(table, available_actions):

    # extract all needed state info
    player_total = table.curr().value() 
    is_soft = table.curr().is_soft()
    dealer_up = table.dealer_hand.cards[0]
    true_count = table.true_count()

    # grab heuristic action from BlackJackHeuristic
    action = choose_action(
        player_total,
        is_soft,
        dealer_up,
        true_count,
        min_samples=5,
        default='S'
    )

    # ensure action is valid
    if action in available_actions:
        return action

    # else case : return first available action
    return available_actions[0]


def play_hand(decks=8, penetration=6.5):

    # works off Blackjack_Sim 
    table = Table() 
    table.new_shoe(decks=decks, penetration=penetration)
    table.initial_deal()

    # continue until hand is over
    while table.curr_idx is not None:
        actions = table.available_actions()
        if not actions:
            break

        action = heuristic_policy(table, actions)
        table.do_action(action)

    # only return final win amount
    return table.results()['win']


def main():
    num_hands = 500_000 # Can be adjusted as seen fit
    total_win = 0.0

    for _ in range(num_hands):
        total_win += play_hand()

    avg_win = total_win / num_hands
    print(f"Heuristic Average Win : Over {num_hands} hands: {avg_win:.3f}")


if __name__ == '__main__':
    main()
