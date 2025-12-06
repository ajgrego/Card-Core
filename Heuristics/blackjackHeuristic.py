import csv
import ast
import argparse
from collections import defaultdict


def extract_state(row):
    """Convert a CSV row into a blackjack state tuple."""
    hand = ast.literal_eval(row['initial_hand'])
    total = sum(hand)
    is_soft = 11 in hand
    dealer_card = int(row['dealer_up'])
    tc = int(row['true_count'])
    return (total, is_soft, dealer_card, tc)


def extract_first_action(row):
    """Extract the player's first action from the actions list."""
    raw = row.get('actions_taken')
    if not raw:
        return None
    actions = ast.literal_eval(raw)
    if not actions or not actions[0]:
        return None
    return actions[0][0]


def compute_ev_tables(csv_path):
    """Build V(s) and Q(s,a) from the simulation file."""

    V_total = defaultdict(float)
    V_count = defaultdict(int)
    Q_total = defaultdict(float)
    Q_count = defaultdict(int)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row.get('win'):
                continue

            win = float(row['win'])
            state = extract_state(row)

            V_total[state] += win
            V_count[state] += 1

            action = extract_first_action(row)
            if action is not None:
                Q_total[(state, action)] += win
                Q_count[(state, action)] += 1

    V = {s: V_total[s] / V_count[s] for s in V_total}
    Q = {(s, a): Q_total[(s, a)] / Q_count[(s, a)] for (s, a) in Q_total}

    return V, Q, V_count, Q_count


def show_best_worst_V(V, V_count, n=10):
    if not V:
        print("No V(s) values.")
        return

    ordered = sorted(V.items(), key=lambda x: x[1])

    print("\n--- Worst States ---")
    for state, ev in ordered[:n]:
        print(f"{state} -> EV {ev:.3f} (samples: {V_count[state]})")

    print("\n--- Best States ---")
    for state, ev in ordered[-n:]:
        print(f"{state} -> EV {ev:.3f} (samples: {V_count[state]})")


def show_best_worst_Q(Q, Q_count, n=10):
    if not Q:
        print("No Q(s,a) values.")
        return

    ordered = sorted(Q.items(), key=lambda x: x[1])

    print("\n--- Worst State/Action ---")
    for (state, action), ev in ordered[:n]:
        print(f"{state}, {action} -> EV {ev:.3f} (samples: {Q_count[(state, action)]})")

    print("\n--- Best State/Action ---")
    for (state, action), ev in ordered[-n:]:
        print(f"{state}, {action} -> EV {ev:.3f} (samples: {Q_count[(state, action)]})")


def save_V_csv(V, V_count, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "is_soft", "dealer_up", "true_count", "EV", "samples"])
        for (total, is_soft, dealer_up, tc), ev in V.items():
            writer.writerow([total, is_soft, dealer_up, tc, round(ev, 3),
                             V_count[(total, is_soft, dealer_up, tc)]])


def save_Q_csv(Q, Q_count, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "is_soft", "dealer_up", "true_count", "action", "EV", "samples"])
        for (state, action), ev in Q.items():
            total, is_soft, dealer_up, tc = state
            writer.writerow([total, is_soft, dealer_up, tc, action,
                             round(ev, 3), Q_count[(state, action)]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="blackjack_dev.csv", help="Path to simulation CSV")
    args = parser.parse_args()

    print(f"\nReading data from {args.input}")

    V, Q, V_count, Q_count = compute_ev_tables(args.input)

    print(f"\nBuilt {len(V)} states (V)")
    print(f"Built {len(Q)} state/action pairs (Q)")

    show_best_worst_V(V, V_count)
    show_best_worst_Q(Q, Q_count)

    print("\nSaving CSV files...")
    save_V_csv(V, V_count, "V_full.csv")
    save_Q_csv(Q, Q_count, "Q_full.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
