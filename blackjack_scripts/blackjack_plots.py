import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "blackjack_data" / "training"

Q_INIT_PATH = DATA_DIR / "Q_full.csv"
Q_RL_PATH = DATA_DIR / "Q_rl.csv"


def load_Q(path):
    Q = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = int(row["player_total"])
            is_soft = row["is_soft"]          # keep as "True"/"False"
            dealer_up = int(row["dealer_up"])
            true_count = int(row["true_count"])
            action = row["action"]
            ev = float(row["EV"])
            key = (total, is_soft, dealer_up, true_count, action)
            Q[key] = ev
    return Q


def plot_Q_scatter(Q_init, Q_rl):
    xs = []
    ys = []
    for key, ev_init in Q_init.items():
        if key in Q_rl:
            ev_rl = Q_rl[key]
            xs.append(ev_init)
            ys.append(ev_rl)

    if not xs:
        print("No overlapping (state, action) pairs for scatter.")
        return

    plt.figure()
    plt.scatter(xs, ys, s=5)
    plt.xlabel("Initial Q(s,a) EV (heuristic)")
    plt.ylabel("RL Q(s,a) EV")
    plt.title("Q_full vs Q_rl (per state-action)")
    plt.axline((0, 0), slope=1)
    plt.tight_layout()
    plt.savefig("Q_init_vs_Q_rl_scatter.png", dpi=200)


def plot_Q_improvement_hist(Q_init, Q_rl):
    diffs = []
    for key, ev_init in Q_init.items():
        if key in Q_rl:
            ev_rl = Q_rl[key]
            diffs.append(ev_rl - ev_init)

    if not diffs:
        print("No overlapping (state, action) pairs for histogram.")
        return

    plt.figure()
    plt.hist(diffs, bins=50)
    plt.xlabel("Q_rl(s,a) - Q_init(s,a)")
    plt.ylabel("Count")
    plt.title("Distribution of Q Improvements (RL - Heuristic)")
    plt.tight_layout()
    plt.savefig("Q_improvement_hist.png", dpi=200)


def compute_policy(Q):
    best = {}
    for (total, is_soft, dealer_up, true_count, action), ev in Q.items():
        s = (total, is_soft, dealer_up, true_count)
        if s not in best or ev > best[s][1]:
            best[s] = (action, ev)
    return best


def plot_policy_change(Q_init, Q_rl):
    pol_init = compute_policy(Q_init)
    pol_rl = compute_policy(Q_rl)

    same = 0
    changed = 0
    for s, (a_init, _) in pol_init.items():
        if s in pol_rl:
            a_rl, _ = pol_rl[s]
            if a_init == a_rl:
                same += 1
            else:
                changed += 1

    if same + changed == 0:
        print("No overlapping states for policy comparison.")
        return

    plt.figure()
    plt.bar(["Same action", "Changed action"], [same, changed])
    plt.ylabel("Number of states")
    plt.title("Heuristic vs RL: Policy Agreement")
    plt.tight_layout()
    plt.savefig("Policy_change_bar.png", dpi=200)


def main():
    print("Loading Q_full (heuristic) from:", Q_INIT_PATH)
    Q_init = load_Q(Q_INIT_PATH)

    print("Loading Q_rl (RL) from:", Q_RL_PATH)
    Q_rl = load_Q(Q_RL_PATH)

    print("Entries: heuristic =", len(Q_init), ", RL =", len(Q_rl))

    print("Plotting Q_full vs Q_rl scatter...")
    plot_Q_scatter(Q_init, Q_rl)

    print("Plotting Q improvement histogram...")
    plot_Q_improvement_hist(Q_init, Q_rl)

    print("Plotting policy change bar chart...")
    plot_policy_change(Q_init, Q_rl)

    print("\nDone. PNGs saved in", SCRIPT_DIR)


if __name__ == "__main__":
    main()
