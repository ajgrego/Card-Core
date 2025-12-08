import csv
import ast
from collections import defaultdict

"""
Heuristic policy construction from simulator CSV data.
"""


def make_state_key(row):

    initial_hand = ast.literal_eval(row['initial_hand'])
    dealer_up = int(row['dealer_up'])
    true_count = int(row['true_count'])

    total = sum(initial_hand)
    is_soft = 11 in initial_hand

    # return the state of the hand as a tuple
    return (total, bool(is_soft), dealer_up, true_count)


def first_action(row):

    # extract first action taken from the actions_taken column
    actions_str = row['actions_taken']
    if not actions_str:
        return None
    actions_nested = ast.literal_eval(actions_str)
    if not actions_nested or not actions_nested[0]:
        return None
    return actions_nested[0][0]




def build_V_and_Q(csv_paths):

    V_sum = defaultdict(float)
    V_n = defaultdict(int)
    Q_sum = defaultdict(float)
    Q_n = defaultdict(int)

    # AI Assisted Data Processing
    for csv_path in csv_paths:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                win_str = row.get('win', '')
                if win_str is None or win_str == '':
                    continue
               
                try:
                    G = float(win_str)
                except ValueError:
                    continue

                try:
                    s = make_state_key(row)
                except Exception as e:
                    continue

                V_sum[s] += G
                V_n[s] += 1

                try:
                    a = first_action(row)
                except Exception as e:
                    a = None

                if a is not None:
                    Q_sum[(s, a)] += G
                    Q_n[(s, a)] += 1

    # AI Assited Data Processing
    
    V = {s: V_sum[s] / V_n[s] for s in V_sum} # average V(s)
    Q = {(s, a): Q_sum[(s, a)] / Q_n[(s, a)] for (s, a) in Q_sum} # average Q(s,a)
    return V, Q, V_n, Q_n # return count dictionaries 



# AI Functions to save/load V and Q to CSV Files
def save_V_csv(V, V_n, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "is_soft", "dealer_up", "true_count", "EV", "samples"])
        for (total, is_soft, dealer_up, true_count), ev in V.items():
            ev = round(ev, 3)
            samples = V_n[(total, is_soft, dealer_up, true_count)]
            writer.writerow([total, is_soft, dealer_up, true_count, ev, samples])


def save_Q_csv(Q, Q_n, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "is_soft", "dealer_up", "true_count", "action", "EV", "samples"])
        for (s, a), ev in Q.items():
            total, is_soft, dealer_up, true_count = s
            ev = round(ev, 3)
            samples = Q_n[(s, a)]
            writer.writerow([total, is_soft, dealer_up, true_count, a, ev, samples])

_Q_CACHE = None
_QN_CACHE = None
_Q_PATH = "Q_full.csv"


def load_Q_from_csv(path="Q_full.csv"):

    Q = {}
    Q_n = {}

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = int(row["player_total"])
            # is_soft was saved as True/False string
            is_soft_str = str(row["is_soft"]).strip().lower()
            is_soft = (is_soft_str == "true")
            dealer_up = int(row["dealer_up"])
            true_count = int(row["true_count"])
            action = row["action"]
            ev = float(row["EV"])
            samples = int(row["samples"])

            s = (total, is_soft, dealer_up, true_count)
            Q[(s, action)] = ev
            Q_n[(s, action)] = samples

    return Q, Q_n


def _ensure_Q_loaded(q_path=None):

    global _Q_CACHE, _QN_CACHE, _Q_PATH

    if q_path is not None and q_path != _Q_PATH:
        # If caller passes a new file path, reset cache
        _Q_CACHE = None
        _QN_CACHE = None
        _Q_PATH = q_path

    if _Q_CACHE is None or _QN_CACHE is None:
        _Q_CACHE, _QN_CACHE = load_Q_from_csv(_Q_PATH)

# AI Functions to save/load V and Q to CSV Files


# Heuristic Action Choose *Most Important Function*
def choose_action(total, is_soft, dealer_up, true_count,
                  min_samples=1, default=None, q_path=None):

    _ensure_Q_loaded(q_path) # load Q

    s = (int(total), bool(is_soft), int(dealer_up), int(true_count)) # create the state tuple


    candidates = [] # list of tuples
    for (state, action), ev in _Q_CACHE.items():
        if state == s:
            samples = _QN_CACHE[(state, action)]
            if samples >= min_samples:
                candidates.append((action, ev)) # add actions with enough samples

    if not candidates:

        if default is not None:
            return default
        return "HIT" if total < 12 else "STAND" # default fallback

    best_action, _ = max(candidates, key=lambda x: x[1]) # choose action with highest EV
    return best_action # return the best action if found



# runs heuristic simulation
def run_heuristic(csv_paths=None):

    if csv_paths is None:
        csv_paths = [r"...\training\blackjack_train_100k.csv"]
   
    V, Q, V_n, Q_n = build_V_and_Q(csv_paths)

    print("Saving V(s)")
    save_V_csv(V, V_n, "V_full.csv")

    print("Saving Q(s,a)")
    save_Q_csv(Q, Q_n, "Q_full.csv")

    print("\nDone.")



if __name__ == '__main__':
    run_heuristic()
