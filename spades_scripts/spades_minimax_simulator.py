from dataclasses import dataclass, asdict
import os
from typing import List, Dict, Tuple
import json, random, itertools
from collections import defaultdict
import os
from spades_moves import SpadesMoves
from spades_minimax import minimax_choose_move
from card import Card

game_state = SpadesMoves()


SUITS = ["C","D","H","S"]  # Spades are trump
RANKS = list(range(2, 15)) # 11=J, 12=Q, 13=K, 14=A

def deck():
    return [Card(i) for i in range(52)]

class GameLogger:
    def __init__(self, path_jsonl="spades_logs.jsonl"):
        self.path = path_jsonl
        self.buffer = []
        self.game_id = None
        self.hand_id = 0

    def start_game(self, game_id, rng_seed):
        self.game_id, self.hand_id = game_id, 0
        self._write({"event":"game_start","game_id":game_id,"rng_seed":rng_seed})

    def log_deal(self, hands, scores, seats=("N","E","S","W")):
        self.hand_id += 1
        payload = {
            "event":"deal","game_id":self.game_id,"hand_id":self.hand_id,
            "scores":scores, "seats":seats,
            # store open hands for research; strip later if training imperfect-info
            "hands": [[f"{c.suite}{c.order}" for c in h] for h in hands]
        }
        self._write(payload)

    def log_bid(self, seat, hand_feats, bid, nil=False):
        self._write({
            "event":"bid","game_id":self.game_id,"hand_id":self.hand_id,
            "seat":seat, "features":hand_feats, "bid":bid, "nil":nil
        })

    def log_play(self, trick_id, seat, legal_moves, chosen, lead_suit, spades_broken, belief=None):
        self._write({
            "event":"play","game_id":self.game_id,"hand_id":self.hand_id,"trick_id":trick_id,
            "seat":seat, "lead_suit":lead_suit, "spades_broken":spades_broken,
            "legal_moves":[f"{c.suite}{c.order}" for c in legal_moves], "chosen":f"{chosen.suite}{chosen.order}",
            "belief":belief or {}
        })

    def log_trick_end(self, trick_id, winner_seat, cards_played):
        self._write({
            "event":"trick_end","game_id":self.game_id,"hand_id":self.hand_id,
            "trick_id":trick_id, "winner_seat":winner_seat,
            "cards":[(s, f"{c.suite}{c.order}") for s,c in cards_played]
        })

    def log_hand_end(self, tricks_won, bids, score_delta, totals):
        self._write({
            "event":"hand_end","game_id":self.game_id,"hand_id":self.hand_id,
            "tricks_won":tricks_won, "bids":bids,
            "score_delta":score_delta, "totals":totals
        })

    def _write(self, obj):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj)+"\n")

# --- Simple bots for self-play ---
def legal_moves(hand: List[Card], lead_suit: int, spades_broken: bool):
    if lead_suit is None:
        # cannot lead spades unless broken or only spades remain
        non_spades = [c for c in hand if c.suiteID != 3]
        if non_spades:
            return non_spades
        return hand
    follows = [c for c in hand if c.suiteID == lead_suit]
    return follows if follows else hand

def greedy_bot_choose(hand, lead_suit, spades_broken):
    moves = legal_moves(hand, lead_suit, spades_broken)
    # trivial heuristic: if following suit, play the lowest; else dump highest non-spade, else lowest spade
    following = [c for c in moves if lead_suit is not None and c.suiteID == lead_suit]
    if following: return min(following, key=lambda c:c.orderID)
    non_spades = [c for c in moves if c.suiteID != 3]
    if non_spades: return max(non_spades, key=lambda c:c.orderID)
    return min(moves, key=lambda c:c.orderID)

# --- Example self-play loop stub (play phase only) ---
def play_one_hand(logger: GameLogger, rng: random.Random):
    d = deck(); rng.shuffle(d)
    hands = [sorted(d[i*13:(i+1)*13], key=lambda c:(c.suiteID,c.orderID)) for i in range(4)]
    scores = [0,0]  # team NS vs EW (example)
    logger.log_deal(hands, scores)

    # TODO: bidding; for now, fixed bids as a placeholder
    bids = [3,3,3,3]
    for seat in range(4):
        hand_feats = {
            "spade_len": sum(1 for c in hands[seat] if c.suiteID==3),
            "aces": sum(1 for c in hands[seat] if c.orderID==14),
        }
        logger.log_bid(seat, hand_feats, bids[seat], nil=False)

    spades_broken = False
    leader = 0
    tricks_won = [0,0,0,0]

    
    game_state.state["hands"] = hands
    game_state.state["tricks_won"] = tricks_won

    for trick_id in range(13):
        trick = []
        lead_suit = None
        for i in range(4):
            seat = (leader + i) % 4
            if seat == 0:
                move = minimax_choose_move(game_state, hero_seat=0, depth=2)
            else:
                move = greedy_bot_choose(hands[seat], lead_suit, spades_broken)

            if lead_suit is None: lead_suit = move.suiteID
            if move.suiteID == 3: spades_broken = True
            logger.log_play(trick_id, seat, legal_moves(hands[seat], lead_suit, spades_broken), move, lead_suit, spades_broken)
            hands[seat].remove(move)
            trick.append((seat, move))

        # decide winner
        led = lead_suit
        winning = None
        for s, c in trick:
            if winning is None:
                winning = (s,c)
            else:
                ws, wc = winning
                if (c.suiteID == 3 and wc.suiteID != 3) or \
                   (c.suiteID == wc.suiteID and c.orderID > wc.orderID) or \
                   (c.suiteID == 3 and wc.suiteID == 3 and c.orderID > wc.orderID):
                    winning = (s,c)
        winner_seat, _ = winning
        tricks_won[winner_seat] += 1
        logger.log_trick_end(trick_id, winner_seat, trick)
        leader = winner_seat

    # simple scoring placeholder
    team_tricks = [tricks_won[0]+tricks_won[2], tricks_won[1]+tricks_won[3]]
    score_delta = [team_tricks[0], team_tricks[1]]
    totals = score_delta  # if per-hand only
    logger.log_hand_end(tricks_won, bids, score_delta, totals)

PROJECT_ROOT = os.path.dirname(__file__)
LOG_PATH = os.path.join(PROJECT_ROOT, "spades_minimax_data", "spades_logs.jsonl")

if __name__ == "__main__":
    NUM_GAMES = 1000

    print("CURRENT WORKING DIRECTORY:", os.getcwd())
    print("Log file will be written to:", LOG_PATH)

    logger = GameLogger(LOG_PATH)
    rng = random.Random(42)

    for game_id in range(1, NUM_GAMES + 1):
        if game_id % 100 == 0:
            print(f"Simulating game {game_id}/{NUM_GAMES}...")

        logger.start_game(game_id=game_id, rng_seed=game_id)
        play_one_hand(logger, rng)

    print("Done!")
