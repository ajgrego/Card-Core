import math
from heuristic_spades_ai import HeuristicAI
from spades_moves import SpadesMoves

heuristic_spades_ai = HeuristicAI()
spades_moves = SpadesMoves()

def team_of(seat: int) -> int:
    """
    Team 0 = seats 0 & 2 (e.g., N/S), Team 1 = seats 1 & 3 (e.g., E/W).
    Change this if your team mapping is different.
    """
    return 0 if seat in (0, 2) else 1

def next_seat(seat: int) -> int:
    """Assuming seats are 0,1,2,3 in clockwise order."""
    return (seat + 1) % 4


def minimax_value(state,
                  seat_to_act: int,
                  hero_seat: int,
                  depth: int,
                  alpha: float,
                  beta: float) -> float:
    """
    Compute the minimax value of `state` from the perspective of hero_seat.
    Positive values = good for hero's team, negative = bad.
    """

    # Base case: depth limit or game/hand over
    if depth == 0 or len(state.state["hands"][seat_to_act]) == 0:
        return heuristic_spades_ai.evaluate_game(state.state["tricks_won"],[0,0,0,0], hero_seat)

    legal_moves = state.getAllValidMoves(state.state["hands"][seat_to_act])

    # If no moves, treat as terminal (shouldn't usually happen if rules are correct)
    if not legal_moves:
        return heuristic_spades_ai.evaluate_game(state.state["tricks_won"], [0,0,0,0], hero_seat)

    is_max_node = (team_of(seat_to_act) == team_of(hero_seat))

    if is_max_node:
        value = -math.inf
        for move in legal_moves:
            state.makeMove(seat_to_act, move)
            child_val = minimax_value(
                state,
                next_seat(seat_to_act),
                hero_seat,
                depth - 1,
                alpha,
                beta
            )
            state.undoMove()
            value = max(value, child_val)
            alpha = max(alpha, value)
            if beta <= alpha:  # alpha–beta pruning
                break
        return value
    else:
        value = math.inf
        for move in legal_moves:
            state.makeMove(seat_to_act, move)
            child_val = minimax_value(
                state,
                next_seat(seat_to_act),
                hero_seat,
                depth - 1,
                alpha,
                beta
            )
            state.undoMove()
            value = min(value, child_val)
            beta = min(beta, value)
            if beta <= alpha:  # alpha–beta pruning
                break
        return value


def minimax_choose_move(state,
                        hero_seat: int,
                        depth: int = 2):
    """
    Pick the best move for hero_seat using depth-limited minimax.
    Increase `depth` for stronger (but slower) play.
    """
    legal_moves = state.getAllValidMoves(state.state["hands"][hero_seat])
    if not legal_moves:
        return None  # no legal move available

    best_move = None
    best_val = -math.inf

    for move in legal_moves:
        state.makeMove(hero_seat, move)
        val = minimax_value(
            state,
            next_seat(hero_seat),
            hero_seat,
            depth - 1,
            alpha=-math.inf,
            beta=math.inf
        )
        state.undoMove()
        if val > best_val:
            best_val = val
            best_move = move

    return best_move
