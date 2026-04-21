"""
alphabeta_baseline.py - Baseline Alpha-Beta naïf pour comparaison.

Agent minimaliste utilisé uniquement comme point de comparaison pour
les expérimentations du rapport. Il N'EST PAS soumis sur Inginious.

Caractéristiques :
  - Profondeur fixe (pas d'iterative deepening)
  - Heuristique très simple (compte de pièces par couleur)
  - Pas de table de transposition
  - Pas de move ordering
"""

import random
from agent import Agent
from oxono import Game


FIXED_DEPTH = 2   # profondeur de recherche fixe
WIN_SCORE   = 100_000


class AlphaBetaBaseline(Agent):
    """Alpha-Beta minimaliste, à profondeur fixe."""

    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

        best_value  = -float('inf')
        best_action = actions[0]
        alpha, beta = -float('inf'), float('inf')

        # On mélange l'ordre pour éviter un biais dû à l'ordre des actions
        shuffled = list(actions)
        random.shuffle(shuffled)

        for action in shuffled:
            child = state.copy()
            Game.apply(child, action)
            value = -self._negamax(child, FIXED_DEPTH - 1, -beta, -alpha)
            if value > best_value:
                best_value  = value
                best_action = action
            if value > alpha:
                alpha = value
        return best_action

    def _negamax(self, state, depth, alpha, beta):
        if Game.is_terminal(state):
            return Game.utility(state, state.current_player) * WIN_SCORE
        if depth == 0:
            return self._evaluate(state, state.current_player)

        value = -float('inf')
        for action in Game.actions(state):
            child = state.copy()
            Game.apply(child, action)
            v = -self._negamax(child, depth - 1, -beta, -alpha)
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    def _evaluate(self, state, player):
        """Heuristique naïve : simple différence de pièces placées."""
        opponent = 1 - player
        my_placed  = 8 - state.pieces_o[player]   + 8 - state.pieces_x[player]
        opp_placed = 8 - state.pieces_o[opponent] + 8 - state.pieces_x[opponent]
        return my_placed - opp_placed