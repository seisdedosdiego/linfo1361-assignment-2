"""
LINFO1361 - Intelligence Artificielle, Assignment 2
Auteurs : Seisdedos Stoz Diego (4659-23-00), Muylkens Justin (8004-22-00)
"""

import time
import random as _random_module
from agent import Agent
from oxono import Game


# ----------------------------------------------------------------------
# Poids de la fonction d'évaluation heuristique
# ----------------------------------------------------------------------
COLOR_WEIGHTS_SELF  = (0, 1, 10, 500)
COLOR_WEIGHTS_OPP   = (0, 1, 10, 100)
SYMBOL_WEIGHTS_SELF = (0, 1,  5, 200)
SYMBOL_WEIGHTS_OPP  = (0, 1,  5,  60)
FORK_SELF_BONUS  = 5000
FORK_OPP_PENALTY = 2000
WIN_SCORE = 100_000


# ----------------------------------------------------------------------
# Pré-calcul des 36 lignes potentiellement gagnantes sur le plateau 6x6
# ----------------------------------------------------------------------
_LINES = []
for _r in range(6):
    for _c in range(3):
        _LINES.append(tuple((_r, _c + i) for i in range(4)))
for _c in range(6):
    for _r in range(3):
        _LINES.append(tuple((_r + i, _c) for i in range(4)))
_LINES = tuple(_LINES)


# ----------------------------------------------------------------------
# Zobrist hashing - clés aléatoires 64 bits pour chaque composant d'état
# ----------------------------------------------------------------------
# Un hash Zobrist représente un état comme le XOR d'une clé aléatoire
# pour chaque composant (pièce sur case, position du totem, joueur à
# jouer). C'est extrêmement rapide à calculer et les collisions sur
# 64 bits sont négligeables en pratique.
_rng = _random_module.Random(42)   # seed fixe pour la reproductibilité
_ZOBRIST_PIECE = [[
    {
        ('x', 0): _rng.getrandbits(64),
        ('x', 1): _rng.getrandbits(64),
        ('o', 0): _rng.getrandbits(64),
        ('o', 1): _rng.getrandbits(64),
    }
    for _ in range(6)
] for _ in range(6)]
_ZOBRIST_TOTEM_O = [[_rng.getrandbits(64) for _ in range(6)] for _ in range(6)]
_ZOBRIST_TOTEM_X = [[_rng.getrandbits(64) for _ in range(6)] for _ in range(6)]
_ZOBRIST_PLAYER  = _rng.getrandbits(64)   # XORé quand current_player == 1


def _zobrist_hash(state):
    """Calcule le hash Zobrist d'un état en partant de zéro."""
    h = 0
    board = state.board
    for r in range(6):
        for c in range(6):
            cell = board[r][c]
            if cell is not None:
                h ^= _ZOBRIST_PIECE[r][c][cell]
    h ^= _ZOBRIST_TOTEM_O[state.totem_O[0]][state.totem_O[1]]
    h ^= _ZOBRIST_TOTEM_X[state.totem_X[0]][state.totem_X[1]]
    if state.current_player == 1:
        h ^= _ZOBRIST_PLAYER
    return h


# ----------------------------------------------------------------------
# Table de transposition
# ----------------------------------------------------------------------
# Pour chaque état, on stocke (profondeur_recherche, valeur, flag, meilleur_coup).
# Les flags indiquent si la valeur est exacte ou seulement une borne
# (résultat d'une coupure alpha-beta).
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

# Taille maximale de la TT (nombre d'entrées) pour respecter la limite
# de 2 GB de RAM de l'énoncé. Au-delà, on vide la table complètement.
# ~500k entrées * ~100 octets = ~50 MB, bien en dessous de la limite.
_TT_MAX_ENTRIES = 500_000


class MyAgent(Agent):
    """
    Agent Oxono : Negamax Alpha-Beta avec approfondissement itératif,
    table de transposition et move ordering par meilleur coup TT.
    """

    def __init__(self, player):
        super().__init__(player)
        self._safety_margin = 0.5
        # La table de transposition est persistante entre les appels à act()
        # car les mêmes positions peuvent être ré-évaluées au fil de la partie.
        self._tt = {}

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------
    def act(self, state, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

        # Nettoyage de la TT si elle devient trop grosse (limite mémoire).
        if len(self._tt) > _TT_MAX_ENTRIES:
            self._tt.clear()

        # Budget de temps pour ce coup
        my_pieces  = state.pieces_o[self.player] + state.pieces_x[self.player]
        moves_left = max(my_pieces, 1)
        time_budget = max(
            remaining_time / (moves_left + 1) - self._safety_margin,
            0.05,
        )
        deadline = time.perf_counter() + time_budget

        # Approfondissement itératif
        best_action = actions[0]
        depth = 1
        while depth <= 20:
            try:
                action = self._search_root(state, depth, deadline)
                if action is not None:
                    best_action = action
            except TimeoutError:
                break
            if time.perf_counter() >= deadline:
                break
            depth += 1
        return best_action

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------
    def _search_root(self, state, depth, deadline):
        """Recherche à la racine : essaie d'abord le meilleur coup connu (TT)."""
        key = _zobrist_hash(state)
        tt_entry = self._tt.get(key)
        tt_move = tt_entry[3] if tt_entry is not None else None

        actions = self._ordered_actions(state, tt_move)

        best_action = None
        best_value  = -float('inf')
        alpha, beta = -float('inf'), float('inf')

        for action in actions:
            if time.perf_counter() > deadline:
                raise TimeoutError
            child = state.copy()
            Game.apply(child, action)
            value = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if value > best_value:
                best_value  = value
                best_action = action
            if value > alpha:
                alpha = value

        # Stocke le résultat racine dans la TT (flag EXACT car sans coupure).
        self._tt[key] = (depth, best_value, EXACT, best_action)
        return best_action

    def _negamax(self, state, depth, alpha, beta, deadline):
        if time.perf_counter() > deadline:
            raise TimeoutError

        alpha_orig = alpha
        key = _zobrist_hash(state)

        # ---- Consultation de la TT ----
        tt_entry = self._tt.get(key)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_value, tt_flag, tt_move = tt_entry
            # On n'utilise la valeur que si elle a été calculée à une
            # profondeur au moins aussi grande que celle qu'on demande.
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_value
                elif tt_flag == LOWERBOUND and tt_value > alpha:
                    alpha = tt_value
                elif tt_flag == UPPERBOUND and tt_value < beta:
                    beta = tt_value
                if alpha >= beta:
                    return tt_value

        # ---- Feuille : état terminal ou profondeur atteinte ----
        if Game.is_terminal(state):
            return Game.utility(state, state.current_player) * WIN_SCORE
        if depth == 0:
            return self._evaluate(state, state.current_player)

        # ---- Exploration récursive avec move ordering ----
        actions = self._ordered_actions(state, tt_move)

        value = -float('inf')
        best_move = None
        for action in actions:
            child = state.copy()
            Game.apply(child, action)
            v = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if v > value:
                value = v
                best_move = action
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break   # coupure beta

        # ---- Stockage dans la TT avec le bon flag ----
        # UPPERBOUND : toutes les branches ont fait moins bien que alpha_orig
        #              -> la valeur réelle est au plus `value`.
        # LOWERBOUND : coupure beta -> la valeur réelle est au moins `value`.
        # EXACT      : recherche complète sans coupure.
        if value <= alpha_orig:
            flag = UPPERBOUND
        elif value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        self._tt[key] = (depth, value, flag, best_move)

        return value

    def _ordered_actions(self, state, tt_move=None):
        """Renvoie les actions légales avec `tt_move` en tête si présent."""
        actions = Game.actions(state)
        if tt_move is None:
            return actions
        # Vérifie que le tt_move est bien une action légale de cet état
        # (il devrait toujours l'être, mais on est prudent en cas de
        # collision Zobrist extrêmement rare).
        if tt_move not in actions:
            return actions
        ordered = [tt_move]
        for a in actions:
            if a != tt_move:
                ordered.append(a)
        return ordered

    # ------------------------------------------------------------------
    # Fonction d'évaluation heuristique (identique à la v2)
    # ------------------------------------------------------------------
    def _evaluate(self, state, player):
        board = state.board
        score = 0

        my_three_threats  = 0
        opp_three_threats = 0

        for line in _LINES:
            cells = [board[r][c] for r, c in line]
            filled = [c for c in cells if c is not None]
            if not filled:
                continue

            n = len(filled)

            # Alignement par couleur
            colors = {c[1] for c in filled}
            if len(colors) == 1:
                color = next(iter(colors))
                if color == player:
                    score += COLOR_WEIGHTS_SELF[n]
                    if n == 3:
                        my_three_threats += 1
                else:
                    score -= COLOR_WEIGHTS_OPP[n]
                    if n == 3:
                        opp_three_threats += 1

            # Alignement par symbole
            symbols = {c[0] for c in filled}
            if len(symbols) == 1:
                own = sum(1 for c in filled if c[1] == player)
                opp = n - own
                if own > opp:
                    score += SYMBOL_WEIGHTS_SELF[n]
                elif opp > own:
                    score -= SYMBOL_WEIGHTS_OPP[n]

        if my_three_threats >= 2:
            score += FORK_SELF_BONUS
        if opp_three_threats >= 2:
            score -= FORK_OPP_PENALTY

        return score