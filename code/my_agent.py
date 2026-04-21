"""
LINFO1361 - Intelligence Artificielle, Assignment 2
Auteurs : Seisdedos Stoz Diego (4659-23-00), Muylkens Justin (8004-22-00)
"""

import time
from agent import Agent
from oxono import Game

# ----------------------------------------------------------------------
# Poids de la fonction d'évaluation heuristique
# ----------------------------------------------------------------------
# Les poids sont asymétriques : un alignement à 3 pour NOUS est quasi
# une victoire (on joue tout de suite), alors qu'un alignement à 3 pour
# l'adversaire est moins urgent car on joue avant lui et on peut bloquer.
#
# L'indice du tuple = nombre de pièces déjà alignées (0 à 3).
COLOR_WEIGHTS_SELF  = (0, 1, 10, 500)   # alignement exclusif par couleur (nous)
COLOR_WEIGHTS_OPP   = (0, 1, 10, 100)   # alignement exclusif par couleur (adversaire)
SYMBOL_WEIGHTS_SELF = (0, 1,  5, 200)   # alignement par symbole (nous)
SYMBOL_WEIGHTS_OPP  = (0, 1,  5,  60)   # alignement par symbole (adversaire)

# Bonus/pénalité pour une fourchette (au moins 2 menaces à 3 simultanées).
# Une fourchette est quasi imparable : l'adversaire ne peut bloquer qu'une
# menace par tour et nous en avons plusieurs.
FORK_SELF_BONUS  = 5000
FORK_OPP_PENALTY = 2000

# Valeur d'un état terminal détecté pendant la recherche.
WIN_SCORE = 100_000

# ----------------------------------------------------------------------
# Pré-calcul des 36 lignes de 4 cases potentiellement gagnantes
# ----------------------------------------------------------------------
# Plateau 6x6 -> 6 lignes * 3 décalages horizontaux + 6 colonnes *
# 3 décalages verticaux = 36 lignes. On les calcule une seule fois
# au chargement du module pour éviter de les recalculer à chaque
# évaluation de position.
_LINES = []
for _r in range(6):
    for _c in range(3):
        _LINES.append(tuple((_r, _c + i) for i in range(4)))   # lignes horizontales
for _c in range(6):
    for _r in range(3):
        _LINES.append(tuple((_r + i, _c) for i in range(4)))   # lignes verticales
_LINES = tuple(_LINES)

class MyAgent(Agent):
    """
    Agent Oxono basé sur Negamax avec élagage Alpha-Beta et approfondissement itératif.
    """

    def __init__(self, player):
        """
        Initialise l'agent.

        Parameters
        ----------
        player : int
            Indice du joueur (0 = rose, 1 = noir).
        """
        super().__init__(player)
        # Marge de sécurité (en secondes) conservée sur chaque coup pour
        # éviter tout dépassement du budget temps et donc un timeout.
        self._safety_margin = 0.5

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------
    def act(self, state, remaining_time):
        """
        Calcule et renvoie le prochain coup à jouer.

        Parameters
        ----------
        state : State
            État courant du jeu.
        remaining_time : float
            Temps (en secondes) restant sur notre horloge pour toute
            la partie (pas seulement ce coup).

        Returns
        -------
        tuple
            Une action légale de la forme (totem, totem_pos, piece_pos).
        """
        actions = Game.actions(state)
        if not actions:
            # Cas défensif : un état non terminal possède toujours au
            # moins une action possible, mais on ne veut pas planter.
            return None

        # --- Calcul du budget de temps pour CE coup ---------------
        # On estime qu'il nous reste à peu près autant de coups à jouer
        # que de pièces dans notre réserve. Le budget par coup diminue
        # donc naturellement vers la fin de la partie.
        my_pieces = state.pieces_o[self.player] + state.pieces_x[self.player]
        moves_left = max(my_pieces, 1)
        time_budget = max(
            remaining_time / (moves_left + 1) - self._safety_margin,
            0.05,  # plancher : jamais moins de 50 ms par coup
        )
        deadline = time.perf_counter() + time_budget

        # --- Approfondissement itératif ---------------------------
        # On lance une recherche complète à profondeur 1, puis 2, puis 3,
        # etc. On conserve toujours le meilleur coup trouvé à la dernière
        # profondeur COMPLETEMENT explorée. Si on timeout au milieu d'une
        # profondeur, on garde le résultat de la précédente.
        best_action = actions[0] # coup par défaut si on ne finit même pas la profondeur 1
        depth = 1
        while depth <= 20: # borne de sécurité
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
    # Recherche adverse
    # ------------------------------------------------------------------
    def _search_root(self, state, depth, deadline):
        """
        Recherche à la racine : renvoie la meilleure action trouvée
        à la profondeur donnée.

        Lève TimeoutError si le temps imparti est dépassé.
        """
        best_action = None
        best_value = -float('inf')
        alpha, beta = -float('inf'), float('inf')

        for action in Game.actions(state):
            if time.perf_counter() > deadline:
                raise TimeoutError
            # Game.apply modifie l'état en place : il faut donc copier
            # avant d'appliquer pour ne pas polluer l'état parent.
            child = state.copy()
            Game.apply(child, action)
            # Le fils renvoie son score de SON point de vue -> on inverse.
            value = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if value > best_value:
                best_value = value
                best_action = action
            if value > alpha:
                alpha = value
        return best_action

    def _negamax(self, state, depth, alpha, beta, deadline):
        """
        Recherche Negamax avec élagage Alpha-Beta.

        La valeur renvoyée est exprimée du point de vue du joueur
        qui doit jouer dans `state` (convention Negamax).
        """
        # Vérification du temps à chaque noeud pour pouvoir sortir
        # proprement en levant une exception.
        if time.perf_counter() > deadline:
            raise TimeoutError

        # Etat terminal : on renvoie directement la valeur "réelle".
        # Game.utility(state, player) renvoie +1/-1/0 du point de vue
        # du joueur spécifié.
        if Game.is_terminal(state):
            return Game.utility(state, state.current_player) * WIN_SCORE

        # Profondeur atteinte : on évalue heuristiquement.
        if depth == 0:
            return self._evaluate(state, state.current_player)

        value = -float('inf')
        for action in Game.actions(state):
            child = state.copy()
            Game.apply(child, action)
            # Appel récursif : alpha et beta sont inversés, car le point
            # de vue change (c'est à l'adversaire de jouer ensuite).
            v = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            # Coupure beta : l'adversaire ne laissera jamais arriver
            # jusqu'ici, inutile d'explorer les autres coups.
            if alpha >= beta:
                break
        return value

    # ------------------------------------------------------------------
    # Fonction d'évaluation heuristique
    # ------------------------------------------------------------------
    def _evaluate(self, state, player):
        """
        Renvoie un score heuristique du point de vue de `player`.

        Positif = bon pour `player`, négatif = bon pour l'adversaire.

        Stratégie :
        - Pour chaque ligne de 4 cases, score les alignements partiels
            (couleur et symbole) en tenant compte de qui doit jouer.
        - Bonus spécial pour les fourchettes (2+ menaces à 3 simultanées).
        """
        board = state.board
        score = 0

        # Compteurs de menaces à 3 pour détecter les fourchettes
        my_three_threats  = 0
        opp_three_threats = 0

        for line in _LINES:
            cells = [board[r][c] for r, c in line]
            filled = [c for c in cells if c is not None]
            if not filled:
                continue

            n = len(filled)

            # ----- Potentiel d'alignement par couleur -----
            # Ligne vivante pour une couleur seulement si toutes les
            # pièces présentes sont de la MEME couleur.
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

            # ----- Potentiel d'alignement par symbole -----
            # Menace partagée : biais vers le joueur qui a investi le plus
            # de pièces dans cette ligne.
            symbols = {c[0] for c in filled}
            if len(symbols) == 1:
                own = sum(1 for c in filled if c[1] == player)
                opp = n - own
                if own > opp:
                    score += SYMBOL_WEIGHTS_SELF[n]
                elif opp > own:
                    score -= SYMBOL_WEIGHTS_OPP[n]

        # ----- Bonus fourchette -----
        # Plusieurs menaces simultanées = victoire quasi assurée.
        if my_three_threats >= 2:
            score += FORK_SELF_BONUS
        if opp_three_threats >= 2:
            score -= FORK_OPP_PENALTY

        return score