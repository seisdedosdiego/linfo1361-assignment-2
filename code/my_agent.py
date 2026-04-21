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
# L'indice du tuple = nombre de pièces déjà alignées sur une ligne de 4
# cases (de 0 à 3). Une ligne avec 4 pièces de même couleur/symbole
# est une victoire et est gérée par Game.is_terminal() + Game.utility(),
# elle n'a donc pas besoin de poids ici.
#
# La progression est exponentielle car une ligne à 3 pièces est une
# menace immédiate (un coup pour gagner) : elle doit dominer fortement
# une ligne à 2 pièces.
COLOR_WEIGHTS = (0, 1, 10, 100)      # alignement exclusif d'une même couleur
SYMBOL_WEIGHTS = (0, 1, 5, 50)      # alignement d'un même symbole (partagé, donc moins décisif)

# Valeur renvoyée lorsqu'un état terminal (victoire/défaite) est
# détecté pendant la recherche. Elle doit écraser n'importe quelle
# valeur heuristique pour que l'agent préfère toujours une vraie
# victoire à une position "qui a l'air bonne".
WIN_SCORE = 100000

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

        Score positif = bon pour `player`, négatif = bon pour l'adversaire.

        Principe : on parcourt les 36 lignes de 4 cases et on regarde,
        pour chacune, si elle peut encore devenir une ligne gagnante
        (par couleur ou par symbole). Plus il y a de pièces alignées,
        plus le score est élevé.
        """
        board = state.board
        score = 0

        for line in _LINES:
            cells = [board[r][c] for r, c in line]
            filled = [c for c in cells if c is not None]
            if not filled:
                continue   # ligne vide : pas d'information

            # ----- Potentiel d'alignement par couleur -----
            # Une ligne ne peut encore devenir une victoire par couleur
            # QUE si toutes les pièces présentes sont de la MEME couleur.
            # Dès qu'il y a au moins une pièce de chaque couleur, la
            # ligne est "morte" pour la couleur.
            colors = {c[1] for c in filled}
            if len(colors) == 1:
                color = next(iter(colors))
                w = COLOR_WEIGHTS[len(filled)]
                score += w if color == player else -w

            # ----- Potentiel d'alignement par symbole -----
            # Une ligne ne peut encore devenir une victoire par symbole
            # QUE si toutes les pièces partagent le MEME symbole.
            # Cette menace est "partagée" : n'importe quel joueur peut
            # la compléter en plaçant une pièce du bon symbole.
            # On biaise le score vers le joueur qui a déjà investi le
            # plus de pièces dans cette ligne.
            symbols = {c[0] for c in filled}
            if len(symbols) == 1:
                own = sum(1 for c in filled if c[1] == player)
                opp = len(filled) - own
                w = SYMBOL_WEIGHTS[len(filled)]
                if own > opp:
                    score += w
                elif opp > own:
                    score -= w

        return score