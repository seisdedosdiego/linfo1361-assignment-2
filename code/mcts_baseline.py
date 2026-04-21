"""
MCTS baseline pour le jeu Oxono.

Agent Monte-Carlo Tree Search basique, utilisé uniquement comme point de
comparaison pour le rapport (cf. consignes : "A comparison between your
final agent and several reference agents, including at least a random
agent, a baseline Alpha-Beta agent, and a baseline MCTS agent").

Volontairement simple : UCB1 + rollouts aléatoires, sans RAVE, sans
progressive bias, sans heuristique dans les rollouts. L'idée est de
mesurer l'apport de notre agent Alpha-Beta par rapport à une approche
Monte-Carlo "scolaire".

Ce fichier n'est PAS soumis sur Inginious — il sert uniquement aux
expérimentations locales.
"""

from agent import Agent
from oxono import Game, State
import random
import math
import time


# Constante d'exploration d'UCB1 (valeur classique sqrt(2)).
_C_UCB = math.sqrt(2)

# Marge de sécurité : on s'arrête un peu avant la deadline pour éviter de
# dépasser le budget à cause du temps de finalisation.
_SAFETY_MARGIN = 0.3


class _Node:
    """Nœud de l'arbre MCTS.

    Chaque nœud stocke un état de jeu et les statistiques accumulées des
    simulations qui sont passées par là.

    Convention : les statistiques (visits, wins) sont exprimées du point
    de vue du joueur qui a joué l'action menant à ce nœud. Autrement dit,
    plus ``wins`` est élevé, plus l'action qui mène à ce nœud est bonne
    pour le parent. C'est la convention classique (Kocsis & Szepesvári).

    Attributes
    ----------
    state : State
        État du jeu à ce nœud.
    parent : _Node | None
        Nœud parent, None pour la racine.
    action : tuple | None
        Action qui a mené du parent à ce nœud, None pour la racine.
    children : list[_Node]
        Enfants déjà développés.
    untried : list[tuple]
        Actions pas encore explorées depuis ce nœud (pour expansion
        progressive).
    visits : int
        Nombre total de simulations passées par ce nœud.
    wins : float
        Somme des utilités remontées (peut être négative si on stocke
        des scores dans [-1, 1]).
    """
    __slots__ = ("state", "parent", "action", "children", "untried", "visits", "wins")

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        # On mélange pour casser les biais liés à l'ordre d'énumération
        # des actions (important car on expanse en LIFO via pop()).
        self.untried = Game.actions(state)
        random.shuffle(self.untried)
        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def best_uct_child(self, c):
        """Retourne l'enfant avec la meilleure valeur UCB1.

        UCB1(child) = wins/visits + c * sqrt(ln(parent.visits) / child.visits)

        Le premier terme est l'exploitation (winrate moyen), le second
        est l'exploration (favorise les enfants peu visités).
        """
        log_n = math.log(self.visits)
        best = None
        best_score = -float("inf")
        for child in self.children:
            exploit = child.wins / child.visits
            explore = c * math.sqrt(log_n / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best


class MCTSBaseline(Agent):
    """Agent MCTS basique (UCB1 + rollouts aléatoires).

    Cet agent sert de baseline MCTS pour les expérimentations du rapport.
    Il n'est volontairement pas optimisé : pas d'heuristique dans les
    rollouts, pas de biais sur le tree policy, pas de réutilisation
    d'arbre entre coups.

    Budget de temps : le temps restant est divisé par le nombre de coups
    que l'agent devra encore jouer (estimé par ses pièces en réserve),
    avec une petite marge de sécurité.
    """

    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        # Estimation du nombre de coups qu'il nous reste à jouer :
        # chaque coup consomme exactement une de nos pièces.
        my_pieces = state.pieces_o[self.player] + state.pieces_x[self.player]
        budget = max(0.1, remaining_time / max(1, my_pieces) - _SAFETY_MARGIN)
        deadline = time.time() + budget

        root = _Node(state.copy())

        # Cas trivial : aucun coup légal (ne devrait pas arriver mais
        # protégeons-nous) ou un seul coup possible.
        if not root.untried and not root.children:
            # Filet de sécurité : on doit retourner UNE action. Si aucune
            # action n'est légale à la racine, on laisse le manager gérer
            # (il considérera ça comme une défaite). On retourne None ici
            # ne fonctionnerait pas, donc on appelle actions() directement.
            actions = Game.actions(state)
            return actions[0] if actions else None
        if len(root.untried) == 1 and not root.children:
            return root.untried[0]

        # Boucle principale : tant qu'on a du budget, on fait des
        # itérations MCTS complètes (selection -> expansion -> rollout
        # -> backpropagation).
        while time.time() < deadline:
            node = self._select(root)
            if not Game.is_terminal(node.state):
                node = self._expand(node)
            reward = self._rollout(node.state)
            self._backpropagate(node, reward)

        # Sélection finale : on choisit l'action la plus visitée (robust
        # child). C'est plus stable que de choisir sur le winrate car
        # un winrate très élevé sur 2 visites est souvent du bruit.
        if not root.children:
            return random.choice(Game.actions(state))
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    # ---------- Phases de MCTS ----------

    def _select(self, node):
        """Phase de selection : descend l'arbre via UCB1 jusqu'à tomber
        sur un nœud non entièrement développé ou un nœud terminal."""
        while node.is_fully_expanded() and node.children:
            node = node.best_uct_child(_C_UCB)
        return node

    def _expand(self, node):
        """Phase d'expansion : développe un enfant non exploré du nœud,
        s'il en reste. Sinon on retourne le nœud tel quel."""
        if not node.untried:
            return node
        action = node.untried.pop()
        new_state = node.state.copy()
        Game.apply(new_state, action)
        child = _Node(new_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _rollout(self, state):
        """Phase de simulation : joue la partie au hasard jusqu'à la fin.

        Oxono a au plus 32 pièces à placer donc un rollout termine en au
        plus 32 coups : pas besoin de cap explicite.

        Retourne +1 si self.player gagne la partie simulée, -1 s'il perd,
        0 en cas de nulle. C'est la convention standard qu'on rebranche
        côté parent dans le backpropagate.
        """
        sim = state.copy()
        while not Game.is_terminal(sim):
            actions = Game.actions(sim)
            if not actions:
                break
            Game.apply(sim, random.choice(actions))
        return Game.utility(sim, self.player)

    def _backpropagate(self, node, reward_for_self):
        """Remonte la récompense dans l'arbre.

        À chaque nœud du chemin, on met à jour visits et wins. Les stats
        étant stockées du point de vue du joueur qui a joué l'action
        menant au nœud, on inverse le signe quand ce joueur n'est pas
        self.player.
        """
        cur = node
        while cur is not None:
            cur.visits += 1
            if cur.parent is not None:
                # Le joueur qui a joué pour arriver à cur est le
                # current_player au niveau du parent (avant apply).
                mover = cur.parent.state.current_player
                if mover == self.player:
                    cur.wins += reward_for_self
                else:
                    cur.wins -= reward_for_self
            cur = cur.parent