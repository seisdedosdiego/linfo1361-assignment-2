class Agent:
    """
    Abstract base class that every agent must extend.

    To implement your own agent, create a new class that inherits from Agent
    and overrides the act() method.

    Attributes
    ----------
    player : int
        The index assigned to this agent for the current game (0 or 1).
    """


    def __init__(self, player):
        """
        Initialise the agent.

        This constructor is called once before the game starts. You may override
        it to set up any data structures or models your agent needs, as long as
        you call super().__init__(player) first and your constructor expect only
        the player parameter.

        Parameters
        ----------
        player : int
            The index assigned to this agent for the current game (0 or 1).
        """
        self.player = player
    
    def act(self, state, remaining_time):
        """
        Choose and return an action for the current turn.

        This method is called once per turn and must return a legal action
        before remaining_time runs out. If it raises an exception, returns an
        illegal action, or exceeds the time limit, the game is lost.

        Parameters
        ----------
        state : State
            The current game state.
        remaining_time : float
            Total seconds remaining on your clock for the rest of the game.

        Returns
        -------
        tuple
            A legal action. Must be present in Game.actions(state).
        """
        raise NotImplementedError