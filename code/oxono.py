from dataclasses import dataclass, field

@dataclass(slots=True)
class State:
    """
    Represents the complete state of an Oxono game at a given point in time.

    Attributes
    ----------
    board : list[list[tuple[str, int] | None]]
        A 6x6 grid representing the board. Each cell is either None (empty) or a
        tuple (symbol, player) where symbol is 'x' or 'o' and player is 0 or 1.
    totem_O : tuple[int, int]
        Current (row, col) position of the O Totem on the board.
    totem_X : tuple[int, int]
        Current (row, col) position of the X Totem on the board.
    current_player : int
        Index of the player whose turn it is (0 or 1).
    pieces_x : list[int]
        Number of X pieces remaining in reserve for each player.
        pieces_x[0] is the count for player 0, pieces_x[1] for player 1.
    pieces_o : list[int]
        Number of O pieces remaining in reserve for each player.
        pieces_o[0] is the count for player 0, pieces_o[1] for player 1.
    last_move : tuple[int, int] | None
        Position (row, col) of the last piece placed on the board, or None if no
        piece has been placed yet. Used to efficiently check win conditions.
    """
    board: list[list[tuple[str, int] | None]] = field(
        default_factory=lambda: [[None for _ in range(6)] for _ in range(6)]
    )
    totem_O: tuple[int, int] = (2, 2)
    totem_X: tuple[int, int] = (3, 3)
    current_player: int = 0
    pieces_x: list[int] = field(
        default_factory=lambda: [8, 8]
    )
    pieces_o: list[int, int] = field(
        default_factory=lambda: [8, 8]
    )
    last_move: tuple[int, int] | None = None

    def copy(self) -> 'State':
        """
        Return a deep copy of this state.

        The board and piece counts are fully copied so that modifying the returned
        state does not affect the original. Totem positions and last_move are
        tuples (immutable) and are therefore safe to share.

        Returns
        -------
        State
            An independent copy of this state.
        """
        new_board = [row[:] for row in self.board]
        return State(new_board, self.totem_O, self.totem_X, self.current_player, self.pieces_x[:], self.pieces_o[:], self.last_move)

class Game:
    """
    Static class implementing the rules of Oxono.

    All methods are static: they take a State as input and either query or
    mutate it. You should never need to instantiate this class.

    An action is represented as a tuple (totem, totem_pos, piece_pos) where:
        - totem     : str            - the Totem moved, either 'O' or 'X'
        - totem_pos : tuple[int,int] - the (row, col) the Totem is moved to
        - piece_pos : tuple[int,int] - the (row, col) where the player's piece is placed
    """

    @staticmethod
    def to_move(state: State) -> int:
        """
        Return the index of the player whose turn it is.

        Parameters
        ----------
        state : State
            The current game state.

        Returns
        -------
        int
            0 for player 0 (pink), 1 for player 1 (black).
        """
        return state.current_player
    
    @staticmethod
    def _totems_actions(state: State, totem: str) -> list[tuple[str, tuple[int, int]]]:
        """
        Return all valid destination squares for a given Totem.

        Implements the three movement tiers in order:
          1. Normal movement - slide any number of squares in one direction over
             empty squares only.
          2. Surrounded jump - if no normal move exists, jump over consecutive
             pieces in each direction and land on the first free square beyond them.
          3. Teleport - if the entire row and column of the Totem are occupied,
             the Totem may be placed on any free square on the board.

        Parameters
        ----------
        state : State
            The current game state.
        totem : str
            The Totem to move, either 'O' or 'X'.

        Returns
        -------
        list[tuple[int, int]]
            List of (row, col) positions the Totem can legally move to.
        """
        totem_actions = []
        board = state.board

        r, c = state.totem_O if totem == 'O' else state.totem_X
        other_totem = state.totem_X if totem == 'O' else state.totem_O

        movable = False
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            for i in range(1, 6):
                nr, nc = r + i*dr, c + i*dc
                if not ((0 <= nr < 6) and (0 <= nc < 6)):  # If outside of board
                    break
                if (board[nr][nc] is not None) or ((nr, nc) == other_totem):  # If not free
                    break
                # We can move the totem to (nr, nc)
                movable = movable or True
                totem_actions.append((nr, nc))
        if not movable:  # Cannot move freely in one direction
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                for i in range(1, 6):
                    nr, nc = r + i*dr, c + i*dc
                    if not ((0 <= nr < 6) and (0 <= nc < 6)):  # If outside of board
                        break
                    if (board[nr][nc] is not None) or ((nr, nc) == other_totem):  # If not free
                        continue
                    # We finaly found a free spot
                    totem_actions.append((nr, nc))
                    movable = True
                    break
            if not movable:
                for nr in range(0, 6):
                    for nc in range(0, 6):
                        if (board[nr][nc] is not None) or (nr, nc) == (r, c) or (nr, nc) == other_totem:  # If not free
                            continue
                        movable = True
                        totem_actions.append((nr, nc))
        return totem_actions

    @staticmethod
    def actions(state: State) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
        """
        Return all legal actions available to the current player.

        For each Totem the player has pieces for, every combination of a valid
        Totem destination and a valid piece placement adjacent to that destination
        is enumerated. If no adjacent square is free after moving a Totem to a
        given position (surrounded destination), the piece may be placed on any
        free square on the board instead.

        Parameters
        ----------
        state : State
            The current game state.

        Returns
        -------
        list[tuple[str, tuple[int, int], tuple[int, int]]]
            List of actions, each of the form (totem, totem_pos, piece_pos).
        """
        all_actions = []

        board = state.board

        # Totem O
        if (state.pieces_o[state.current_player] > 0):
            other_totem = state.totem_X
            for r, c in Game._totems_actions(state, 'O'):
                placable = False
                for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if not ((0 <= nr < 6) and (0 <= nc < 6)):  # If outside
                        continue
                    if (board[nr][nc] is not None) or (nr, nc) == other_totem:  # If not free
                        continue
                    placable = placable or True
                    all_actions.append(('O', (r, c), (nr, nc)))
                if not placable:
                    for nr in range(0, 6):
                        for nc in range(0, 6):
                            if (board[nr][nc] is not None) or (nr, nc) == (r, c) or (nr, nc) == other_totem:  # If not free
                                continue
                            all_actions.append(('O', (r, c), (nr, nc)))

        # Totem X
        if (state.pieces_x[state.current_player] > 0):
            other_totem = state.totem_O
            for r, c in Game._totems_actions(state, 'X'):
                placable = False
                for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if not ((0 <= nr < 6) and (0 <= nc < 6)):  # If outside
                        continue
                    if (board[nr][nc] is not None) or (nr, nc) == other_totem:  # If not free
                        continue
                    placable = placable or True
                    all_actions.append(('X', (r, c), (nr, nc)))
                if not placable:
                    for nr in range(0, 6):
                        for nc in range(0, 6):
                            if (board[nr][nc] is not None) or (nr, nc) == (r, c) or (nr, nc) == other_totem:  # If not free
                                continue
                            all_actions.append(('X', (r, c), (nr, nc)))
        
        return all_actions

    @staticmethod
    def apply(state: State, action: tuple[str, tuple[int, int], tuple[int, int]]):
        """
        Apply an action to the state, mutating it in place.

        Updates the Totem position, places the player's piece on the board,
        decrements the appropriate piece counter, records last_move, and
        advances current_player, unless the opponent has no pieces left, in
        which case the current player keeps their turn.

        Parameters
        ----------
        state : State
            The game state to mutate.
        action : tuple[str, tuple[int, int], tuple[int, int]]
            The action to apply, of the form (totem, totem_pos, piece_pos).
        """
        totem, totem_pos, piece_pos = action
        if totem == 'O':
            state.totem_O = totem_pos
        else:
            state.totem_X = totem_pos
        state.board[piece_pos[0]][piece_pos[1]] = ('o' if totem == 'O' else 'x', state.current_player)
        
        if totem == 'O':
            state.pieces_o[state.current_player] -= 1
        else: 
            state.pieces_x[state.current_player] -= 1
        
        state.last_move = piece_pos

        if state.pieces_o[1 - state.current_player] > 0 or state.pieces_x[1 - state.current_player] > 0:
            state.current_player = 1 - state.current_player
    
    @staticmethod
    def _last_piece_won(state: State) -> bool:
        """
        Check whether the last piece placed on the board created a winning alignment.

        Scans horizontally and vertically through the last placed piece, counting
        consecutive pieces sharing the same symbol or the same color. A count of
        4 or more in either direction constitutes a win.

        Parameters
        ----------
        state : State
            The current game state. Must have a valid last_move.

        Returns
        -------
        bool
            True if the last move created a winning alignment, False otherwise.
        """
        last_move = state.last_move
        if last_move is None:
            return False
        
        board = state.board
        r, c = last_move
        cell = board[r][c]
        symbol, color = cell

        for dr, dc in [(1, 0), (0, 1)]:

            count_symbol = 1
            count_color = 1

            valid_symbol = True
            valid_color = True
            for i in range(1, 4):
                nr, nc = r + i*dr, c + i*dc
                if 0 <= nr < 6 and 0 <= nc < 6 and board[nr][nc]:
                    sym, col = board[nr][nc]
                    if sym == symbol and valid_symbol:
                        count_symbol += 1
                    else:
                        valid_symbol = False
                    
                    if col == color and valid_color:
                        count_color += 1
                    else:
                        valid_color = False
                    
                    if not valid_symbol and not valid_color:
                        break
                else:
                    break
            
            valid_symbol = True
            valid_color = True
            for i in range(1, 4):
                nr, nc = r + -i*dr, c + -i*dc
                if 0 <= nr < 6 and 0 <= nc < 6 and board[nr][nc]:
                    sym, col = board[nr][nc]
                    if sym == symbol and valid_symbol:
                        count_symbol += 1
                    else:
                        valid_symbol = False
                    
                    if col == color and valid_color:
                        count_color += 1
                    else:
                        valid_color = False
                    
                    if not valid_symbol and not valid_color:
                        break
                else:
                    break

            if count_symbol >= 4 or count_color >= 4:
                return True

        return False

    @staticmethod
    def is_terminal(state: State) -> bool:
        """
        Return True if the game is over.

        The game ends in two situations:
          - A player has formed a winning alignment with their last piece.
          - All 32 pieces have been placed with no winner (draw).

        Parameters
        ----------
        state : State
            The current game state.

        Returns
        -------
        bool
            True if the game has ended, False if play should continue.
        """
        if Game._last_piece_won(state):
            return True
        
        if state.pieces_o == [0, 0] and state.pieces_x == [0, 0]:  # No one can move anymore -> Draw
            return True
            
        return False

    @staticmethod
    def utility(state: State, player: int) -> int:
        """
        Return the utility of a terminal state for the given player.

        Should only be called once is_terminal() returns True.

        Parameters
        ----------
        state : State
            A terminal game state.
        player : int
            The player whose utility is being queried (0 or 1).

        Returns
        -------
        int
            1  if the player won,
            -1 if the player lost,
            0  if the game ended in a draw.
        """
        if Game._last_piece_won(state):
            if player == (1 - state.current_player):
                return 1
            else:
                return -1
        return 0