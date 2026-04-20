from oxono import Game, State
from pathlib import Path

import pygame
import sys
import math
import argparse

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

class Replayer:

    def __init__(self, path_to_file):

        time_limit, steps = self._read(path_to_file)

        self.steps = steps

        self.index = 0
        self.frames = [(0, State(), time_limit, time_limit, None)]

        self.dim = (6, 6)

        pygame.init()
        
        self.screen = pygame.display.set_mode((70*self.dim[1] + 100, 70*self.dim[0] + 150))
        pygame.display.set_caption("Replayer Oxono")

        self.number_font = pygame.font.Font(None, 36)
        self.win_font = pygame.font.Font(None, 52)

        self.pieces = {
            ('x', 0) : self._create_piece_surface('x', 0),
            ('x', 1) : self._create_piece_surface('x', 1),
            ('o', 0) : self._create_piece_surface('o', 0),
            ('o', 1) : self._create_piece_surface('o', 1)
        }

        self.totems = {
            'X' : self._create_totem_surface('x'),
            'O' : self._create_totem_surface('o')
        }

        self.clock = pygame.time.Clock()
        self.running = True
    
    def _read(self, path_to_file):
        path = Path(path_to_file)
        with path.open('r', encoding="utf-8") as f:
            lines = f.readlines()
            time_limit = float(lines[0].strip())
            steps = []
            for line in lines[1:]:
                line = line.strip()
                if "exception" in line:
                    steps.append("exception")
                elif "invalid" in line:
                    steps.append("invalid")
                else:
                    action_str = line[:21]
                    remaining_time_str = line[22:]
                    action = eval(action_str)
                    remaining_time = float(remaining_time_str)
                    steps.append((action, remaining_time))
            return time_limit, steps

    def _create_piece_surface(self, symbol, player):
        surface = pygame.Surface((70, 70), pygame.SRCALPHA)
        center = (35, 35)
        pygame.draw.circle(surface, (51, 63, 73) if player == 1 else (254, 44, 135), center, 28)
        
        if symbol == 'o':
            pygame.draw.circle(surface, (255, 255, 255), center, 15, width=6)
        else:
            w = 3  # half-width of the arms
            s = 12 # spread from center
            
            # Coordinates relative to center (35, 35)
            points = [
                (35-s, 35-s+w), (35-s+w, 35-s), (35, 35-w), (35+s-w, 35-s),
                (35+s, 35-s+w), (35+w, 35), (35+s, 35+s-w), (35+s-w, 35+s),
                (35, 35+w), (35-s+w, 35+s), (35-s, 35+s-w), (35-w, 35)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), points)
        
        return surface

    def _create_totem_surface(self, symbol):
        surface = pygame.Surface((70, 70), pygame.SRCALPHA)

        rect_area = pygame.Rect(7, 7, 56, 56)

        pygame.draw.rect(surface, (1, 195, 255), rect_area, border_radius=12)
        
        if symbol == 'o':
            pygame.draw.circle(surface, (255, 255, 255), (35, 35), 15, width=6)
        else:
            w = 3  # half-width of the arms
            s = 12 # spread from center
            
            # Coordinates relative to center (35, 35)
            points = [
                (35-s, 35-s+w), (35-s+w, 35-s), (35, 35-w), (35+s-w, 35-s),
                (35+s, 35-s+w), (35+w, 35), (35+s, 35+s-w), (35+s-w, 35+s),
                (35, 35+w), (35-s+w, 35+s), (35-s, 35+s-w), (35-w, 35)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), points)
        
        return surface
    
    def forward(self):
        if self.index >= len(self.steps):  # Cannot go too far in the future
            return
        
        if self.index < len(self.frames)-1:  # We can simply increase the index
            self.index += 1
        else:  # We have to create new state based on the steps
            current_player, state, time_0, time_1, reason = self.frames[self.index]

            next_player = 1 - current_player
            next_state = state.copy()
            next_time_0 = time_0
            next_time_1 = time_1
            next_reason = reason

            step = self.steps[self.index]

            if step == "exception" or step == "invalid":
                next_reason = step
            else:
                action, remaining_time = step
                Game.apply(next_state, action)
                if current_player == 0:
                    next_time_0 = remaining_time
                else:
                    next_time_1 = remaining_time
            
            self.frames.append((next_player, next_state, next_time_0, next_time_1, next_reason))
            self.index += 1

    def backward(self):
        if self.index > 0:
            self.index -= 1
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_LEFT:
                    self.backward()
                elif event.key == pygame.K_RIGHT:
                    self.forward()
    
    def _draw_board(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                pygame.draw.rect(self.screen, 'Black', (70*j + 50, 70*i + 50, 70, 70), 1)
    
    def _draw_piece(self, position, value):
        self.screen.blit(self.pieces[value], (70*position[1] + 50, 70*position[0] + 50))
    
    def _draw_totem(self, position, value):
        self.screen.blit(self.totems[value], (70*position[1] + 50, 70*position[0] + 50))

    def _draw_pieces(self):
        state = self.frames[self.index][1]
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if state.board[i][j] is not None:
                    self._draw_piece((i, j), state.board[i][j])
                else:
                    if (i, j) == state.totem_O:
                        self._draw_totem((i, j), 'O')
                    elif (i, j) == state.totem_X:
                        self._draw_totem((i, j), 'X')
    
    def draw(self):
        self.screen.fill('White')

        self._draw_board()
        self._draw_pieces()

        frame = self.frames[self.index]

        if frame[4] is not None:
            text = self.win_font.render(f"{'Black' if frame[0] == 1 else 'Pink'} wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            reason = "Invalid action" if frame[4] == "invalid" else "Raised Exception"

            text = self.number_font.render(reason, True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif Game.is_terminal(frame[1]):
            u0 = Game.utility(frame[1], 0)
            u1 = Game.utility(frame[1], 1)
            text = None
            if u0 == u1:
                text = self.win_font.render("Draw!", True, 'Black')
            else:
                text = self.win_font.render(f"{'Pink' if u0 > u1 else 'Black'} wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 100))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif frame[2] <= 0:
            text = self.win_font.render(f"Black wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            reason = "Timeout"

            text = self.number_font.render(reason, True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif frame[3] <= 0:
            text = self.win_font.render(f"Pink wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            reason = "Timeout"

            text = self.number_font.render(reason, True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        else:
            text = self.number_font.render(f"{'Black' if frame[0] == 1 else 'Pink'} to play", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 75))
            self.screen.blit(text, text_rect)
            text = self.number_font.render(f"Pink: {truncate(frame[2], 2)} s", True, 'Black')
            text_rect = text.get_rect(topleft=(50, 70*self.dim[0] + 110))
            self.screen.blit(text, text_rect)
            text = self.number_font.render(f"Black: {truncate(frame[3], 2)} s", True, 'Black')
            text_rect = text.get_rect(topleft=(320, 70*self.dim[0] + 110))
            self.screen.blit(text, text_rect)
        
        text = self.number_font.render(f"Turn {self.index+1}/{len(self.steps)+1}", True, 'Black')
        text_rect = text.get_rect(topleft=(50, 12))
        self.screen.blit(text, text_rect)

        pygame.display.flip()
    
    def play(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    """
    Replay a saved Oxono game from the command line in an interactive window.

    Usage
    -----
        python replayer.py FILE

    Arguments
    ---------
        FILE    Path to a game log file produced by manager.py.

    Controls
    --------
        Right Arrow:   Advance to the next turn.
        Left Arrow:    Go back to the previous turn.
        Escape:        Close the window.

    Example
    -------
        python replayer.py logs/log_0.txt
    """
    parser = argparse.ArgumentParser(description="Replay Oxono game")
    parser.add_argument("file", type=str, help="Path to the saved log of the game")
    args = parser.parse_args()

    r = Replayer(args.file)
    r.play()