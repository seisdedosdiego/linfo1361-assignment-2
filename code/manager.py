from pathlib import Path
import importlib.util
import inspect
import time
import multiprocessing
import argparse

from oxono import Game, State
from agent import Agent

def find_agent_class(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Agent) and obj is not Agent:
            return obj
    return None

def run_agent_process(filename, player, conn):
    agent_class = find_agent_class(filename)
    agent = agent_class(player)

    conn.send("ready")

    while True:
        try:
            message = conn.recv()
            if message is None:  # shutdown signal
                break
            state, remaining_time = message
            try:
                action = agent.act(state, remaining_time)
                conn.send(("ok", action))
            except Exception as e:
                conn.send(("error", e))
        except EOFError:
            break

class AgentProcess:

    def __init__(self, agent_file, player):
        parent_conn, child_conn = multiprocessing.Pipe()
        self.conn = parent_conn
        self.player = player
        self.process = multiprocessing.Process(
            target=run_agent_process,
            args=(agent_file, player, child_conn),
            daemon=True
        )
        self.process.start()
        child_conn.close()

        if not self.conn.poll(timeout=30):
            self.kill()
            raise RuntimeError(f"Agent {player} failed to initialize in time")
        self.conn.recv()
    
    def get_action(self, state, remaining_time):
        self.conn.send((state, remaining_time))

        start = time.perf_counter()
        has_response = self.conn.poll(remaining_time)
        elapsed = time.perf_counter() - start

        if not has_response:
            self.kill()
            raise TimeoutError(f"Agent {self.player} exceeded time limit")

        status, value = self.conn.recv()
        if status == "error":
            raise RuntimeError(f"Agent {self.player} raised: {value}")

        return value, elapsed
    
    def kill(self):
        self.process.kill()
        self.process.join()
    
    def shutdown(self):
        try:
            self.conn.send(None)
        except Exception:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.kill()

class Manager:
    """
    Orchestrates one or more Oxono games between two agents.

    Each agent runs in an isolated subprocess, so crashes, infinite loops, and
    third-party libraries cannot interfere with the manager or with each other.
    Agent files are validated at construction time, so errors are caught before
    any game is launched.

    Parameters
    ----------
    agent_files : list[str]
        A list of exactly two filenames (e.g. ["my_agent.py", "random_agent.py"]).
        Each file must be located in the current working directory and contain a
        class that extends Agent.
    time_limit : int
        Total number of seconds each player has for the entire game (default: 300).
    """

    def __init__(self, agent_files, time_limit=300):
        self.agent_files = agent_files
        self.time_limit = time_limit

        for agent_file in self.agent_files:
            if find_agent_class(agent_file) is None:
                raise ValueError(f"No Agent subclass found in {agent_file}.")
    
    def play(self, path_to_file=None):
        """
        Run a single game and return the result.

        Each player's subprocess is started fresh, plays the full game, then is
        shut down. If a player times out, raises an exception, or returns an
        illegal action, they immediately lose.

        Parameters
        ----------
        path_to_file : str, optional
            Path to a log file where the game will be recorded. The file is
            created along with any missing parent directories. If None, no log
            is written.

        Returns
        -------
        tuple[int, int]
            A (utility_p0, utility_p1) pair.
        """

        path = None
        if path_to_file:
            path = Path(path_to_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                f.write(f"{self.time_limit}\n")

        agent_0 = AgentProcess(self.agent_files[0], 0)
        agent_1 = AgentProcess(self.agent_files[1], 1)
        remaining_times = [self.time_limit, self.time_limit]

        state = State()
        result = None
        try:
            while not Game.is_terminal(state) and all(t > 0 for t in remaining_times):
                current = Game.to_move(state)
                agent = agent_0 if current == 0 else agent_1

                try:
                    action, elapsed = agent.get_action(state.copy(), remaining_times[current])
                    remaining_times[current] -= elapsed
                except TimeoutError:
                    if path:
                        with path.open("a", encoding="utf-8") as f:
                            f.write("timeout\n")
                    result = (-1, 1) if current == 0 else (1, -1)
                    break
                except RuntimeError as e:
                    print(e)
                    if path:
                        with path.open("a", encoding="utf-8") as f:
                            f.write("exception\n")
                    result = (-1, 1) if current == 0 else (1, -1)
                    break
                
                if action not in Game.actions(state):
                    if path:
                        with path.open("a", encoding="utf-8") as f:
                            f.write("invalid\n")
                    result =  (-1, 1) if Game.to_move(state) == 0 else (1, -1)
                    break
                
                Game.apply(state, action)
                
                if path:
                    with path.open("a", encoding="utf-8") as f:
                        f.write(f"{action}, {remaining_times[current]}\n")
            
            if result is None:
                if remaining_times[0] <= 0:
                    result = (-1, 1)
                elif remaining_times[1] <= 0:
                    result = (1, -1)
                else:
                    result = Game.utility(state, 0), Game.utility(state, 1)
        
        finally:
            agent_0.shutdown()
            agent_1.shutdown()
        
        return result

if __name__ == "__main__":

    """
    Run one or more Oxono games from the command line and print aggregated results.

    Usage
    -----
        python manager.py [-n N] [-p0 FILE] [-p1 FILE] [-l DIR] [-t SECONDS]

    Arguments
    ---------
        -n  N           Number of games to play. Temporary standings are printed
                        after each game except the last. Default: 1.
        -p0 FILE        Python file for player 0. Default: random_agent.py.
        -p1 FILE        Python file for player 1. Default: random_agent.py.
        -l  DIR         Directory where game logs are saved as log_0.txt,
                        log_1.txt, etc. If omitted, no logs are written.
        -t  SECONDS     Time limit per player per game in seconds. Default: 300.

    Examples
    --------
        # One game, both players are the default random agent
        python manager.py

        # 100 games between two agents, logged to logs/, 60s time limit
        python manager.py -n 100 -p0 my_agent.py -p1 random_agent.py -l logs/ -t 60
    """

    DEFAULT_AGENT = "random_agent.py"

    parser = argparse.ArgumentParser(description="Run Oxono games between two agents")
    parser.add_argument("-n", type=int, default=1, help="Number of games to run (default: 1)")
    parser.add_argument("-p0", type=str, default=DEFAULT_AGENT, help="First player file (default: random_agent.py)")
    parser.add_argument("-p1", type=str, default=DEFAULT_AGENT, help="Second player file (default: random_agent.py)")
    parser.add_argument("-l", type=str, default=None, metavar="LOG_DIR", help="Log directory (default: no logging)")
    parser.add_argument("-t", type=int, default=300, help="Time limit for each player (default: 300)")
    args = parser.parse_args()

    manager = Manager(agent_files=[args.p0, args.p1], time_limit=args.t)

    results = {(1, -1): 0, (-1, 1): 0, (0, 0): 0}
    for i in range(args.n):
        log_path = str(Path(args.l) / f"log_{i}.txt") if args.l else None
        result = manager.play(path_to_file=log_path)
        results[result] = results.get(result, 0) + 1

        if i+1 < args.n:
            wins_p0   = results.get((1, -1), 0)
            wins_p1   = results.get((-1, 1), 0)
            draws     = results.get((0, 0), 0)

            print(f"\n=== Temporary results over {i+1} game(s) ===")
            print(f"  {args.p0:<20} wins: {wins_p0:>4}  ({100 * wins_p0 / (i+1):.1f}%)")
            print(f"  {args.p1:<20} wins: {wins_p1:>4}  ({100 * wins_p1 / (i+1):.1f}%)")
            print(f"  Draws:                     {draws:>4}  ({100 * draws / (i+1):.1f}%)")
    
    wins_p0   = results.get((1, -1), 0)
    wins_p1   = results.get((-1, 1), 0)
    draws     = results.get((0, 0), 0)

    print(f"\n=== Results over {args.n} game(s) ===")
    print(f"  {args.p0:<20} wins: {wins_p0:>4}  ({100 * wins_p0 / args.n:.1f}%)")
    print(f"  {args.p1:<20} wins: {wins_p1:>4}  ({100 * wins_p1 / args.n:.1f}%)")
    print(f"  Draws:                     {draws:>4}  ({100 * draws / args.n:.1f}%)")