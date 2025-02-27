from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta

from Managers.GameDirector import GameDirector
AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]


...

# Ejemplo de ejecución
try:
    game_director = GameDirector(agents=all_agents, max_rounds=200, store_trace=False)
    game_trace = game_director.game_start(print_outcome=False)
except Exception as e:
    print(f"Error: {e}")
    return 0

...

# Análisis de resultados
last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
   
winner = max(victory_points, key=lambda player: int(victory_points[player]))
fitness = 0
if all_agents.index(chosen_agent) == int(winner.lstrip("J")):  
    fitness += 1