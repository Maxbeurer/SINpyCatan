import random
import numpy as np
import time
import os
import multiprocessing
from datetime import datetime

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
from deap import base, creator, tools, algorithms

# Lista de todos los agentes disponibles
AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
NUM_AGENTS = len(AGENTS)

# Parámetros del algoritmo genético (configurables)
POPULATION_SIZE = 50  # Tamaño de la población
CXPB = 0.8           # Probabilidad de cruce
MUTPB = 0.2           # Probabilidad de mutación
NGEN = 200          # Número de generaciones
TOURNAMENT_SIZE = 3   # Tamaño del torneo para selección
ELITE_SIZE = 5        # Número de mejores individuos que pasan directamente a la siguiente generación
GAMES_PER_EVAL = 10   # Número de partidas para evaluar cada individuo
SIGMA = 0.1           # Desviación estándar para la mutación gaussiana

# Configuración de DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Inicializar toolbox
toolbox = base.Toolbox()

def create_individual():
    """
    Crea un individuo como un vector de probabilidades que suman 1.0
    Cada valor representa la probabilidad de seleccionar un agente específico.
    """
    # Generar valores aleatorios
    ind = [random.random() for _ in range(NUM_AGENTS)]
    # Normalizar para que sumen 1.0
    total = sum(ind)
    return [val/total for val in ind]

# Registrar funciones en el toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_individual(individual, games_per_eval=GAMES_PER_EVAL):
    """
    Evalúa un individuo ejecutando partidas de Catan.
    
    Args:
        individual: Vector de probabilidades para seleccionar agentes
        games_per_eval: Número de partidas para evaluar
        
    Returns:
        Tupla con el fitness (valor compuesto que considera múltiples factores)
    """
    total_fitness = 0.0
    games_completed = 0
    
    # Pesos para los diferentes componentes del fitness
    WIN_WEIGHT = 0.6  # Peso para ganar la partida
    POINTS_WEIGHT = 0.3  # Peso para los puntos de victoria
    POSITION_WEIGHT = 0.1  # Peso para la posición final
    
    for _ in range(games_per_eval):
        # Seleccionar un agente según las probabilidades del individuo
        chosen_agent_idx = random.choices(range(NUM_AGENTS), weights=individual, k=1)[0]
        chosen_agent = AGENTS[chosen_agent_idx]
        
        # Seleccionar oponentes equiprobablemente entre todos los agentes
        all_agents = []
        for i in range(4):  # 4 jugadores en total
            if i == 0:  # El primer jugador es el agente elegido
                all_agents.append(chosen_agent)
            else:
                # Seleccionar un oponente aleatorio
                opponent_idx = random.randrange(NUM_AGENTS)
                all_agents.append(AGENTS[opponent_idx])
        
        # Ejecutar la partida
        try:
            game_director = GameDirector(agents=all_agents, max_rounds=200, store_trace=False)
            game_trace = game_director.game_start(print_outcome=False)
            
            # Verificar que la partida se completó correctamente
            if "game" not in game_trace:
                print("Error: La partida no generó un registro de juego válido")
                continue
                
            # Analizar resultados
            try:
                # Obtener la última ronda
                last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
                
                # Verificar que la ronda tiene turnos
                if not game_trace["game"][last_round]:
                    print("Error: La última ronda no tiene turnos registrados")
                    continue
                    
                # Obtener el último turno de la última ronda
                last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
                
                # Verificar que el turno tiene información de puntos de victoria
                if "end_turn" not in game_trace["game"][last_round][last_turn] or "victory_points" not in game_trace["game"][last_round][last_turn]["end_turn"]:
                    print("Error: No se encontró información de puntos de victoria")
                    continue
                    
                victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
                
                # Recopilar puntos de victoria de todos los jugadores
                player_points = {}
                for player, points in victory_points.items():
                    player_points[player] = int(points)
                
                # Ordenar jugadores por puntos (de mayor a menor)
                sorted_players = sorted(player_points.items(), key=lambda x: x[1], reverse=True)
                
                # Calcular componentes del fitness
                game_fitness = 0.0
                
                # 1. Victoria (0.6 del fitness total)
                if sorted_players[0][0] == "J0" or (len(sorted_players) > 1 and sorted_players[0][1] == sorted_players[1][1] and sorted_players[1][0] == "J0"):
                    game_fitness += WIN_WEIGHT
                
                # 2. Puntos de victoria normalizados (0.3 del fitness total)
                # Normalizar a un valor entre 0 y 1 (asumiendo que 10 puntos es el máximo)
                max_possible_points = 10.0
                points_ratio = min(1.0, player_points.get("J0", 0) / max_possible_points)
                game_fitness += POINTS_WEIGHT * points_ratio
                
                # 3. Posición final (0.1 del fitness total)
                # Calcular la posición del agente elegido
                position = 1
                for player, _ in sorted_players:
                    if player == "J0":
                        break
                    position += 1
                
                # Normalizar posición (1 es mejor, 4 es peor)
                position_score = (5 - position) / 4.0  # Convierte posición 1->1.0, 2->0.75, 3->0.5, 4->0.25
                game_fitness += POSITION_WEIGHT * position_score
                
                # Acumular fitness de esta partida
                total_fitness += game_fitness
                games_completed += 1
                    
            except (KeyError, ValueError) as e:
                print(f"Error al analizar los resultados: {e}")
                continue
                
        except Exception as e:
            print(f"Error en la partida: {e}")
            continue
    
    # Si no se completó ninguna partida, devolver 0
    if games_completed == 0:
        return (0.0,)
    
    # Devolver el fitness promedio de todas las partidas
    return (total_fitness / games_completed,)

# Registrar la función de evaluación
toolbox.register("evaluate", evaluate_individual)

def crossover_normalized(ind1, ind2):
    """
    Operador de cruce que mantiene la suma 1.0 en los individuos resultantes.
    """
    # Aplicar cruce de un punto
    tools.cxOnePoint(ind1, ind2)
    
    # Normalizar para mantener la suma 1.0
    sum1 = sum(ind1)
    sum2 = sum(ind2)
    
    ind1[:] = [val/sum1 for val in ind1]
    ind2[:] = [val/sum2 for val in ind2]
    
    return ind1, ind2

def mutate_gaussian_normalized(individual, sigma=SIGMA):
    """
    Operador de mutación gaussiana que mantiene la suma 1.0.
    """
    # Aplicar mutación gaussiana
    for i in range(len(individual)):
        individual[i] += random.gauss(0, sigma)
        # Asegurar que no hay valores negativos
        individual[i] = max(0.0, individual[i])
    
    # Normalizar para mantener la suma 1.0
    total = sum(individual)
    individual[:] = [val/total for val in individual]
    
    return individual,

# Registrar operadores genéticos
toolbox.register("mate", crossover_normalized)
toolbox.register("mutate", mutate_gaussian_normalized)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# Configuración inicial para map (se reemplazará en run_genetic_algorithm)
toolbox.register("map", map)

def run_genetic_algorithm(pop_size=POPULATION_SIZE, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                         elite_size=ELITE_SIZE, games_per_eval=GAMES_PER_EVAL):
    """
    Ejecuta el algoritmo genético con los parámetros especificados.
    
    Args:
        pop_size: Tamaño de la población
        cxpb: Probabilidad de cruce
        mutpb: Probabilidad de mutación
        ngen: Número de generaciones
        elite_size: Número de mejores individuos que pasan directamente
        games_per_eval: Número de partidas para evaluar cada individuo
        
    Returns:
        Población final, registro de estadísticas, y mejor individuo
    """
    try:
        # Crear población inicial
        population = toolbox.population(n=pop_size)
        total_workers = os.cpu_count()
        using_workers = max(1, int(total_workers*0.9))
        print(f"Usando {using_workers} trabajadores para evaluar individuos")
        pool = multiprocessing.Pool(using_workers)
        toolbox.register("map", pool.map)
        
        # Crear el objeto para registrar estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Crear el objeto para guardar el hall of fame (mejores individuos)
        hof = tools.HallOfFame(elite_size)
        
        # Evaluar la población inicial
        print("Evaluando población inicial...")
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Registrar estadísticas iniciales
        record = stats.compile(population)
        print(f"Gen 0: {record}")
        
        # Algoritmo genético con elitismo
        for gen in range(1, ngen + 1):
            print(f"Generación {gen}/{ngen}")
            start_time = time.time()
            
            try:
                # Seleccionar la siguiente generación
                offspring = toolbox.select(population, len(population) - elite_size)
                offspring = list(map(toolbox.clone, offspring))
                
                # Aplicar cruce
                for i in range(1, len(offspring), 2):
                    if i < len(offspring) - 1 and random.random() < cxpb:
                        toolbox.mate(offspring[i - 1], offspring[i])
                        del offspring[i - 1].fitness.values
                        del offspring[i].fitness.values
                
                # Aplicar mutación
                for i in range(len(offspring)):
                    if random.random() < mutpb:
                        toolbox.mutate(offspring[i])
                        del offspring[i].fitness.values
                
                # Evaluar individuos con fitness invalidado
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Actualizar el hall of fame
                hof.update(offspring)
                
                # Elitismo: añadir los mejores individuos de la generación anterior
                elite = tools.selBest(population, elite_size)
                offspring.extend(elite)
                
                # Reemplazar la población
                population[:] = offspring
                
                # Registrar estadísticas
                record = stats.compile(population)
                print(f"Gen {gen}: {record}")
                print(f"Tiempo de generación: {time.time() - start_time:.2f} segundos")
                
                # Guardar el mejor individuo de esta generación
                best_ind = tools.selBest(population, 1)[0]
                print(f"Mejor individuo: {best_ind}")
                print(f"Fitness: {best_ind.fitness.values[0]}")
                
            except Exception as e:
                print(f"Error en la generación {gen}: {e}")
                print("Continuando con la siguiente generación...")
                continue
        
        # Cerrar el pool de procesos
        pool.close()
        pool.join()
        
        return population, stats, hof
        
    except Exception as e:
        print(f"Error en el algoritmo genético: {e}")
        # Asegurar que el pool se cierra incluso si hay un error
        if 'pool' in locals():
            pool.close()
            pool.join()
        raise

def experiment_hyperparameters():
    """
    Realiza experimentos con diferentes configuraciones de hiperparámetros.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ga_experiments_{timestamp}.txt"
    
    # Configuraciones a probar
    pop_sizes = [30, 50, 100]
    cxpbs = [0.7, 0.8, 0.9]
    mutpbs = [0.1, 0.2, 0.3]
    tournament_sizes = [2, 3, 5]
    games_per_evals = [5, 10, 20]
    
    try:
        with open(results_file, "w") as f:
            f.write("Experimentos con hiperparámetros del algoritmo genético\n")
            f.write("=====================================================\n\n")
            
            # Experimento 1: Tamaño de población
            f.write("Experimento 1: Tamaño de población\n")
            f.write("----------------------------------\n")
            for pop_size in pop_sizes:
                try:
                    print(f"\nProbando tamaño de población: {pop_size}")
                    f.write(f"\nTamaño de población: {pop_size}\n")
                    
                    # Ejecutar algoritmo con número reducido de generaciones para pruebas
                    _, stats, hof = run_genetic_algorithm(pop_size=pop_size, ngen=10)
                    
                    # Guardar resultados
                    if len(hof) > 0:
                        best_ind = hof[0]
                        f.write(f"Mejor fitness: {best_ind.fitness.values[0]}\n")
                        f.write(f"Mejor individuo: {best_ind}\n\n")
                    else:
                        f.write("No se encontraron individuos válidos en el Hall of Fame\n\n")
                except Exception as e:
                    print(f"Error en experimento con tamaño de población {pop_size}: {e}")
                    f.write(f"Error: {e}\n\n")
                    continue
            
            # Experimento 2: Probabilidad de cruce
            f.write("\nExperimento 2: Probabilidad de cruce\n")
            f.write("------------------------------------\n")
            for cxpb in cxpbs:
                try:
                    print(f"\nProbando probabilidad de cruce: {cxpb}")
                    f.write(f"\nProbabilidad de cruce: {cxpb}\n")
                    
                    _, stats, hof = run_genetic_algorithm(cxpb=cxpb, ngen=10)
                    
                    if len(hof) > 0:
                        best_ind = hof[0]
                        f.write(f"Mejor fitness: {best_ind.fitness.values[0]}\n")
                        f.write(f"Mejor individuo: {best_ind}\n\n")
                    else:
                        f.write("No se encontraron individuos válidos en el Hall of Fame\n\n")
                except Exception as e:
                    print(f"Error en experimento con probabilidad de cruce {cxpb}: {e}")
                    f.write(f"Error: {e}\n\n")
                    continue
            
            # Experimento 3: Probabilidad de mutación
            f.write("\nExperimento 3: Probabilidad de mutación\n")
            f.write("---------------------------------------\n")
            for mutpb in mutpbs:
                try:
                    print(f"\nProbando probabilidad de mutación: {mutpb}")
                    f.write(f"\nProbabilidad de mutación: {mutpb}\n")
                    
                    _, stats, hof = run_genetic_algorithm(mutpb=mutpb, ngen=10)
                    
                    if len(hof) > 0:
                        best_ind = hof[0]
                        f.write(f"Mejor fitness: {best_ind.fitness.values[0]}\n")
                        f.write(f"Mejor individuo: {best_ind}\n\n")
                    else:
                        f.write("No se encontraron individuos válidos en el Hall of Fame\n\n")
                except Exception as e:
                    print(f"Error en experimento con probabilidad de mutación {mutpb}: {e}")
                    f.write(f"Error: {e}\n\n")
                    continue
            
            # Experimento 4: Tamaño del torneo
            f.write("\nExperimento 4: Tamaño del torneo\n")
            f.write("----------------------------------\n")
            # Guardar el valor original del tamaño del torneo para restaurarlo al final
            original_tournament_size = TOURNAMENT_SIZE
            
            for tournament_size in tournament_sizes:
                try:
                    print(f"\nProbando tamaño de torneo: {tournament_size}")
                    f.write(f"\nTamaño de torneo: {tournament_size}\n")
                    
                    # Actualizar el tamaño del torneo
                    toolbox.unregister("select")
                    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
                    
                    _, stats, hof = run_genetic_algorithm(ngen=10)
                    
                    if len(hof) > 0:
                        best_ind = hof[0]
                        f.write(f"Mejor fitness: {best_ind.fitness.values[0]}\n")
                        f.write(f"Mejor individuo: {best_ind}\n\n")
                    else:
                        f.write("No se encontraron individuos válidos en el Hall of Fame\n\n")
                except Exception as e:
                    print(f"Error en experimento con tamaño de torneo {tournament_size}: {e}")
                    f.write(f"Error: {e}\n\n")
                    continue
            
            # Restaurar el tamaño del torneo original
            toolbox.unregister("select")
            toolbox.register("select", tools.selTournament, tournsize=original_tournament_size)
            
            # Experimento 5: Número de partidas por evaluación
            f.write("\nExperimento 5: Número de partidas por evaluación\n")
            f.write("------------------------------------------------\n")
            for games_per_eval in games_per_evals:
                try:
                    print(f"\nProbando número de partidas: {games_per_eval}")
                    f.write(f"\nNúmero de partidas: {games_per_eval}\n")
                    
                    _, stats, hof = run_genetic_algorithm(games_per_eval=games_per_eval, ngen=10)
                    
                    if len(hof) > 0:
                        best_ind = hof[0]
                        f.write(f"Mejor fitness: {best_ind.fitness.values[0]}\n")
                        f.write(f"Mejor individuo: {best_ind}\n\n")
                    else:
                        f.write("No se encontraron individuos válidos en el Hall of Fame\n\n")
                except Exception as e:
                    print(f"Error en experimento con número de partidas {games_per_eval}: {e}")
                    f.write(f"Error: {e}\n\n")
                    continue
        
        print(f"Resultados de experimentos guardados en {results_file}")
    
    except Exception as e:
        print(f"Error general en los experimentos: {e}")

def main():
    """
    Función principal que ejecuta el algoritmo genético completo.
    """
    try:
        print("Algoritmo Genético para optimizar la selección de agentes en Catán")
        print("================================================================")
        print(f"Número de agentes disponibles: {NUM_AGENTS}")
        print(f"Parámetros: POPULATION_SIZE={POPULATION_SIZE}, CXPB={CXPB}, MUTPB={MUTPB}, NGEN={NGEN}")
        print(f"Ejecución paralela con {os.cpu_count()} núcleos disponibles")
        
        # Menú de opciones
        print("\nOpciones:")
        print("1. Ejecutar algoritmo genético con parámetros por defecto")
        print("2. Experimentar con diferentes hiperparámetros")
        print("3. Salir")
        
        choice = input("Seleccione una opción (1-3): ")
        
        if choice == "1":
            try:
                # Ejecutar algoritmo genético
                start_time = time.time()
                final_pop, stats, hof = run_genetic_algorithm()
                total_time = time.time() - start_time
                
                # Verificar que se obtuvieron resultados válidos
                if len(hof) == 0:
                    print("No se encontraron individuos válidos en el Hall of Fame")
                    return
                
                # Mostrar resultados
                print("\nResultados finales:")
                print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
                
                # Guardar el mejor individuo
                best_ind = hof[0]
                print("\nMejor individuo encontrado:")
                print(best_ind)
                print(f"Fitness: {best_ind.fitness.values[0]}")
                
                # Interpretar el mejor individuo
                print("\nProbabilidades de selección de agentes:")
                for i, prob in enumerate(best_ind):
                    agent_name = AGENTS[i].__name__
                    print(f"{agent_name}: {prob:.4f}")
                
                # Guardar resultados en un archivo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"ga_results_{timestamp}.txt"
                
                try:
                    with open(results_file, "w") as f:
                        f.write("Resultados del Algoritmo Genético para Catán\n")
                        f.write("==========================================\n\n")
                        f.write(f"Parámetros: POPULATION_SIZE={POPULATION_SIZE}, CXPB={CXPB}, MUTPB={MUTPB}, NGEN={NGEN}\n")
                        f.write(f"Tiempo total de ejecución: {total_time:.2f} segundos\n\n")
                        
                        f.write("Mejor individuo encontrado:\n")
                        f.write(f"{best_ind}\n")
                        f.write(f"Fitness: {best_ind.fitness.values[0]}\n\n")
                        
                        f.write("Probabilidades de selección de agentes:\n")
                        for i, prob in enumerate(best_ind):
                            agent_name = AGENTS[i].__name__
                            f.write(f"{agent_name}: {prob:.4f}\n")
                    
                    print(f"\nResultados guardados en {results_file}")
                except Exception as e:
                    print(f"Error al guardar los resultados: {e}")
                
            except Exception as e:
                print(f"Error al ejecutar el algoritmo genético: {e}")
            
        elif choice == "2":
            # Experimentar con hiperparámetros
            experiment_hyperparameters()
            
        else:
            print("Saliendo...")
            
    except Exception as e:
        print(f"Error general: {e}")

if __name__ == '__main__':
    main()
