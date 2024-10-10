import random
from random import shuffle
from random import choice
import time
random.seed(time.time())
from random import shuffle
import itertools

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad

def Gen_dijsktra(map, training_size, testing_size) :

    railroads_dijkstra = []
    time_dijkstra = []
    railroads_astar = []
    time_astar = []

    couples = list(itertools.product(list(map.graph), list(map.graph)))
    random.shuffle(couples)

    print("Determining Dijkstra and A* performances")
    i = 1
    for (departure, arrival) in itertools.islice(couples, 0, training_size + testing_size):
        if departure != arrival:
            if i%100 == 0:
                print(str(i) + "th iteration")
            start = time.time()
            path_dijkstra = nx.shortest_path(map.graph, source=departure, target = arrival, weight = 'weight')
            end = time.time()
            optimal_time = end - start

            start = time.time()
            path_astar = nx.astar_path(map.graph, source=departure, target = arrival, weight = 'weight', heuristic = None) #Determinist, no need for heuristic
            end = time.time()
            heurist_time = end - start

            rl = Railroad()
            for town in path_dijkstra:
                map.current_node = town
                if rl.last_node != '':
                    rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                else:
                    rl.graph.add_node(map.current_node)
                rl.last_node = map.current_node
            railroads_dijkstra.append(rl)
            time_dijkstra.append(optimal_time)

            rl = Railroad()
            for town in path_astar:
                map.current_node = town
                if rl.last_node != '':
                    rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                else:
                    rl.graph.add_node(map.current_node)
                rl.last_node = map.current_node
            railroads_astar.append(rl)
            time_astar.append(heurist_time)
            i += 1

    return railroads_dijkstra, time_dijkstra, railroads_astar, time_astar