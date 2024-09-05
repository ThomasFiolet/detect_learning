import random

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad

N_st_per_node = 3

map = Map()
railroads = []

for city in map.graph:
    for k in range(0, N_st_per_node):
        path = nx.shortest_path(map.graph, source=city, target = random.choice(list(map.graph)), weight = 'weight')
        rl = Railroad()
        for town in path:
            map.current_node = town
            if rl.last_node != '':
                rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
            
