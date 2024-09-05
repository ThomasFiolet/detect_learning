import random
from random import shuffle

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad

N_st_per_node = 3

map = Map('maps/europe/cities', 'maps/europe/distances')

edges_list = map.graph.edges
N_edges_to_remove = int(len(edges_list)/2)

r = list(range(0, len(edges_list)))
random.shuffle(r)

edges_list_random = []
for k, i in zip(r, range(len(edges_list))):
    edges_list_random.append(edges_list[k])
    

for k, e in zip(range(0, N_edges_to_remove), edges_list):
    map.graph.remove_edge(e)

railroads = []




for city in map.graph: 
    for k in range(0, N_st_per_node):
        path = nx.shortest_path(map.graph, source=city, target = random.choice(list(map.graph)), weight = 'weight')
        rl = Railroad()
        for town in path:
            map.current_node = town
            if rl.last_node != '':
                rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                
            else:
                rl.graph.add_node(map.current_node)
            rl.last_node = map.current_node
        railroads.append(rl)
            
for railroad in railroads:
    print(railroad.graph)