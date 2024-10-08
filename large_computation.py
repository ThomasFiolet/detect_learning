import random
import time
random.seed(time.time())
from random import shuffle
import itertools
import os

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad

graph_size = 1000
f_save = open("maps/large_1000/nodes", "w")
for i in range(1,graph_size + 1):
    f_save.write(str(i))
    f_save.write("\n")
f_save.close()

adj = np.random.randint(size = (graph_size, graph_size), low = 1, high = 10000)
f_save = open("maps/large_1000/distances_comma", "w")
for arr in adj:
    for d in arr:
        f_save.write(str(int(d)))
        f_save.write(",")
    f_save.write("\n")
f_save.close()

with open("maps/large_1000/distances_comma", "r") as file, open(r"maps/large_1000/distances", "w") as target:
    target.write(file.read().replace(',\n','\n'))

os.remove("maps/large_1000/distances_comma")

map = Map('maps/large_1000/nodes', 'maps/large_1000/distances', [])

edges_list = map.graph.edges
edges_list = np.array(edges_list)
N_edges_to_remove = int(0.5*len(edges_list))

r = list(range(0, len(edges_list)))
random.shuffle(r)  

edges_list_random = []

for k in r:
    edges_list_random.append(edges_list[k])

for e, i in zip(edges_list_random, range(len(edges_list_random))):
    if i > N_edges_to_remove: break
    map.graph.remove_edge(*e)

print(nx.to_numpy_array(map.graph))

f_save = open("maps/large_1000/distances_comma", "w")
for arr in nx.to_numpy_array(map.graph):
    for d in arr:
        f_save.write(str(int(d)))
        f_save.write(",")
    f_save.write("\n")
f_save.close()

# f_save.write(open("maps/europe/uncomplete/distances", "w").read().replace(',\n','\n'))

with open("maps/large_1000/distances_comma", "r") as file, open(r"maps/large_1000/distances", "w") as target:
    target.write(file.read().replace(',\n','\n'))

os.remove("maps/large_1000/distances_comma") 