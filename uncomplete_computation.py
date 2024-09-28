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

map = Map('maps/europe/raw/cities', 'maps/europe/raw/distances')
edges_list = map.graph.edges
edges_list = np.array(edges_list)
N_edges_to_remove = int(0.60*len(edges_list))

r = list(range(0, len(edges_list)))
random.shuffle(r)  

edges_list_random = []

for k in r:
    edges_list_random.append(edges_list[k])

for e, i in zip(edges_list_random, range(len(edges_list_random))):
    if i > N_edges_to_remove: break
    map.graph.remove_edge(*e)

print(nx.to_numpy_array(map.graph))

f_save = open("maps/europe/uncomplete/distances_comma", "w")
for arr in nx.to_numpy_array(map.graph):
    for d in arr:
        f_save.write(str(int(d)))
        f_save.write(",")
    f_save.write("\n")
f_save.close()

# f_save.write(open("maps/europe/uncomplete/distances", "w").read().replace(',\n','\n'))

with open("maps/europe/uncomplete/distances_comma", "r") as file, open(r"maps/europe/uncomplete/distances", "w") as target:
    target.write(file.read().replace(',\n','\n'))

os.remove("maps/europe/uncomplete/distances_comma") 