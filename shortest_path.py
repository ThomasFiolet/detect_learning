import random
from random import shuffle
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad
from utils import iter_extract
from utils import indx_extract

N_st_per_node = 4
dataset_size = 200
training_size = 160
testing_size = 40
EPOCH = 20

map = Map('maps/europe/cities', 'maps/europe/distances')

edges_list = map.graph.edges
edges_list = np.array(edges_list)
N_edges_to_remove = int(len(edges_list)/2)

r = list(range(0, len(edges_list)))
random.shuffle(r)  

edges_list_random = []

for k in r:
    edges_list_random.append(edges_list[k])

for e, i in zip(edges_list_random, range(len(edges_list_random))):
    if i > N_edges_to_remove: break
    map.graph.remove_edge(*e)

railroads = []

while len(railroads) < dataset_size:
    departure = ''; arrival = ''

    while departure == arrival:
        departure = random.choice(list(map.graph))
        arrival = random.choice(list(map.graph))

    path = nx.shortest_path(map.graph, source=departure, target = arrival, weight = 'weight')

    rl = Railroad()
    for town in path:
        map.current_node = town
        if rl.last_node != '':
            rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
            
        else:
            rl.graph.add_node(map.current_node)
        rl.last_node = map.current_node
    if rl not in railroads: railroads.append(rl)

# for city in map.graph: 
#     for k in range(0, N_st_per_node):
#         path = nx.shortest_path(map.graph, source=city, target = random.choice(list(map.graph)), weight = 'weight')
#         rl = Railroad()
#         for town in path:
#             map.current_node = town
#             if rl.last_node != '':
#                 rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                
#             else:
#                 rl.graph.add_node(map.current_node)
#             rl.last_node = map.current_node
#         if rl not in railroads: railroads.append(rl)

# print(len(railroads))

criterion = nn.CrossEntropyLoss()

for k in range(0, EPOCH): 
    epoch_loss = 0
    i = 0
    for railroad in itertools.islice(railroads,0,training_size):
        i += 1
        iidx = list(map.graph).index(list(railroad.graph)[-1]) #Get last city
        input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
        input[iidx] = 1 #Indicates the last city to reach to the neural net
        for city in railroad.graph:
            if city is list(railroad.graph)[-1] : break
            output = map.graph.nodes[city]['QTable'].forward(input)
            target = torch.zeros(output.size(), device="cuda")
            next_city = iter_extract(railroad.graph.successors(city), 0) 
            succ = map.graph.successors(city)
            oidx = indx_extract(succ, next_city)
            target[oidx] = 1

            parameters = map.graph.nodes[city]['QTable'].parameters()
            optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
    epoch_loss /= map.graph.number_of_nodes()*N_st_per_node
    print(epoch_loss)

for railroad in itertools.islice(railroads,training_size,training_size + testing_size):
    optimal_distance = railroad.graph.size(weight="weight")

    departure = list(railroad.graph)[0]
    arrival = list(railroad.graph)[-1]

    map.current_node = departure
    current_distance = 0
    print('------------')
    print('Departure : ' + departure + ', arrival : ' + arrival)  
    while map.current_node != arrival:

        iidx = list(map.graph).index(arrival)
        input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
        input[iidx] = 1
        output = map.graph.nodes[map.current_node]['QTable'].forward(input)
        oidx = torch.argmax(output)
        oidx = oidx.item()
        succ = map.graph.successors(map.current_node)
        next_city = iter_extract(succ, oidx)
        current_distance += map.graph[map.current_node][next_city]['weight']
        map.current_node = next_city
        if current_distance > map.graph.size(weight="weight"):
            print('No solution found...')
            break

    if current_distance < map.graph.size(weight="weight"):
        #for city in railroad.graph: print(city)
        print('Optimal distance : ' + str(optimal_distance))
        print('Current distance : ' + str(current_distance))
