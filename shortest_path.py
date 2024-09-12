import random
from random import shuffle
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad
from utils import iter_extract
from utils import indx_extract

N_st_per_node = 6
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
        if rl not in railroads: railroads.append(rl)

for k in range(0, EPOCH): 
    for railroad in railroads:
        iidx = list(map.graph).index(list(railroad.graph)[-1]) #Get last city
        input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
        input[iidx] = 1 #Indicates the last city to reach to the neural net
        for city in railroad.graph:
            if city is list(railroad.graph)[-1] : break
            map.graph.nodes[city]['QTable'].train()
            output = map.graph.nodes[city]['QTable'].forward(input)
            target = torch.zeros(output.size(), device="cuda")
            next_city = iter_extract(railroad.graph.successors(city), 0) 
            succ = map.graph.successors(city)
            oidx = indx_extract(succ, next_city)
            target[oidx] = 1
            #print(output)
            #print(target)
            loss_item = map.graph.nodes[city]['learner'].train(map.graph.nodes[city]['QTable'].last_prediction, target)
            if city == "Paris":
                print('--------------')
                print("City : " + str(city) + " ; Loss : " + str(loss_item))
                print("Railroad : " + str(list(railroad.graph)))
                # for name, param in map.graph.nodes[city]['QTable'].named_parameters():
                #     if param.requires_grad:
                #         print(param.data)
                #print(list(map.graph.nodes[city]['QTable'].parameters())[0].grad)