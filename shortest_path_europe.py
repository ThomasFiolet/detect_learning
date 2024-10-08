import random
from random import shuffle
from random import choice
import time
random.seed(time.time())
from random import shuffle
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
from processing_py import *

import networkx as nx
import numpy as np

from graph import Map
from graph import Railroad
from utils import iter_extract
from utils import indx_extract
from shortest import Shortest

#---START PROCESSING---#
#WIN_W = 500
#WIN_H = 500
#app = App(WIN_W, WIN_H)
#app.background(255)

EPOCH = 30

activation = 'elu'
criterion = 'cross_entropy'
activation_function = nn.ELU()
criterion_function = nn.CrossEntropyLoss()

#---------------------------------------------------
#FULL
# activation_list = ['elu', 'hardshrink', 'hardsigmoid', 'hardtanh', 'hardswish', 'leakyrelu', 'logsigmoid', 'multiheadattention', 'prelu', 'relu', 'relu6', 'rrelu', 'selu', 'celu', 'gelu', 'sigmoid', 'silu', 'mish', 'softplus', 'softshrink', 'softsign', 'tanh', 'tanhshrink', 'threshold', 'glu', 'softmin', 'softmax', 'softmax2d', 'logsoftmax', 'adaptivelogsoftmaxwithloss']

# criterion_list = ['l1', 'mse', 'crossentropy', 'ctc', 'nll', 'poissonnll', 'gaussiannll', 'kldiv', 'bce', 'bcewithlogits', 'marginranking', 'hingeembedding', 'multilabelmargin', 'huber', 'smoothl1', 'softmargin', 'multilabelsoftmargin', 'cosineembedding', 'multimargin', 'tripletmargin', 'tripletmarginwithdistance']

# activation_function_list = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid, nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU, nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss]

# criterion_function_list = [nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.CTCLoss, nn.NLLLoss, nn.PoissonNLLLoss, nn.GaussianNLLLoss, nn.KLDivLoss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.MarginRankingLoss, nn.HingeEmbeddingLoss, nn.MultiLabelMarginLoss, nn.HuberLoss, nn.SmoothL1Loss, nn.SoftMarginLoss, nn.MultiLabelSoftMarginLoss, nn.CosineEmbeddingLoss, nn.MultiMarginLoss, nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss]
#---------------------------------------------------
#FILTERED
#activation_list = ['elu', 'hardshrink', 'hardsigmoid', 'hardtanh', 'hardswish', 'leakyrelu', 'logsigmoid', 'relu', 'relu6', 'rrelu', 'selu', 'celu', 'gelu', 'sigmoid', 'silu', 'mish', 'softplus', 'softshrink', 'softsign', 'tanh', 'tanhshrink']
#activation_function_list = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink]

# criterion_function_list = [nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.BCEWithLogitsLoss, nn.HingeEmbeddingLoss, nn.HuberLoss, nn.SmoothL1Loss, nn.SoftMarginLoss]
# criterion_list = ['l1', 'mse', 'crossentropy', 'bcewithlogits', 'hingeembedding', 'huber', 'smoothl1', 'softmargin']
#---------------------------------------------------

#Negative Loss with logsimoid hingeembedding

activation_function_list = [nn.LogSigmoid]
activation_list = ['logsigmoid']

criterion_function_list = [nn.HingeEmbeddingLoss]
criterion_list = ['hingeembedding']



for activation, activation_function in zip(activation_list, activation_function_list):
    #activation_function = activation_function()
    for criterion, criterion_function in zip(criterion_list, criterion_function_list):
        #criterion_function = criterion_function()
        print(activation + ' ' + criterion)
        map = Map('maps/europe/uncomplete/cities', 'maps/europe/uncomplete/distances', activation_function())
        Shortest(map, activation, criterion, criterion_function())

# n_cities = map.graph.number_of_nodes()
# print('Number of cities in map : ' + str(n_cities))
# dataset_size = int(n_cities*(n_cities - 1)/2)
# print('Number of path possible : ' + str(dataset_size))
# training_size = 50
# testing_size = dataset_size - training_size

# # edges_list = map.graph.edges
# # edges_list = np.array(edges_list)
# # N_edges_to_remove = int(0.75*len(edges_list))

# # r = list(range(0, len(edges_list)))
# # random.shuffle(r)  

# # edges_list_random = []

# # for k in r:
# #     edges_list_random.append(edges_list[k])

# # for e, i in zip(edges_list_random, range(len(edges_list_random))):
# #     if i > N_edges_to_remove: break
# #     map.graph.remove_edge(*e)

# railroads_dijkstra = []
# time_dijkstra = []
# railroads_astar = []
# time_astar = []

# #map.set_pos(WIN_W, WIN_H)
# #map.draw(app)

# # rnd_node_d = list(map.graph)
# # shuffle(rnd_node_d)
# # print(rnd_node_d)
# # rnd_node_a = list(map.graph)
# # shuffle(rnd_node_a)
# # print(rnd_node_a)

# # for departure in rnd_node_d:
# #     for arrival in rnd_node_a:

# couples = list(itertools.product(list(map.graph), list(map.graph)))
# random.shuffle(couples)

# for (departure, arrival) in couples:
#     if departure != arrival:

#         start = time.time()
#         path_dijkstra = nx.shortest_path(map.graph, source=departure, target = arrival, weight = 'weight')
#         end = time.time()
#         optimal_time = end - start

#         start = time.time()
#         path_astar = nx.astar_path(map.graph, source=departure, target = arrival, weight = 'weight', heuristic = None) #Determinist, no need for heuristic
#         end = time.time()
#         heurist_time = end - start

#         rl = Railroad()
#         for town in path_dijkstra:
#             map.current_node = town
#             if rl.last_node != '':
#                 rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
#             else:
#                 rl.graph.add_node(map.current_node)
#             rl.last_node = map.current_node
#         railroads_dijkstra.append(rl)
#         time_dijkstra.append(optimal_time)

#         rl = Railroad()
#         for town in path_astar:
#             map.current_node = town
#             if rl.last_node != '':
#                 rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
#             else:
#                 rl.graph.add_node(map.current_node)
#             rl.last_node = map.current_node
#         railroads_astar.append(rl)
#         time_astar.append(heurist_time)

# criterion = nn.CrossEntropyLoss()

# for k in range(0, EPOCH): 

#     for city in map.graph.nodes:
#         map.graph.nodes[city]['c_loss'] = 0
#         map.graph.nodes[city]['i_loss'] = 0

#     i = 0
#     for railroad in itertools.islice(railroads_dijkstra, 0, training_size):
#         i += 1
#         iidx = list(map.graph).index(list(railroad.graph)[-1]) #Get last city
#         input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
#         input[iidx] = 1 #Indicates the last city to reach to the neural net
#         for city in railroad.graph:
#             if city is list(railroad.graph)[-1] : break
#             #print(city)
#             output = map.graph.nodes[city]['QTable'].forward(input)
#             target = torch.zeros(output.size(), device="cuda")
#             next_city = iter_extract(railroad.graph.successors(city), 0) 
#             succ = map.graph.successors(city)
#             oidx = indx_extract(succ, next_city)
#             target[oidx] = 1

#             parameters = map.graph.nodes[city]['QTable'].parameters()
#             optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             map.graph.nodes[city]['c_loss'] += loss.item()
#             map.graph.nodes[city]['i_loss'] += 1

#     for city in map.graph.nodes:
#        if map.graph.nodes[city]['i_loss'] > 0:
#             map.graph.nodes[city]['loss'].append(map.graph.nodes[city]['c_loss']/map.graph.nodes[city]['i_loss'])

#     #print("-------------------------")

# f_save = open("results_shortest/" + activation + "/loss.csv", "w")
# for city in map.graph.nodes:
#     f_save.write(map.graph.nodes[city]['name'])
#     f_save.write(";")
#     for l in map.graph.nodes[city]['loss']:
#         f_save.write(str(l))
#         f_save.write(";")
#     f_save.write("\n")
# f_save.close()

# f_save = open("results_shortest/" + activation + "/distance.csv", "w")
# f_save.write("Path")
# f_save.write(";")
# f_save.write("Dijkstra")
# f_save.write(";")
# f_save.write("A Star")
# f_save.write(";")
# f_save.write("Ours")
# f_save.write(";")
# f_save.write("Dijkstra")
# f_save.write(";")
# f_save.write("A Star")
# f_save.write(";")
# f_save.write("Ours")
# f_save.write(";")
# f_save.write("\n")

# for railroad_d, railroad_a, time_d, time_a in itertools.islice(zip(railroads_dijkstra, railroads_astar, time_dijkstra, time_astar), training_size, training_size + testing_size):
#     optimal_distance = railroad_d.graph.size(weight="weight")
#     optimal_time = time_d
#     heurist_distance = railroad_a.graph.size(weight="weight")
#     heurist_time = time_a

#     departure = list(railroad_d.graph)[0]
#     arrival = list(railroad_d.graph)[-1]

#     map.current_node = departure    
#     current_distance = 0
#     print('------------')
#     print('Departure : ' + departure + ', arrival : ' + arrival)

#     start = time.time()

#     while map.current_node != arrival:

#         iidx = list(map.graph).index(arrival)
#         input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
#         input[iidx] = 1
#         output = map.graph.nodes[map.current_node]['QTable'].forward(input)
#         oidx = torch.argmax(output)
#         oidx = oidx.item()
#         succ = map.graph.successors(map.current_node)
#         next_city = iter_extract(succ, oidx)
#         current_distance += map.graph[map.current_node][next_city]['weight']
#         map.current_node = next_city
#         if current_distance > map.graph.size(weight="weight"):
#             print('Optimal distance : ' + str(optimal_distance))
#             print('Heurist distance : ' + str(heurist_distance))
#             print('No solution found...')
#             current_distance = 0
#             break

#     if current_distance < map.graph.size(weight="weight") and current_distance > 0 :
#         #for city in railroad.graph: print(city)
#         print('Optimal distance : ' + str(optimal_distance))
#         print('Heurist distance : ' + str(heurist_distance))
#         print('Current distance : ' + str(current_distance))

#     end = time.time()
#     exec_time = end - start
#     print('Optimal time : ' + str(time_d))
#     print('Heurist time : ' + str(time_a))
#     print('Current time : ' + str(exec_time))

#     f_save.write(departure)
#     f_save.write("-")
#     f_save.write(arrival)
#     f_save.write(";")

#     f_save.write(str(optimal_distance))
#     f_save.write(";")
#     f_save.write(str(heurist_distance))
#     f_save.write(";")
#     f_save.write(str(current_distance))
#     f_save.write(";")
#     f_save.write(str(time_d))
#     f_save.write(";")
#     f_save.write(str(time_a))
#     f_save.write(";")
#     f_save.write(str(exec_time))
#     f_save.write(";")

#     f_save.write("\n")

# f_save.close()