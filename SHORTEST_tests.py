import random
import time
random.seed(time.time())
from random import shuffle
import itertools
import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')

from graph import Map
from graph import Railroad
from utils import iter_extract
from utils import indx_extract

edges_ratio = 0.9
testing_size = 200
epoch = 30

CREATE_GRAPH = True
PRECOMPUTATION = True
TRAINING = True

p_min = 2
p_max = 5
#Real values between 6 and 11

activation = nn.Softplus
criterion = nn.CrossEntropyLoss

if CREATE_GRAPH:
    #Variating graph size from 64 to 2048
    print('Creating random graph with uniformly distributed weights between 0 :')
    for p in range(p_min, p_max + 1):
        print('-------------------')
        graph_size = pow(2, p)
        folder = 'maps/nodes_' + str(graph_size) + '/'

        f_save = open(folder + 'nodes', 'w')
        for i in range(1,graph_size + 1):
            f_save.write(str(i)); f_save.write("\n")
        f_save.close()

        #Create complete random graph
        adj = np.random.randint(size = (graph_size, graph_size), low = 1, high = 10000)
        print('Graph of size ' + str(graph_size) + ' generated.')

        #Save complete graph
        f_save = open(folder + 'distances_comma', 'w')
        for arr in adj:
            for d in arr:
                f_save.write(str(int(d))); f_save.write(",")
            f_save.write("\n")
        f_save.close()

        with open(folder + 'distances_comma', 'r') as file, open(folder + 'distances', 'w') as target:
            target.write(file.read().replace(',\n','\n'))
        os.remove(folder + 'distances_comma')

        #Create a map based on the complete graph adjacency matrix
        map = Map(folder + 'nodes', folder + 'distances', [])

        #Remove randomly half of edges
        edges_list = map.graph.edges
        edges_list = np.array(edges_list)
        N_edges_to_remove = int((1 - edges_ratio)*len(edges_list))
        r = list(range(0, len(edges_list)))
        random.shuffle(r)  
        edges_list_random = []

        for k in r:
            edges_list_random.append(edges_list[k])

        for e, i in zip(edges_list_random, range(len(edges_list_random))):
            if i > N_edges_to_remove: break
            map.graph.remove_edge(*e)

        print(str((1 - edges_ratio)*100) + ' percent of edges removed.')

        #Save uncomplete graph
        f_save = open(folder + 'distances_comma', "w")
        for arr in nx.to_numpy_array(map.graph):
            for d in arr:
                f_save.write(str(int(d)))
                f_save.write(",")
            f_save.write("\n")
        f_save.close()

        with open(folder + 'distances_comma', 'r') as file, open(folder + 'distances', "w") as target:
            target.write(file.read().replace(',\n','\n'))

        os.remove(folder + 'distances_comma')
        print('Graph of size ' + str(graph_size) + ' saved.')

if PRECOMPUTATION:
    for p in range(p_min, p_max + 1):
        print('-------------------')
        graph_size = pow(2, p)
        folder = 'maps/nodes_' + str(graph_size) + '/'

        map = Map(folder + 'nodes', folder + 'distances', activation)
        n_cities = map.graph.number_of_nodes()
        dataset_size = int(n_cities*(n_cities - 1)/2)

        training_size = map.graph.number_of_nodes()

        railroads_dijkstra = []
        time_dijkstra = []
        railroads_astar = []
        time_astar = []

        couples = list(itertools.product(list(map.graph), list(map.graph)))
        random.shuffle(couples)

        print("Determining Dijkstra and A* performances on graph of size " + str())
        for (departure, arrival) in itertools.islice(couples, 0, training_size + testing_size):
            if departure != arrival:

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

        f_save = open(folder + 'railroad_dijsktra_comma', 'w')
        for i in range(1, graph_size + 1):
            for town in railroads_dijkstra[i].graph:
                f_save.write(str(town)); f_save.write(",")
            f_save.write("\n")
        f_save.close()
        with open(folder + 'railroad_dijsktra_comma', "r") as file, open(folder + 'railroad_dijsktra', "w") as target:
            target.write(file.read().replace(',\n','\n'))
        os.remove(folder + 'railroad_dijsktra_comma')

        f_save = open(folder + 'time_dijkstra_comma', 'w')
        for i in range(1, graph_size + 1):
            f_save.write(str(time_dijkstra[i])); f_save.write("\n")
        f_save.close()
        with open(folder + 'time_dijkstra_comma', "r") as file, open(folder + 'time_dijkstra', "w") as target:
            target.write(file.read().replace(',\n','\n'))
        os.remove(folder + 'time_dijkstra_comma')

        f_save = open(folder + 'railroads_astar_comma', 'w')
        for i in range(1, graph_size + 1):
            for town in railroads_astar[i].graph:
                f_save.write(str(town)); f_save.write(",")
            f_save.write("\n")
        f_save.close()
        with open(folder + 'railroads_astar_comma', "r") as file, open(folder + 'railroads_astar', "w") as target:
            target.write(file.read().replace(',\n','\n'))
        os.remove(folder + 'railroads_astar_comma')

        f_save = open(folder + 'time_astar_comma', 'w')
        for i in range(1, graph_size + 1):
            f_save.write(str(time_astar[i])); f_save.write("\n")
        f_save.close()
        with open(folder + 'time_astar_comma', "r") as file, open(folder + 'time_astar', "w") as target:
            target.write(file.read().replace(',\n','\n'))
        os.remove(folder + 'time_astar_comma')

if TRAINING:

    for p in range(p_min, p_max + 1):
        print('-------------------')
        graph_size = pow(2, p)
        folder = 'maps/nodes_' + str(graph_size) + '/'
        map = Map(folder + 'nodes', folder + 'distances', activation)

        railroads_dijkstra = []
        railroads_astar = []

        #railroads_dijkstra_array = np.loadtxt(folder + 'railroad_dijsktra', delimiter=",")
        railroads_astar_array = np.loadtxt(folder + 'railroads_astar', delimiter=",")

        with open(folder + 'railroad_dijsktra') as file : railroads_dijkstra_array = [line.rstrip() for line in file]

        for i in range(0, railroads_dijkstra_array[0] - 1):
            rl = Railroad()
            for j in range(0, railroads_dijkstra_array[1] -1):
                current_node = railroads_dijkstra_array[i][j]
                if rl.last_node != '':
                    rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                else:
                    rl.graph.add_node(map.current_node)
                rl.last_node = current_node
            railroads_dijkstra.append(rl)

        for i in range(0, railroads_astar_array[0] - 1):
            rl = Railroad()
            for j in range(0, railroads_astar_array[1] -1):
                current_node = railroads_astar_array[i][j]
                if rl.last_node != '':
                    rl.append(map.current_node, map.graph[rl.last_node][map.current_node]['weight'])
                else:
                    rl.graph.add_node(map.current_node)
                rl.last_node = current_node
            railroads_astar.append(rl)

        folder = 'results/nodes_' + str(graph_size) + '/'
    
        for k in range(0, epoch): 
            print("EPOCH " + str(k))

            for city in map.graph.nodes:
                map.graph.nodes[city]['c_loss'] = 0
                map.graph.nodes[city]['i_loss'] = 0

            i = 0
            for railroad in itertools.islice(railroads_dijkstra, 0, training_size):
                i += 1
                iidx = list(map.graph).index(list(railroad.graph)[-1]) #Get last city
                input = torch.zeros([map.graph.number_of_nodes()], device="cuda")
                input[iidx] = 1 #Indicates the last city to reach to the neural net
                for city in railroad.graph:
                    if city is list(railroad.graph)[-1] : break
                    #print(city)
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

                    map.graph.nodes[city]['c_loss'] += loss.item()
                    map.graph.nodes[city]['i_loss'] += 1

            for city in map.graph.nodes:
                if map.graph.nodes[city]['i_loss'] > 0:
                        map.graph.nodes[city]['loss'].append(map.graph.nodes[city]['c_loss']/map.graph.nodes[city]['i_loss'])

        f_save = open("folder" + "/loss.csv", "w")
        for city in map.graph.nodes:
            f_save.write(map.graph.nodes[city]['name'])
            f_save.write(";")
            for l in map.graph.nodes[city]['loss']:
                f_save.write(str(l))
                f_save.write(";")
            f_save.write("\n")
        f_save.close()

        f_save = open(folder + "/results.csv", "w")
        f_save.write("Path")
        f_save.write(";")
        f_save.write("Dijkstra")
        f_save.write(";")
        f_save.write("A Star")
        f_save.write(";")
        f_save.write("Ours")
        f_save.write(";")
        f_save.write("Dijkstra")
        f_save.write(";")
        f_save.write("A Star")
        f_save.write(";")
        f_save.write("Ours")
        f_save.write(";")
        f_save.write("\n")

        for railroad_d, railroad_a, time_d, time_a in itertools.islice(zip(railroads_dijkstra, railroads_astar, time_dijkstra, time_astar), training_size, training_size + testing_size):
            optimal_distance = railroad_d.graph.size(weight="weight")
            optimal_time = time_d
            heurist_distance = railroad_a.graph.size(weight="weight")
            heurist_time = time_a

            departure = list(railroad_d.graph)[0]
            arrival = list(railroad_d.graph)[-1]

            map.current_node = departure    
            current_distance = 0
            print('------------')
            print('Departure : ' + departure + ', arrival : ' + arrival)

            start = time.time()

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
                if time.time() - start > 1.0:
                    print('Optimal distance : ' + str(optimal_distance))
                    print('Heurist distance : ' + str(heurist_distance))
                    print('No solution found...')
                    current_distance = 0
                    break

            if time.time() - start < 1.0 and current_distance > 0 :
                print('Optimal distance : ' + str(optimal_distance))
                print('Heurist distance : ' + str(heurist_distance))
                print('Current distance : ' + str(current_distance))

            end = time.time()
            exec_time = end - start
            print('Optimal time : ' + str(time_d))
            print('Heurist time : ' + str(time_a))
            print('Current time : ' + str(exec_time))

            f_save.write(departure)
            f_save.write("-")
            f_save.write(arrival)
            f_save.write(";")

            f_save.write(str(optimal_distance))
            f_save.write(";")
            f_save.write(str(heurist_distance))
            f_save.write(";")
            f_save.write(str(current_distance))
            f_save.write(";")
            f_save.write(str(time_d))
            f_save.write(";")
            f_save.write(str(time_a))
            f_save.write(";")
            f_save.write(str(exec_time))
            f_save.write(";")

            f_save.write("\n")
        f_save.close()