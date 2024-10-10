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

#europe : 35
#large_100 : 15
#large_1000 : 15
EPOCH = 15

def Shortest(map, activation, criterion, criterion_function, training_size, testing_size, railroads_dijkstra, time_dijkstra, railroads_astar, time_astar):
    for k in range(0, EPOCH): 
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
                loss = criterion_function(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                map.graph.nodes[city]['c_loss'] += loss.item()
                map.graph.nodes[city]['i_loss'] += 1

        for city in map.graph.nodes:
            if map.graph.nodes[city]['i_loss'] > 0:
                    map.graph.nodes[city]['loss'].append(map.graph.nodes[city]['c_loss']/map.graph.nodes[city]['i_loss'])

    f_save = open("results_shortest_large/large_1000/" + "/loss_" + activation + "_" + criterion + ".csv", "w")
    for city in map.graph.nodes:
        f_save.write(map.graph.nodes[city]['name'])
        f_save.write(";")
        for l in map.graph.nodes[city]['loss']:
            f_save.write(str(l))
            f_save.write(";")
        f_save.write("\n")
    f_save.close()

    f_save = open("results_shortest_large/large_1000/" + "/results_" + activation + "_" + criterion + ".csv", "w")
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
            #europe
            #if current_distance > map.graph.size(weight="weight"):
            #large_100 large_1000
            if time.time() - start > 1.0:
                print('Optimal distance : ' + str(optimal_distance))
                print('Heurist distance : ' + str(heurist_distance))
                print('No solution found...')
                current_distance = 0
                break

        #europe
        #if current_distance < map.graph.size(weight="weight") and current_distance > 0 :
        #large_100 large_1000
        if time.time() - start < 1.0 and current_distance > 0 :
            #for city in railroad.graph: print(city)
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