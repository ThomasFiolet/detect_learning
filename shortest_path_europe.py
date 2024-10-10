import random
from random import shuffle
from random import choice
import time
random.seed(time.time())
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_device('cuda')

import networkx as nx
import numpy as np

from graph import Map
from utils import iter_extract
from utils import indx_extract
from shortest import Shortest
from shortest import Gen_dijsktra

#---------------------------------------------------
#FILTERED
# activation_list = ['elu', 'hardshrink', 'hardsigmoid', 'hardtanh', 'hardswish', 'leakyrelu', 'logsigmoid', 'relu', 'relu6', 'rrelu', 'selu', 'celu', 'gelu', 'sigmoid', 'silu', 'mish', 'softplus', 'softshrink', 'softsign', 'tanh', 'tanhshrink']
# activation_function_list = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink]

# criterion_function_list = [nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.BCEWithLogitsLoss, nn.HingeEmbeddingLoss, nn.HuberLoss, nn.SmoothL1Loss, nn.SoftMarginLoss]
# criterion_list = ['l1', 'mse', 'crossentropy', 'bcewithlogits', 'hingeembedding', 'huber', 'smoothl1', 'softmargin']
#---------------------------------------------------

activation_list = ['elu']
activation_function_list = [nn.ELU]

criterion_function_list = [nn.L1Loss]
criterion_list = ['l1']

#Negative Loss with logsimoid hingeembedding
map = Map('maps/europe/uncomplete/cities', 'maps/europe/uncomplete/distances', [])
n_cities = map.graph.number_of_nodes()
print('Number of cities in map : ' + str(n_cities))
dataset_size = int(n_cities*(n_cities - 1)/2)
print('Number of path possible : ' + str(dataset_size))

#europe : 150 approx. 75%
#large_100 = 495 approx. 10%
#large_1000 = 9990 approx. 2%
#training_size = 495
training_size = 150

#europe
testing_size = dataset_size - training_size

#large_100
#testing_size = 200

map = Map('maps/europe/uncomplete/cities', 'maps/europe/uncomplete/distances', [])
railroads_dijkstra, time_dijkstra, railroads_astar, time_astar = Gen_dijsktra(map, training_size, testing_size)

for activation, activation_function in zip(activation_list, activation_function_list):
    for criterion, criterion_function in zip(criterion_list, criterion_function_list):
        if activation != 'logsigmoid' or criterion != 'hingeembedding' :
            print(activation + ' ' + criterion)
            map = Map('maps/europe/uncomplete/cities', 'maps/europe/uncomplete/distances', activation_function())
            Shortest(map, activation, criterion, criterion_function(), training_size, testing_size, railroads_dijkstra, time_dijkstra, railroads_astar, time_astar)
