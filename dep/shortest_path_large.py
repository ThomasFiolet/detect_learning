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

#---------------------------------------------------
#large_100
activation_function_list = [nn.CELU, nn.ReLU, nn.Mish, nn.Softplus, nn.LeakyReLU, nn.Hardshrink, nn.CELU, nn.SELU, nn.LogSigmoid, nn.CELU, nn.ELU, nn.SELU, nn.SELU, nn.SiLU, nn.RReLU, nn.SiLU, nn.ELU]
activation_list = ['celu', 'relu', 'mish', 'softplus', 'leakyrelu', 'hardshrink', 'celu', 'selu', 'logsigmoid', 'celu', 'elu', 'selu', 'selu', 'silu', 'rrelu', 'silu', 'elu']

criterion_function_list = [nn.SmoothL1Loss, nn.CrossEntropyLoss, nn.MSELoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.SmoothL1Loss, nn.SoftMarginLoss, nn.BCEWithLogitsLoss, nn.CrossEntropyLoss, nn.HuberLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.MSELoss, nn.L1Loss]
criterion_list = ['smoothl1', 'crossentropy', 'mse', 'crossentropy', 'crossentropy', 'crossentropy', 'crossentropy', 'smoothl1', 'softmargin', 'bcewithlogits', 'crossentropy', 'huber', 'crossentropy', 'crossentropy', 'crossentropy', 'mse', 'l1']

#large_1000
#---------------------------------------------------
# activation_function_list = [nn.Softplus, nn.ELU]
# activation_list = ['softplus', 'elu']

# criterion_function_list = [nn.CrossEntropyLoss, nn.CrossEntropyLoss]
# criterion_list = ['crossentropy', 'crossentropy']

map = Map('maps/large_100/nodes', 'maps/large_100/distances', [])
n_cities = map.graph.number_of_nodes()
print('Number of cities in map : ' + str(n_cities))
dataset_size = int(n_cities*(n_cities - 1)/2)
print('Number of path possible : ' + str(dataset_size))

training_size = 400
testing_size = 200

for activation, activation_function, criterion, criterion_function in zip(activation_list, activation_function_list, criterion_list, criterion_function_list):
    print(activation + ' ' + criterion)
    map = Map('maps/large_100/nodes', 'maps/large_100/distances', activation_function())
    Shortest(map, activation, criterion, criterion_function())
