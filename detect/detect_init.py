from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from metrics import reward

def detect_init(function_folder):
    source_file, pipes_file, sinks_file, adjacency_file = read_functions(function_folder)
    conv_net = Conv()
    spl = Sample(source_file, sinks_file, pipes_file, adjacency_file, conv_net)
    
    return spl, conv_net