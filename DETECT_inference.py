from os import listdir
from os.path import join, isfile
import os
import re
import time
import random
random.seed(time.time())
import math

import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import numpy as np
from processing_py import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx
import tesserocr
import zxingcpp

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import iter_extract
from utils import indx_extract
from utils import read_files
from utils import read_join_dataset
from utils import read_functions
from utils import sort_training_test
from utils import sort_no_training
from utils import zxing
from utils import tesser
from utils import zbar
from utils import conditionnal
from utils import printProgressBar
from metrics import reward
from detect import detect_init
from detect import detect_unsupervised
from detect import detect_learning

#----------INITIALIZING----------#
print("0. Initializing")
printProgressBar(0, 1)

#PARAMETERS
#--NN META-PARAMETERS
activation = nn.Softplus

#--INIT STRUCTURES
spl, conv_net = detect_init('tree_reduced', True, activation())

#--FOLDERS NAME
folder = 'models/detect/'

#--FUNCTIONS
SOURCE = 0
PIPE = 1
SINK = 2

printProgressBar(1, 1)

#----------INITIALIZING----------#
print("1. Retrieving models")  ; inc_ct = 0 ; max_ct = len(spl.graph) + 1

for alg in spl.graph:
    if spl.graph.nodes[alg]['subset'] != SINK:
        spl.graph.nodes[alg]['QTable'].load_state_dict(torch.load(folder + spl.graph.nodes[alg]['name'] + '.pt', weights_only=True))
        spl.graph.nodes[alg]['QTable'].eval()
    inc_ct += 1
    printProgressBar(inc_ct, max_ct)

conv_net.load_state_dict(torch.load(folder + 'conv_net.pt', weights_only=True))
conv_net.eval()
inc_ct += 1
printProgressBar(inc_ct, max_ct)

#----------INFERING------------#
print("1. Infering")   ; inc_ct = 0 ; max_ct = len(spl.graph) + 1