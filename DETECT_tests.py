from os import listdir
from os.path import join, isfile
import time
random.seed(time.time())
import random
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
from utils import read_files
from utils import read_join_dataset
from utils import read_functions
from utils import sort_training_test
from utils import sort_no_training
from utils import zxing
from utils import tesser
from utils import zbar
from utils import conditionnal
from metrics import reward
from detect import detect_init
from detect import detect_unsupervised
from detect import detect_learning

PRECOMPUTATION = True
TRAINING = True

dataset_size = 100
training_size = 30
testing_size = 50
testing_size = min(testing_size, dataset_size-training_size)

down_width = 128
down_height = 128
down_points = (down_width, down_height)

PIPE = 1
SOURCE = 0
SINK = 2

activation = nn.Softplus
criterion = nn.CrossEntropyLoss()

#---GET DATA---#
#images, ground_truth, len_files = read_join_dataset(['real', 'BarcodeTestDataset'])
images, ground_truth, len_files = read_join_dataset(['real'])

training_set, test_set, training_label, test_label = sort_training_test(training_size, images, ground_truth)

if PRECOMPUTATION:
    for k in range(len(test_set)):
        im_g = cv2.cvtColor(test_set[k], cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        print('--------------------------------------------')
        print('Testing image ' + str(k) + ' : ' + str(test_label[k]))

if TRAINING:

    spl, conv_net = detect_init('tree_reduced', True)

    for k in range(len(test_set)):
        ground_truth = training_label[k]
        im_g = cv2.cvtColor(test_set[k], cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        print('--------------------------------------------')
        print('Testing image ' + str(k) + ' : ' + str(test_label[k]))

        pipeline = Pipeline()

        im = im_g

        spl.current_node = "im = im_g"
        spl.graph.nodes[spl.current_node]['nuse'] += 1
        pipeline.append(spl.current_node)
        while spl.graph.nodes[spl.current_node]['subset']  != SINK :
            exec(spl.current_node)
            im_p = im
            im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
            im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            c_im = conv_net.forward(im_t)
            idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
            idx = idx.item()
            succ = spl.graph.successors(spl.current_node)
            spl.current_node = iter_extract(succ, idx)
            pipeline.append(spl.current_node)
            spl.graph.nodes[spl.current_node]['nuse'] += 1

        pipeline.browse(im_g)
        pipeline.score(ground_truth)
        barre_code = pipeline.barre_code

        for alg in pipeline.graph:
            if spl.graph.nodes[alg]['subset'] != SINK :
                parameters = list(conv_net.parameters) + list(spl.graph.nodes[alg]['QTable'].parameters())
                optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
                loss = criterion(spl.graph.nodes[alg]['QTable'].last_prediction, target)