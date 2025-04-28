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
from metrics import reward
from detect import detect_init
from detect import detect_unsupervised
from detect import detect_learning

#DATASET_SIZES
#real + BarcodeTestDataset
# dataset_size = 331 
# training_size = 100
# testing_size = 231

#real
dataset_size = 105
training_size = 60
testing_size = 45

#IMAGE SIZE REDUCTION FOR NN
down_width = 128
down_height = 128
down_points = (down_width, down_height)

n_ppl = 200 #Number of pipelines
PIPE = 1
SOURCE = 0
SINK = 2

activation = nn.Softplus
criterion = nn.CrossEntropyLoss()

#INIT STRUCTURES
spl, conv_net = detect_init('tree_reduced', True, activation())

EPOCH = 30

#dataset_list = ['real', 'BarcodeTestDataset']
dataset_list = ['real']

#PREPARE FILES & DATASETS
images, ground_truth, len_files = read_join_dataset(dataset_list)
training_set, test_set, training_label, test_label = sort_training_test(training_size, images, ground_truth)

#GENERATE RANDOM PPL
pipeline_list = []
pipeline_list_good = []

while len(pipeline_list) < n_ppl:
    ppl = Pipeline()
    ppl.zero_data()
    spl.current_node = "im = im_g"
    ppl.append(spl.current_node)

    while spl.graph.nodes[spl.current_node]['subset']  != SINK :
        idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
        succ = spl.graph.successors(spl.current_node)
        spl.current_node = iter_extract(succ, idx)
        ppl.append(spl.current_node)

    if ppl not in pipeline_list:
        pipeline_list.append(ppl)

#TESTING GENERATED PPL
for ppl in pipeline_list:
    isWorking = 0
    j = 0
    for im_b, lbl in zip(training_set, training_label):
        im_g = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        ppl.browse(im_g)
        ppl.score(lbl)
        if ppl.reward == 0:
            ppl.working_im.append(im_b)
            isWorking = 1
            j += 1
    if isWorking == 1:
        pipeline_list_good.append(ppl)

#TRAINING
folder = 'results/detect'
for dataset_name in dataset_list:
    folder += '_' + dataset_name
if not os.path.exists(folder):
        os.makedirs(folder)
folder += '/'

for k in range(0, EPOCH):
    epoch_loss = 0

    for node in spl.graph.nodes:
        spl.graph.nodes[node]['c_loss'] = 0
        spl.graph.nodes[node]['i_loss'] = 0

for ppl in pipeline_list_good:
    for im_b in ppl.working_im:

        im_g = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        im = im_g

        for alg in ppl.graph.nodes:
            if spl.graph.nodes[alg]['subset'] != SINK :
                im_p = im
                im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
                im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

                c_im = conv_net.forward(im_t)
                output = spl.graph.nodes[alg]['QTable'].forward(c_im)

                target = torch.clone(spl.graph.nodes[alg]['QTable'].last_prediction)

                for k, t in enumerate(target): target[0][k] = 1 - ppl.reward
                succ = spl.graph.successors(alg)
                next_alg = iter_extract(ppl.graph.successors(alg), 0) 
                oidx = indx_extract(succ, next_alg)
                target[0][oidx] = ppl.reward

                parameters = list(conv_net.parameters()) + list(spl.graph.nodes[alg]['QTable'].parameters())
                optimizer = optim.SGD(parameters, lr=0.001, momentum=0.0)
                #optimizer = optim.Adam(parameters, lr=0.00001)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                spl.graph.nodes[alg]['c_loss'] += loss.item()
                epoch_loss =+ loss.item()
                spl.graph.nodes[alg]['i_loss'] += 1
