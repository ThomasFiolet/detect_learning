from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
from processing_py import *
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import iter_extract
from utils import read_files
from utils import read_functions
from utils import sort_training_test
from utils import sort_no_training
from metrics import reward

PIPE = 1
SOURCE = 0
SINK = 2

LAMBDA = 0.2

function_folder = "tree"
source_file, pipes_file, sinks_file, adjacency_file = read_functions(function_folder)

conv_net = Conv()

spl = Sample(source_file, sinks_file, pipes_file, adjacency_file, conv_net)

down_width = 128
down_height = 128
down_points = (down_width, down_height)

rand_eps = 0.1
P_ground_truth = 1
GROUND_TRUTH = True

max_try = 15

score_eps = 0.1

training_set_size = 50

suffix = 'real'
images, ground_truth, len_files = read_files(suffix)

set, label = sort_no_training(images, ground_truth)

score = 1
complexity = 2
pipeline = Pipeline()

for k in range(len(set)):
    cv2.imshow('Current Image', cv2.resize(set[k], (512, 512), interpolation= cv2.INTER_LINEAR))
    cv2.waitKey(1)
    score = 1
    complexity = 2
    print('---------------------')
    i = 0
    while score > score_eps and i < max_try:
        pipeline.zero_data()
        pipeline.complexity = min(complexity, pipeline.horizon)

        im_g = cv2.cvtColor(set[k], cv2.COLOR_BGR2GRAY)

        spl.current_node = "im = im_g"
        pipeline.append(spl.current_node)

        while spl.graph.nodes[spl.current_node]['subset']  != SINK :
            im_p = pipeline.browse(im_g)
            im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
            im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            c_im = conv_net.forward(im_t)
            if pipeline.graph.number_of_nodes() >= pipeline.complexity - 1:
                CHOOSE = random.randint(0, 2)
                if CHOOSE == 0: last_alg = 'self.barre_code = zxing(im, zxingcpp.BarcodeFormat.EAN13)'
                elif CHOOSE == 1: last_alg = 'self.barre_code = tesser(im)'
                else: last_alg = 'retval, self.barre_code, decoded_type = cv_barcode_detector.detectAndDecode((im*255).astype(np.uint8))'
                idx = spl.find_output_idx(last_alg)
                spl.graph.nodes[spl.current_node]['learner'].choosen_idx = idx
                spl.current_node = last_alg
            elif random.random() < rand_eps:
                idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
                spl.graph.nodes[spl.current_node]['QTable'].forward(c_im)
                succ = spl.graph.successors(spl.current_node)
                spl.current_node = iter_extract(succ, idx)
            else:
                idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
                idx = idx.item()
                succ = spl.graph.successors(spl.current_node)
                spl.graph.nodes[spl.current_node]['learner'].choosen_idx = idx
                spl.current_node = iter_extract(succ, idx)
            pipeline.append(spl.current_node)

        pipeline.browse(im_g)

        if GROUND_TRUTH == True: pipeline.score(label[k])
        else: pipeline.score(None)

        for alg in pipeline.graph:
            if spl.graph.nodes[alg]['subset'] != SINK :
                spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, pipeline.reward)

        score  = reward(pipeline.barre_code, label[k])
        print('Temporary score : ' + str(score))
        complexity += i%2
        i += 1

    if i >= max_try:
        print("Could not read barre_code")
        print('\n')
    else :
        for alg in pipeline.graph:
            print(alg) 
        print('Barrecode : ' + str(pipeline.barre_code))
        print('Groundtruth : ' + str(label[k]))
        score  = reward(pipeline.barre_code, label[k])
        print(score)
        print('\n')

    #score = 1