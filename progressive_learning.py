from os import listdir
from os.path import join, isfile
import random

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

PIPE = 1
SOURCE = 0
SINK = 2

function_folder = "zxing"
source_file, pipes_file, sinks_file = read_functions(function_folder)

conv_net = Conv()

spl = Sample(source_file, sinks_file, pipes_file, conv_net)

down_width = 128
down_height = 128
down_points = (down_width, down_height)

eps = 0.75
P_ground_truth = 1
GROUND_TRUTH = True

train_epoch = 5
exec_epoch = 1

training_set_size = 50

suffix = 'real'
images, ground_truth, len_files = read_files(suffix)

training_set, test_set, training_label, test_label = sort_training_test(training_set_size, images, ground_truth)

for i in range(0,train_epoch):
    for k in range(len(training_set)):
        if random.random() < P_ground_truth: GROUND_TRUTH = True
        else: GROUND_TRUTH = False
        im_g = cv2.cvtColor(training_set[k], cv2.COLOR_BGR2GRAY)
        # print('---------------------------------')
        # print("TRAINING PHASE")
        # print('---------------------------------')
        pipeline = Pipeline()
        spl.current_node = "im = im_g"

        pipeline.append(spl.current_node)
        while spl.graph.nodes[spl.current_node]['subset']  != SINK :
            im_p = pipeline.browse(im_g)
            im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
            im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            c_im = conv_net.forward(im_t)

            c_eps = (pipeline.graph.number_of_nodes()/pipeline.horizon)**2
            if random.random() < c_eps:
                    spl.current_node = 'self.barre_code = zxing(im, zxingcpp.BarcodeFormat.EAN13)'
            elif random.random() > eps:
                idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
                idx = idx.item()
                succ = spl.graph.successors(spl.current_node)
                spl.current_node = iter_extract(succ, idx)
            else:
                idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
                spl.graph.nodes[spl.current_node]['QTable'].forward(c_im)
                succ = spl.graph.successors(spl.current_node)
                spl.current_node = iter_extract(succ, idx)
            if spl.graph.nodes[spl.current_node]['subset'] != SINK :
                spl.graph.nodes[spl.current_node]['learner'].choosen_idx = idx
            pipeline.append(spl.current_node)
            # print(spl.current_node)     

        # print('Computing score')
        if GROUND_TRUTH == True: pipeline.supervised(training_label[k])
        else: pipeline.unsupervised()

        bc = pipeline.barre_code

        for alg in pipeline.graph:
            if spl.graph.nodes[alg]['subset'] != SINK :
                #if spl.graph.nodes[alg]['QTable'].FORWARDED == 1 :
                # print('Training node : ' + alg)
                spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, pipeline.reward)

for k in range(len(test_set)):
    print('---------------------------------')
    print("EXEC PHASE")
    print('---------------------------------')
    pipeline = Pipeline()
    im_g = cv2.cvtColor(test_set[k], cv2.COLOR_BGR2GRAY)
    spl.current_node = "im = im_g"
    pipeline.append(spl.current_node)
    while spl.graph.nodes[spl.current_node]['subset']  != SINK :
        im_p = pipeline.browse(im_g)
        im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        c_im = conv_net.forward(im_t)

        #print(str(pipeline.graph.number_of_nodes()) + '/' + str(pipeline.horizon) + '**2')
        c_eps = (pipeline.graph.number_of_nodes()/pipeline.horizon)**2
        if random.random() < c_eps:
                spl.current_node = 'self.barre_code = zxing(im, zxingcpp.BarcodeFormat.EAN13)'
        else:
            idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
            idx = idx.item()
            succ = spl.graph.successors(spl.current_node)
            spl.current_node = iter_extract(succ, idx)
        pipeline.append(spl.current_node)
    pipeline.browse(im_g)

    for alg in pipeline.graph:
        print(alg) 
    print('Barrecode : ' + str(pipeline.barre_code))
    print('Groundtruth : ' + str(test_label[k]))
    if pipeline.barre_code is None:
        print(13)
    else:
        print(damerau_levenshtein_distance(str(pipeline.barre_code), test_label[k]))