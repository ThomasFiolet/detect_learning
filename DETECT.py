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

LEARNING = 1
INFERENCE = 1
PRIORS = 1
GENERATE = 1

inc_ct = 0
max_ct = 0

#----------INITIALIZING----------#
print("0. Initializing")
printProgressBar(0, 1)

#dataset_sizes
dataset_size = 105
training_size = 60
testing_size = 45

#image_size_reduction_for_nn
down_width = 128
down_height = 128
down_points = (down_width, down_height)

#number_of_pipelines
n_ppl = 10

#functions
SOURCE = 0
PIPE = 1
SINK = 2

#nn_meta_parameters
activation = nn.Softplus
criterion = nn.CrossEntropyLoss()
EPOCH = 30

#init_structures
spl, conv_net = detect_init('tree_reduced', True, activation())

printProgressBar(1, 1)

#----------PREPARING FILES & DATASETS----------#
print("1. Preparing dataset") ; inc_ct = 0 ; max_ct = 1
printProgressBar(inc_ct, max_ct)

dataset_list = ['real']
images, ground_truth, len_files = read_join_dataset(dataset_list)
training_set, test_set, training_label, test_label = sort_training_test(training_size, images, ground_truth)
images_check = [0] * len(training_set)

printProgressBar(1, max_ct)

if LEARNING:
    pipeline_list = []
    if PRIORS:
        #----------READING PROPOSED PPL----------#
        print("2.1. Reading pipelines") ; 

        path = 'functions/pipelines'
        files = [f for f in sorted(listdir(path)) if isfile(join(path,f)) ]

        inc_ct = 0 ; max_ct = len(files)
        printProgressBar(inc_ct, max_ct)

        for k in range(0, len(files)):
            algs = open(join(path,files[k]), 'r').read().splitlines()
            ppl = Pipeline()
            ppl.zero_data()
            for alg in algs:
                ppl.append(alg)
            inc_ct += 1
            pipeline_list.append(ppl)
            printProgressBar(inc_ct, max_ct)

    if GENERATE:
        #----------GENERATE RANDOM PPL----------#
        print("2.2. Generating pipelines") ; inc_ct = 0 ; max_ct = n_ppl
        printProgressBar(inc_ct, max_ct)

        while len(pipeline_list) < n_ppl:
            ppl = Pipeline()
            ppl.zero_data()
            spl.current_node = "im = im_g"
            ppl.append(spl.current_node)

            while spl.graph.nodes[spl.current_node]['subset'] != SINK :
                idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
                succ = spl.graph.successors(spl.current_node)
                spl.current_node = iter_extract(succ, idx)
                ppl.append(spl.current_node)

            if ppl not in pipeline_list:
                pipeline_list.append(ppl)
                inc_ct += 1

            printProgressBar(inc_ct, max_ct)

    print("Total number of pipelines : " + str(len(pipeline_list)) + "\n")

    #----------TESTING PPL----------#
    print("3. Testing pipelines") ; inc_ct = 0 ; max_ct = len(pipeline_list)
    printProgressBar(inc_ct, max_ct)

    pipeline_list_good = []

    for ppl in pipeline_list:
        isWorking = 0
        j = 0
        nb_images_tested = sum(x == 0 for x in images_check)
        for im_r, k, lbl in zip(training_set, range(0, len(images_check)), training_label):
            if images_check[k] == 0:
                im_g = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
                im_b = cv2.rotate(im_g, cv2.ROTATE_180)
                ppl.browse(im_b)
                ppl.score(lbl)
                if ppl.reward == 1:
                    ppl.working_im.append(im_r)
                    ppl.working_lbl.append(lbl)
                    isWorking = 1
                    images_check[k] = 1
                    j += 1
        inc_ct += 1
        printProgressBar(inc_ct, max_ct)
        if isWorking == 1:
            pipeline_list_good.append(ppl)
            print("Pipeline: ")
            for alg in ppl.graph: print(alg)
            print("working on " + str(len(ppl.working_im)) + '/' + str(nb_images_tested) + ' images tested.')
        else :
            print("Pipeline: ")
            for alg in ppl.graph: print(alg)
            print("NOT WORKING.")

    #----------TRAINING----------#
    print("4. Training") ; inc_ct = 0 ; max_ct = EPOCH
    printProgressBar(inc_ct, max_ct)

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
            for im_r, lbl in zip(ppl.working_im, ppl.working_lbl):

                im_g = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
                im_b = cv2.rotate(im_g, cv2.ROTATE_180)
                im = im_b

                ppl.browse(im_b)
                ppl.score(lbl)

                for alg in ppl.graph.nodes:
                    if spl.graph.nodes[alg]['subset'] != SINK :
                        im_p = im
                        im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
                        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

                        c_im = conv_net.forward(im_t)
                        output = spl.graph.nodes[alg]['QTable'].forward(c_im)

                        target = torch.clone(spl.graph.nodes[alg]['QTable'].last_prediction)

                        for k, t in enumerate(target): target[0][k] = (1 - ppl.reward)

                        succ = spl.graph.successors(alg)
                        next_alg = iter_extract(ppl.graph.successors(alg), 0) 
                        oidx = indx_extract(succ, next_alg)
                        target[0][oidx] = ppl.reward

                        parameters = list(conv_net.parameters()) + list(spl.graph.nodes[alg]['QTable'].parameters())
                        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.0)
                        #optimizer = optim.Adam(parameters, lr=0.001)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        spl.graph.nodes[alg]['c_loss'] += loss.item()
                        epoch_loss =+ loss.item()
                        spl.graph.nodes[alg]['i_loss'] += 1
        print(epoch_loss)

        inc_ct += 1
        printProgressBar(inc_ct, EPOCH)  

    if not os.path.exists('models/detect'):
        os.makedirs('models/detect')
    folder = 'models/detect/'

    for alg in spl.graph.nodes:
        if spl.graph.nodes[alg]['subset'] != SINK:
            torch.save(spl.graph.nodes[alg]['QTable'].state_dict(), folder + spl.graph.nodes[alg]['name'] + '.pt')

    torch.save(conv_net.state_dict(), folder + 'conv_net.pt')

if INFERENCE:
    #----------RETRIEVING MODELS----------#
    if not LEARNING:
        print("5. Retrieving models")  ; inc_ct = 0 ; max_ct = len(spl.graph) + 1

        folder = 'models/detect/'

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
    print("6. Infering")   ; inc_ct = 0 ; max_ct = len(spl.graph) + 1
    s = 0
    for im_r, lbl in zip(test_set, test_label):

        pipeline = Pipeline()
        pipeline.zero_data()
        spl.current_node = "im = im_g"
        pipeline.append(spl.current_node)
        im_g = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
        im_b = cv2.rotate(im_g, cv2.ROTATE_180)
        im_s = cv2.resize(im_b, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        c_im = conv_net.forward(im_t)

        while spl.graph.nodes[spl.current_node]['subset'] != SINK :

            output = spl.graph.nodes[spl.current_node]['QTable'].forward(c_im)
            print("\n-------")
            print("LIN_OUTPUT")
            print(output)
            succ = spl.graph.successors(spl.current_node)
            idx = torch.argmax(output[0])
            idx = idx.item()
            next_alg = iter_extract(succ, idx)
            print(spl.current_node)

            spl.current_node = next_alg
            pipeline.append(spl.current_node)

        print(spl.current_node)
        pipeline.browse(im_g)
        s += pipeline.score(lbl)
        barre_code = pipeline.barre_code
        print("----------------")
        print(str(pipeline.score(lbl)) + ' : ' + str(lbl) + ' | ' + str(barre_code))

    print("----------------")
    print("Results : ")
    print(str(s) + '/' + str(testing_size))