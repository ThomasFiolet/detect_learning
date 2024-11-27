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

PRECOMPUTATION = True
TRAINING = True

#real + BarcodeTestDataset
# dataset_size = 331 
# training_size = 100
# testing_size = 231

#real
dataset_size = 105
training_size = 60
testing_size = 45

#tests with real
# dataset_size = 105
# training_size = 10
# testing_size = 60

#testing_size = min(testing_size, dataset_size-training_size)

down_width = 128
down_height = 128
down_points = (down_width, down_height)

n_ppl = 200

PIPE = 1
SOURCE = 0
SINK = 2

activation = nn.Softplus
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
#criterion = nn.L1Loss()

spl, conv_net = detect_init('tree_reduced', True, activation())

EPOCH = 30

#---GET DATA---#
#dataset_list = ['real', 'BarcodeTestDataset']
dataset_list = ['real']
images, ground_truth, len_files = read_join_dataset(dataset_list)

training_set, test_set, training_label, test_label = sort_training_test(training_size, images, ground_truth)

if PRECOMPUTATION:

    pipeline_list = []
    pipeline_list_good = []

    #Generate Random Pipelines
    print("Generating pipelines")
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
            for alg in ppl.graph.nodes:
                print(alg)
        print('---------------------------')

    print(str(len(pipeline_list)) + " pipelines generated")
    
    i = 0
    for ppl in pipeline_list:
        print('\nTesting pipeline ' + str(i))
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
            print("Pipeline selected, " + str(j) + " images working")
            for alg in ppl.graph.nodes:
                print(alg)
        i += 1

    print(str(len(pipeline_list_good)) + " pipelines with a positive score")

if TRAINING:

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

        print("Training epoch " + str(k))
        for ppl in pipeline_list_good:
            for im_b in ppl.working_im:
                im_g = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
                im_g = cv2.rotate(im_g, cv2.ROTATE_180)
                im = im_g
                for alg in ppl.graph.nodes:
                    if spl.graph.nodes[alg]['subset'] != SINK :
                        #exec(alg)
                        im_p = im
                        im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
                        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                        #print("im_t : " + str(im_t))
                        c_im = conv_net.forward(im_t)
                        #print("c_im : " + str(c_im))

                        output = spl.graph.nodes[alg]['QTable'].forward(c_im)
                        #print("output : " + str(output))
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
        print(epoch_loss)
        print("-----------------------------------------------------------------------------")

        for alg in spl.graph.nodes:
            if spl.graph.nodes[alg]['i_loss'] > 0:
                    spl.graph.nodes[alg]['loss'].append(spl.graph.nodes[alg]['c_loss']/spl.graph.nodes[alg]['i_loss'])

    f_save = open(folder + "loss.csv", "w")
    for alg in spl.graph.nodes:
        f_save.write(spl.graph.nodes[alg]['name'])
        f_save.write(";")
        for l in spl.graph.nodes[alg]['loss']:
            f_save.write(str(l))
            f_save.write(";")
        f_save.write("\n")
    f_save.close()

    ZXING = 0
    TESSER = 1
    CVBD = 2
    ZBAR = 3
    COND = 4
    DETECT = 5

    METHOD = ['ZXING', 'TESSER', 'CVBD', 'ZBAR', 'COND', 'DETECT']

    times_ZXING = []
    times_TESSER = []
    times_CVBD = []
    times_ZBAR = []
    times_COND = []
    times_DETECT = []

    results = np.ndarray(shape=(6, len(test_set)), dtype=float)

    for k in range(len(test_set)):
        ground_truth = test_label[k]
        im_g = cv2.cvtColor(test_set[k], cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        print('--------------------------------------------')
        print('Testing image ' + str(k) + ' : ' + str(test_label[k]))

        #ZXING
        start = time.time()
        barre_code = zxing(im_g, zxingcpp.BarcodeFormat.EAN13)
        results[ZXING, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_ZXING.append(exec_time)
        print(barre_code)

        #TESSERACT
        start = time.time()
        barre_code = tesser(im_g)
        results[TESSER, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_TESSER.append(exec_time)
        print(barre_code)

        #OPENCV
        start = time.time()
        barre_code, decoded_info, decoded_type = cv_barcode_detector.detectAndDecode(im_g)
        results[CVBD, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_CVBD.append(exec_time)
        print(barre_code)

        #ZBAR
        start = time.time()
        barre_code = zbar(im_g)
        results[ZBAR, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_ZBAR.append(exec_time)
        print(barre_code)

        #COND
        start = time.time()
        barre_code = conditionnal(im_g)
        results[COND, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_COND.append(exec_time)
        print(barre_code)

        #DETECT
        start = time.time()
        barre_code = detect_unsupervised(im_g, spl, conv_net)
        results[DETECT, k] = reward(barre_code, test_label[k])
        end = time.time()
        exec_time = end - start
        times_DETECT.append(exec_time)
        print(barre_code)

    #---SAVING RESULTS---#
    bin_step = 0.1
    counts_zxing, bins = np.histogram(results[ZXING,:], bins=10, range=(0.0, 1.0))
    counts_pytess, bins = np.histogram(results[TESSER,:], bins=10, range=(0.0, 1.0))
    counts_cvbd, bins = np.histogram(results[CVBD,:], bins=10, range=(0.0, 1.0))
    counts_zbar, bins = np.histogram(results[ZBAR,:], bins=10, range=(0.0, 1.0))
    counts_cond, bins = np.histogram(results[COND,:], bins=10, range=(0.0, 1.0))
    counts_detect, bins = np.histogram(results[DETECT,:], bins=10, range=(0.0, 1.0))

    heatmap = [counts_zxing.tolist(), counts_pytess.tolist(), counts_cvbd.tolist(), counts_zbar.tolist(), counts_cond.tolist(), counts_detect.tolist()]

    f_save = open(folder + "use.csv", "w")
    for alg in spl.graph.nodes:
        f_save.write(spl.graph.nodes[alg]['name'])
        f_save.write(";")
        f_save.write(str(spl.graph.nodes[alg]['nuse']))
        f_save.write("\n")
    f_save.close()

    f_save = open(folder + "heatmap.csv", "w")
    for i in (ZXING, TESSER, CVBD, ZBAR, COND, DETECT):
        f_save.write(METHOD[i])
        f_save.write(";")
        for j in range(len(heatmap[i])):
            f_save.write(str(heatmap[i][j]))
            f_save.write(";")
        f_save.write("\n")
    f_save.close()

    f_save = open(folder + "times.csv", "w")
    for i in (ZXING, TESSER, CVBD, ZBAR, COND, DETECT):
        f_save.write(METHOD[i]); f_save.write(";")
    f_save.write("\n")
    for i in range(len(test_set)):
        f_save.write(str(times_ZXING[i])); f_save.write(";")
        f_save.write(str(times_TESSER[i])); f_save.write(";")
        f_save.write(str(times_CVBD[i])); f_save.write(";")
        f_save.write(str(times_ZBAR[i])); f_save.write(";")
        f_save.write(str(times_COND[i])); f_save.write(";")
        f_save.write(str(times_DETECT[i])); f_save.write(";")
        f_save.write("\n")
    f_save.close()
