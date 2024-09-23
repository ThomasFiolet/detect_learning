from os import listdir
from os.path import join, isfile
import random
import time
random.seed(time.time())
import math
from itertools import product

import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx
from processing_py import *
import networkx as nx
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import PIL
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from utils import indx_extract
from metrics import reward
from metrics import compute_image_metrics

down_width = 128
down_height = 128
down_points = (down_width, down_height)
EPOCH = 50

def detect_learning(training_set, training_label, spl, conv_net, training_metrics):

    PIPE = 1
    SOURCE = 0
    SINK = 2

    pipeline_list = []
    pipeline_list_good = []

    #Generate Random Pipelines
    print("Generating pipelines")
    while len(pipeline_list) < len(training_set)*5:
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

    print(str(len(pipeline_list)) + " pipelines generated")

    cross_table = [[0] * len(training_set)] * len(training_set)

    #for i in range(len(pipeline_list)):
    #    for j in range(len(training_set)):
            #print("Filter pipeline " + str(i) + " with data " + str(j))
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
        i += 1
            #cross_table[i][j] = 1

        #if len(pipeline_list_good) >= len(training_set): break

    print(str(len(pipeline_list_good)) + " pipelines with a positive score")

    # cross_table = [[0] * len(training_set)] * len(pipeline_list_good)

    # for i in range(len(pipeline_list_good)):
    #     for j in range(len(training_set)):
    #         print("Filter pipeline " + str(i) + " with data " + str(j))
    #         im_g = cv2.cvtColor(training_set[j], cv2.COLOR_BGR2GRAY)
    #         im_g = cv2.rotate(im_g, cv2.ROTATE_180)
    #         ppl.browse(im_g)
    #         ppl.score(lbl)
    #         if ppl.reward == 0:
    #             cross_table[i][j] = 1

    #Training
    criterion = nn.CrossEntropyLoss()
    for k in range(0, EPOCH):
        epoch_loss = 0
        print("Training epoch " + str(k))
        for ppl in pipeline_list_good:
            for im_b in ppl.working_im:
                im_g = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
                im_g = cv2.rotate(im_g, cv2.ROTATE_180)
                im = im_g
                for alg in ppl.graph.nodes:
                    if spl.graph.nodes[alg]['subset'] != SINK :
                        exec(alg)
                        im_p = im
                        im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
                        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                        #im_t = 2*im_t - 1

                        if conv_net is None:
                            brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum = training_metrics[j]
                            c_im = torch.tensor([brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum], dtype=torch.float32)
                        else:
                            c_im = conv_net.forward(im_t)

                        output = spl.graph.nodes[alg]['QTable'].forward(c_im)
                        target = torch.clone(spl.graph.nodes[alg]['QTable'].last_prediction)
                        for k, t in enumerate(target): target[0][k] = 1 - ppl.reward
                        succ = spl.graph.successors(alg)
                        next_alg = iter_extract(ppl.graph.successors(alg), 0) 
                        oidx = indx_extract(succ, next_alg)
                        target[0][oidx] = ppl.reward

                        parameters = list(conv_net.parameters()) + list(spl.graph.nodes[alg]['QTable'].parameters())
                        optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        epoch_loss += loss.item()
        print("Epoch loss : " + str(epoch_loss))


    # criterion = nn.CrossEntropyLoss()
    # for k in range(0, EPOCH):
    #     loss_epoch = 0
    #     print("Training epoch " + str(k))
    #     for i in range(len(pipeline_list_good)):
    #         for j in range(len(training_set)):
    #             if cross_table[i][j] == 1:
    #                 im_g = cv2.cvtColor(training_set[j], cv2.COLOR_BGR2GRAY)
    #                 im_g = cv2.rotate(im_g, cv2.ROTATE_180)
    #                 im = im_g
    #                 for alg in pipeline_list_good[i].graph.nodes:
    #                     if spl.graph.nodes[alg]['subset'] != SINK :
    #                         exec(alg)
    #                         im_p = im
    #                         im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
    #                         im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    #                         if conv_net is None:
    #                             brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum = training_metrics[j]
    #                             c_im = torch.tensor([brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum], dtype=torch.float32)
    #                         else:
    #                             c_im = conv_net.forward(im_t)

    #                         output = spl.graph.nodes[alg]['QTable'].forward(c_im)
    #                         target = torch.clone(spl.graph.nodes[alg]['QTable'].last_prediction)
    #                         for k, t in enumerate(target): target[:][k] = 1 - pipeline_list_good[i].reward
    #                         succ = spl.graph.successors(alg)
    #                         next_alg = iter_extract(pipeline_list_good[i].graph.successors(alg), 0) 
    #                         oidx = indx_extract(succ, next_alg)
    #                         target[0][oidx] = pipeline_list_good[i].reward

    #                         parameters = list(conv_net.parameters()) + list(spl.graph.nodes[alg]['QTable'].parameters())
    #                         optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
    #                         loss = criterion(output, target)
    #                         loss.backward()
    #                         loss_epoch += loss.item()
    #                         optimizer.step()
    #                         optimizer.zero_grad()
    #     print("Epoch loss : " + str(loss_epoch))

                            # for k, t in enumerate(target):
                            #     target[:][k] = 1 - reward
                            # if spl.graph.nodes[alg]['learner'].choosen_idx >= 0 and spl.graph.nodes[alg]['learner'].choosen_idx < len(target):
                            #     target[:][spl.graph.nodes[alg]['learner'].choosen_idx] = reward
                            # else:
                            #     idx = torch.argmin(target)
                            #     target[:][idx] = reward

                            # spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, target)
