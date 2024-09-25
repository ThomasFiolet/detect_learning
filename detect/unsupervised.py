from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

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

from utils import zxing
from utils import tesser
from utils import zbar

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from metrics import reward
from metrics import compute_image_metrics

def detect_unsupervised(im_g, spl, conv_net):

    PIPE = 1
    SOURCE = 0
    SINK = 2

    pipeline = Pipeline()

    down_width = 128
    down_height = 128
    down_points = (down_width, down_height)

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
    pipeline.score(None)
    barre_code = pipeline.barre_code

    # complexity = 2
    # rand_eps = 0.0
    # score_eps = 0.2
    # max_try = 20
    # i = 0
    # score = 1
    # barre_code = None

    # while score > score_eps and i < max_try:
    #     pipeline.zero_data()
    #     pipeline.complexity = min(complexity, pipeline.horizon)

    #     spl.current_node = "im = im_g"
    #     pipeline.append(spl.current_node)

    #     while spl.graph.nodes[spl.current_node]['subset']  != SINK :
    #         im_p = pipeline.browse(im_g)
    #         im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
    #         im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #         if conv_net is None:
    #                             brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum = test_metrics
    #                             c_im = torch.tensor([brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum], dtype=torch.float32)
    #         else:
    #             c_im = conv_net.forward(im_t)
    #         if random.random() < rand_eps:
    #             idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
    #             spl.graph.nodes[spl.current_node]['QTable'].forward(c_im)
    #             succ = spl.graph.successors(spl.current_node)
    #             spl.current_node = iter_extract(succ, idx)
    #         else:
    #             idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
    #             idx = idx.item()
    #             succ = spl.graph.successors(spl.current_node)
    #             spl.graph.nodes[spl.current_node]['learner'].choosen_idx = idx
    #             spl.current_node = iter_extract(succ, idx)
    #         pipeline.append(spl.current_node)

    #     pipeline.browse(im_g)

    #     pipeline.score(None)
    #     if pipeline.reward < score:
    #         score = pipeline.reward
    #         barre_code = pipeline.barre_code

        # if pipeline.reward < 0.3:
        #     for alg in pipeline.graph:
        #         #print(alg)
        #         if spl.graph.nodes[alg]['subset'] != SINK :
        #             criterion = nn.CrossEntropyLoss()
        #             output = spl.graph.nodes[alg]['QTable'].last_prediction
        #             parameters = list(conv_net.parameters()) + list(spl.graph.nodes[alg]['QTable'].parameters())
        #             optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)

        #             target = torch.clone(spl.graph.nodes[alg]['QTable'].last_prediction)
        #             for k, t in enumerate(target): target[:][k] = 1 - pipeline.reward
        #             oidx = torch.argmin(spl.graph.nodes[alg]['QTable'].last_prediction)
        #             target[0][oidx] = pipeline.reward
        #             loss = criterion(output, target)
        #             loss.backward()
        #             optimizer.step()
        #             optimizer.zero_grad()

                #spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, pipeline.reward)

        # i += 1
        # if random.random() > 0.5: complexity += 1
        # rand_eps += 0.1

    return barre_code