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
from metrics import reward

down_width = 128
down_height = 128
down_points = (down_width, down_height)
EPOCH = 20

def detect_learning(training_set, training_label, spl, conv_net):

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

    for ppl, (im_b, lbl) in product(pipeline_list, zip(training_set, training_label)):
        im_g = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        im_g = cv2.rotate(im_g, cv2.ROTATE_180)
        ppl.browse(im_g)
        ppl.score(lbl)
        if ppl.reward == 0:
            pipeline_list_good.append(ppl)

        if len(pipeline_list_good) >= len(training_set): break

    print(str(len(pipeline_list_good)) + " pipelines with a positive score")

    cross_table = [[0] * len(training_set)] * len(pipeline_list_good)

    for i in range(len(pipeline_list_good)):
        for j in range(len(training_set)):
            print(str(i) + " " + str(j))
            im_g = cv2.cvtColor(training_set[j], cv2.COLOR_BGR2GRAY)
            im_g = cv2.rotate(im_g, cv2.ROTATE_180)
            ppl.browse(im_g)
            ppl.score(lbl)
            if ppl.reward == 0:
                cross_table[i][j] = 1

    #Training
    for k in range(0, EPOCH):
        print("Training epoch " + str(k))
        for i in range(len(pipeline_list_good)):
            for j in range(len(training_set)):
                if cross_table[i][j] == 1:
                    im_g = cv2.cvtColor(training_set[j], cv2.COLOR_BGR2GRAY)
                    im_g = cv2.rotate(im_g, cv2.ROTATE_180)
                    im = im_g
                    for alg in ppl.graph.nodes:
                        if spl.graph.nodes[alg]['subset'] != SINK :
                            exec(alg)
                            im_p = im
                            im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
                            im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            c_im = conv_net.forward(im_t)
                            spl.graph.nodes[alg]['QTable'].forward(c_im)
                            spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, ppl.reward)
