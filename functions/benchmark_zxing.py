from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import numpy as np
from processing_py import *
import torch
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
from utils import read_functions
from utils import sort_training_test
from utils import sort_no_training
from utils import zxing
from utils import tesser
from utils import zbar
from utils import conditionnal
from metrics import reward
from detect import detect_unsupervised
from detect import detect_supervised

#---CONSTANT DEFINITION---#
ZXING = 0
DETECT = 1

WIN_W = 800
WIN_H = 480

batch_size = 40

#---START PROCESSING---#
#app = App(WIN_W, WIN_H)
#app.background(255)


#---TRAINING PHASE---#
# suffix = 'colors'
# images, ground_truth, len_files = read_files(suffix)
# set, label = sort_no_training(images, ground_truth)

# for k in range(len(set)):
#     print('--------------------------------------------')
#     print('Testing image ' + str(k))
#     im_g = cv2.cvtColor(set[k], cv2.COLOR_BGR2GRAY)
#     detect_supervised(set[k], label[k], 'tree')

#---GET DATA---#
suffix = 'real'
images, ground_truth, len_files = read_files(suffix)
set, label = sort_no_training(images, ground_truth)

#---DEFINE RESULT
results = np.ndarray(shape=(2, len(set)), dtype=float)

#---BENCHMARK LOOP---#
for k in range(len(set)):
    im_g = cv2.cvtColor(set[k], cv2.COLOR_BGR2GRAY)
    print('--------------------------------------------')
    print('Testing image ' + str(k))

    #ZXING
    barre_code = zxing(im_g, zxingcpp.BarcodeFormat.EAN13)
    results[ZXING, k] = reward(barre_code, label[k])
    print(barre_code)

    #DETECT
    barre_code = detect_unsupervised(im_g, 'tree_tesser')
    results[DETECT, k] = reward(barre_code, label[k])
    print(barre_code)

bin_step = 0.1
counts_zxing, bins = np.histogram(results[ZXING,:], bins=10, range=(0.0, 1.0))
counts_detect, bins = np.histogram(results[DETECT,:], bins=10, range=(0.0, 1.0))

heatmap = [counts_zxing.tolist(), counts_detect.tolist()]

f_save = open("heatmap__zxing.csv", "w")
for i in (ZXING, DETECT):
    for j in range(len(heatmap[i])):
        f_save.write(str(heatmap[i][j]))
        f_save.write(";")
    f_save.write("\n")
f_save.close()
