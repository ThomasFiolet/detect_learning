import os
import random
import time
random.seed(time.time())

import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image, ImageOps
import numpy as np
from processing_py import *
import networkx as nx
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance
import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp
import csv

from utils import zxing
from utils import tesser
from utils import zbar
from metrics import reward
from utils import read_files
from utils import sort_no_training
from utils import conditionnal

# down_width = 512
# down_height = 512
# down_points = (down_width, down_height)

# #im_g = cv2.imread("data/real/img_1674223566_547149.jpg")
# im_g = cv2.imread("data/real/img_1674228168_236512.jpg")
# #im_g = cv2.resize(im_g, down_points, interpolation= cv2.INTER_LINEAR)
# im_g = cv2.cvtColor(im_g, cv2.COLOR_BGR2GRAY)
# im_g = cv2.rotate(im_g, cv2.ROTATE_180)

# im = im_g
# #sucess, im = saliency.computeSaliency(im)
# #im = cv2.equalizeHist((im*255).astype(np.uint8))
# #cv2.imshow('image', im)
# #cv2.waitKey(0)
# cv2.imwrite('./figures/image_0.png', im)

# im = cv2.equalizeHist(im.astype(np.uint8))
# cv2.imwrite('./figures/image_1.png', im)

# th, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
# cv2.imwrite('./figures/image_2.png', im)

# barre_code, decoded_info, decoded_type = cv_barcode_detector.detectAndDecode(im.astype(np.uint8))
# print(barre_code)
# # cv2.imwrite('./figures/image_3.png', im)

# with open('maps/europe/distances', 'r') as data:
#     plaintext = data.read()

# plaintext = plaintext.replace(',', ' ')
# print(plaintext)

def heuristic(map, source, target):
    try : return map.graph.edges[source, target]['weight']
    except : return float('inf')

n = random.randint(5,30)
p = random.random()
#G = nx.fast_gnp_random_graph(n, p, directed=True)
G = nx.circular_ladder_graph(15)
for (u,v,w) in G.edges(data=True):
    #w['weight'] = random.randint(0,5000)
    w['weight'] = 1

#source = random.choice(list(G))
#target = random.choice(list(G))

source = 1
target = 7

for i in range(30):
    path = nx.astar_path(G, source=source, target=target, heuristic=None, weight='weight')
    print(path)