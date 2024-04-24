import sys
import time
import random
from os import listdir
from os.path import join, isfile

import math
import numpy as np
from numpy import asarray
import scipy

import PyQt5

import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import PIL
from PIL import Image, ImageOps
import pytesseract
print('PyTesseract version :', pytesseract.__version__)
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

import networkx as nx
from fa2 import ForceAtlas2
import torch

import dtc_draw
import dtc_graph
import dtc_wrap
import dtc_core

from processing_py import *

#---------------------------------------

WIN_W = 1000
WIN_H = 1000

app = App(WIN_W, WIN_H) # create window: width, height
app.background(255) # set background:  red, green, blue
#app.scale(0.9)
# app.width = WIN_W
# app.height = WIN_H

sample, sample_labels = dtc_core.setup_sample_graph()
print(sample_labels)

pipeline = nx.DiGraph()
pipeline.add_node(0)
pipeline_labels = {0: "0"}

barre_code = ""

eps = 0.1
lda = 0
epoch = 20
e_number = 13 - 1

image_path = 'data/real'
files = [ f for f in sorted(listdir(image_path)) if isfile(join(image_path,f)) ]
files = sorted(files)
images = np.empty(len(files), dtype=object)


forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=10.0,
                        strongGravityMode=True,
                        gravity=10.0,

                        # Log
                        verbose=True)

sample_undirected = sample.to_undirected()
sample_numpy = nx.to_numpy_array(sample_undirected)

pos = nx.multipartite_layout(sample, subset_key='subset', align='vertical', center = np.array([0, 0]), scale = 1)

#pos = forceatlas2.forceatlas2_networkx_layout(sample_undirected, pos=None, iterations=2000)

pos_force = forceatlas2.forceatlas2(sample_numpy, pos=np.asarray(list(pos.values()), dtype=np.float32), iterations=2)
pos_force_list = list(pos_force)
for i, k in enumerate(pos):
    pos[k] = pos_force_list[i]

max_0 = 0
max_1 = 1
min_0 = 1000000000
min_1 = 1000000000

for k in pos.keys():
    if max_0 < pos[k][0]: max_0 = pos[k][0]
    if max_1 < pos[k][1]: max_1 = pos[k][1]
    if min_0 > pos[k][0]: min_0 = pos[k][0]
    if min_1 > pos[k][1]: min_1 = pos[k][1]

pos_norm = {}

for k, v in pos.items():
    pos_norm[k] = ((v[0] - (min_0))/(max_0 - min_0)*(WIN_W - 0.1*WIN_W) + 0.05*WIN_W, (v[1] - min_1)/(max_1 - min_1)*(WIN_H - 0.1*WIN_H) + 0.05*WIN_H)

nx.set_node_attributes(sample, pos_norm, 'pos')

app.fill(0)
for node in sample.nodes:
    app.ellipse(sample.nodes[node]['pos'][0], sample.nodes[node]['pos'][1], 20, 20)

app.stroke(0, 100)
app.fill(0,0)

theta = 10
for edge in sample.edges:
    x0 = sample.nodes[edge[0]]['pos'][0]
    y0 = sample.nodes[edge[0]]['pos'][1]
    x1 = sample.nodes[edge[1]]['pos'][0]
    y1 = sample.nodes[edge[1]]['pos'][1]
    
    pt0 = np.array([x0, y0])
    pt1 = np.array([x1, y1])

    if not(np.all(pt0 == pt1)):
        alpha = 180/math.pi*45
        line_vec0 = (pt1-pt0)
        alpha_vec0 = np.array(
            [math.cos(alpha)*line_vec0[0] + math.sin(alpha)*line_vec0[1],
            -math.sin(alpha)*line_vec0[0] + math.cos(alpha)*line_vec0[1]])
        ctrl_vec0 = np.array([alpha_vec0[1], -alpha_vec0[0]])
        cpx0 = pt0[0] + ctrl_vec0[0]
        cpy0 = pt0[1] + ctrl_vec0[1]

        line_vec1 = (pt0-pt1)
        alpha_vec1 = np.array(
            [math.cos(alpha)*line_vec1[0] - math.sin(alpha)*line_vec1[1],
             math.sin(alpha)*line_vec1[0] + math.cos(alpha)*line_vec1[1]])
        ctrl_vec1 = np.array([-alpha_vec1[1], alpha_vec1[0]])
        cpx1 = pt1[0] + ctrl_vec1[0]
        cpy1 = pt1[1] + ctrl_vec1[1]

        app.curve(cpx0, cpy0, x0, y0, x1, y1, cpx1, cpy1)

app.fill(255/2)
app.textSize(20)
app.textAlign(CENTER)
for lbl, node in zip(sample_labels.values(), sample.nodes):
    app.text(lbl, sample.nodes[node]['pos'][0], sample.nodes[node]['pos'][1])

app.redraw() # refresh the window




# for k in range(0, len(files)):
#     images[k] = cv2.imread(join(image_path,files[k]))
#     im_g = cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY)
    
#     for j in range(0, epoch):
#         print('RUNNING ITERATION : ' + str(k) + ', ' + str(j))
#         if j > e_number: EXPLORE = False
#         else: EXPLORE = True
#         pipeline, algs = dtc_core.wave_function_collapse(sample, EXPLORE)
#         for idx in pipeline:
#             exec(algs[idx])
#             dtc_draw.draw_current_image(im)
        
#         if barre_code is None:
#             dtc_core.update_sample_graph(sample, pipeline, algs, -eps)
#         elif len(barre_code) < 13:
#             dtc_core.update_sample_graph(sample, pipeline, algs, -eps)
#         else:
#             dtc_core.update_sample_graph(sample, pipeline, algs, +eps)
#         eps *= (1 - lda)
#         dtc_graph.normalize_graph(sample)
        
#         pipeline_labels, sample_labels = dtc_draw.labels_comp(pipeline_labels, sample_labels, algs)
