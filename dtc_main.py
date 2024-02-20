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

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
#pio.renderers.default = "vscode"

import networkx as nx
import torch

import dtc_draw
import dtc_graph
import dtc_wrap
import dtc_core

#---------------------------------------

sample, sample_labels = dtc_core.setup_sample_graph()

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

# fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
# fig.write_image("test.png")

#fig1 = plt.figure(1, figsize=(8, 8))
#fig2 = plt.figure(2, figsize=(8, 8))

fig = dtc_draw.plot_graph(sample, nx.multipartite_layout(sample), sample_labels)
fig.show(renderer="browser")

for k in range(0, len(files)):
    images[k] = cv2.imread(join(image_path,files[k]))
    im_g = cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY)
    
    for j in range(0, epoch):
        print('RUNNING ITERATION : ' + str(k) + ', ' + str(j))
        if j > e_number: EXPLORE = False
        else: EXPLORE = True
        pipeline, algs = dtc_core.wave_function_collapse(sample, EXPLORE)
        for idx in pipeline:
            exec(algs[idx])
            dtc_draw.draw_current_image(im)
        
        if barre_code is None:
            dtc_core.update_sample_graph(sample, pipeline, algs, -eps)
        elif len(barre_code) < 13:
            dtc_core.update_sample_graph(sample, pipeline, algs, -eps)
        else:
            dtc_core.update_sample_graph(sample, pipeline, algs, +eps)
        eps *= (1 - lda)
        dtc_graph.normalize_graph(sample)
        
        pipeline_labels, sample_labels = dtc_draw.labels_comp(pipeline_labels, sample_labels, algs)
        #dtc_draw.draw_graph(fig1, pipeline, "planar_layout", pipeline_labels)
        #dtc_draw.draw_graph(fig2, sample, "multipartite_layout", sample_labels)
