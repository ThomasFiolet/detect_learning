import sys
print(sys.version)

import numpy as np
from numpy import asarray
print('Numpy version :', np.__version__)
print(np.finfo(float).eps)

import scipy

import cv2 as cv2
print('OpenCV version :', cv2.__version__)

import PyQt5

import matplotlib
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LinearSegmentedColormap
print('MatPlotLib version :', matplotlib.__version__)

import PIL
from PIL import Image, ImageOps
print('Pillow version :', PIL.__version__)

import pytesseract
print('PyTesseract version :', pytesseract.__version__)
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py311/bin/pytesseract'
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py311/bin/tesseract'

import tesserocr

import math

import random

from os import listdir
from os.path import join, isfile

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

import zxingcpp

import networkx as nx

def WFC(spl):
    pl = nx.DiGraph()
    nd = list(spl.nodes)[0]
    pl.add_node(nd)
    while nd != list(spl.nodes)[-1]:
        succ = list(spl.successors(nd))
        u = nd
        nd = random.choice(succ)
        v = nd
        pl.add_node(nd)
        pl.add_edge(u,v)
    return pl

def normalize_graph(G):
    for n in G.nodes:
        succ = G.successors(n)
        s_len = sum(1 for _ in succ)
        succ = G.successors(n)
        for s in succ:
            G[n][s]['weight'] /= s_len

alg = ["im = cv2.imread(r'test_image.jpg')",
       "th, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)",
       "im = cv2.Laplacian(im,cv2.CV_8U)",
       "im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)",
       "barre_codes = zxingcpp.read_barcodes(im, zxingcpp.BarcodeFormat.EAN13)"]

spl = nx.DiGraph()
spl.add_edge(alg[0], alg[1], weight = 1)
spl.add_edge(alg[0], alg[2], weight = 1)
spl.add_edge(alg[0], alg[3], weight = 1)
spl.add_edge(alg[0], alg[4], weight = 1)
spl.add_edge(alg[1], alg[2], weight = 1)
spl.add_edge(alg[1], alg[4], weight = 1)
spl.add_edge(alg[2], alg[1], weight = 1)
spl.add_edge(alg[2], alg[4], weight = 1)
spl.add_edge(alg[3], alg[1], weight = 1)
spl.add_edge(alg[3], alg[2], weight = 1)
spl.add_edge(alg[3], alg[4], weight = 1)

normalize_graph(spl)
#print(nx.adjacency_matrix(spl).todense())
pl = WFC(spl)
#nx.draw_networkx(pl)
#plt.draw()
print(pl.nodes)
for alg in pl:
    exec(alg)
for bc in barre_codes : print(bc.text)
