import sys
print(sys.version)

import time

import numpy as np
from numpy import asarray
print('Numpy version :', np.__version__)
print(np.finfo(float).eps)

import scipy

import cv2 as cv2
print('OpenCV version :', cv2.__version__)
bd = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

import PyQt5

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
print('MatPlotLib version :', matplotlib.__version__)

import PIL
from PIL import Image, ImageOps
print('Pillow version :', PIL.__version__)

import pytesseract
print('PyTesseract version :', pytesseract.__version__)
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'

import tesserocr

import math
import random

from os import listdir
from os.path import join, isfile

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

import zxingcpp

import networkx as nx

import torch

def update_SPL(spl, pl, labels, eps):
    for idx in pl:
        if idx + 1 >= len(pl.nodes): break
        u = labels[idx]
        v = labels[idx + 1]
        spl[u][v]['weight'] = max(spl[u][v]['weight'] + eps,0)

def WFC(spl, EXPLORE):
    pl = nx.DiGraph()
    nd = list(spl.nodes)[0]
    i = 0
    labels = {i: nd}
    pl.add_node(i)
    while list(spl.successors(nd)):
        i = i + 1
        succ = list(spl.successors(nd))
        if not EXPLORE:
            succ_edges = spl.out_edges(nd)
            w = [spl[u][v]['weight'] for u, v in succ_edges]
            nd = random.choices(succ, weights = w, k = 1)[0]
        else:
            nd = random.choice(succ)
        labels[i] = nd
        pl.add_node(i)
        pl.add_edge(i-1,i)
        pl[i-1][i]['weight'] = 1
    return pl, labels

def normalize_graph(G):
    for n in G.nodes:
        succ_edges = G.out_edges(n)
        W = 0
        for u, v in succ_edges:
            W += G[u][v]['weight']
        succ_edges = G.out_edges(n)
        for u, v in succ_edges:
            G[u][v]['weight'] /= W

def edges_list_from_matrix(A, nodes):
    e = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if (A[i][j] != 0).all(): e.append((nodes[i], nodes[j], A[i][j]))
    return e

def draw_graph(G, layout, labels):
    if layout == "planar_layout":
        pos_nodes = nx.planar_layout(G)
    if layout == "multipartite_layout":
        pos_nodes = nx.multipartite_layout(G)
    node_width = [G[u][v]['weight']*3 for u,v in G.edges]
    #node_alpha = [(1 - G[u][v]['weight'])*0.7 + 0.3 for u,v in G.edges]
    node_alpha = 1

    colors = [(255/255, 255/255, 255/255)]
    #colors = [(255/255, 255/255, 255/255), (120/255, 120/255, 120/255)]
    #colors = [(0/255, 69/255, 122/255), (144/255, 3/255, 12/255)]
    #cmap_uni = LinearSegmentedColormap.from_list("uni", colors, N=10)
    #mapper = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1, clip=True), cmap=cmap_uni)
    #edge_colors = [mapper.to_rgba(G[u][v]['weight']) for u,v in G.edges]

    nx.draw_networkx_nodes(G, pos_nodes, node_size=500, node_color = "black", edgecolors = "white")
    nx.draw_networkx_edges(G, pos_nodes, node_size=500, arrows=True, connectionstyle="arc3,rad=0.2", width=node_width, edge_color="white", arrowstyle = "->")

    path = 'SpaceMono-Regular.ttf'
    font_prop = font_manager.FontProperties(fname=path)

    for i, node in enumerate(G.nodes):
        x,y = pos_nodes[node]
        plt.text(x - 0.05, y - 0.06, s = labels[node], font_properties=font_prop, bbox=dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='round, pad=0.2'))
    plt.draw() 

#---------------------------------------

def setup_spl():
    alg = ["im = im_g",
        "th, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)",
        "im = cv2.Laplacian((im*255).astype(np.uint8),cv2.CV_8U)",
        "im = cv2.equalizeHist((im*255).astype(np.uint8))",
        "im = cv2.convertScaleAbs(im, alpha=1, beta=0.5)",
        "sucess, im = saliency.computeSaliency(im)",
        "im = cv2.Sobel(im, -1, 0, 1, ksize=9)",
        "try: barre_code = zxingcpp.read_barcodes(im, zxingcpp.BarcodeFormat.EAN13)[0].text\nexcept: barre_code = '0'",
        "barre_code = ''.join(c for c in ''.join(tesserocr.image_to_text(Image.fromarray((im*255).astype(np.uint8))).splitlines()) if c.isdecimal())",
        "retval, barre_code, decoded_type = bd.detectAndDecode((im*255).astype(np.uint8))"]

    spl_labels = {a: a.split('(', 1)[0] for a in alg}
    spl_labels = {a: spl_labels[a].split(' = ', 1)[1] for a in spl_labels}

    spl = nx.DiGraph()

    A = np.loadtxt("spl_mat.txt", dtype=int)

    edges_list = edges_list_from_matrix(A, alg)
    spl.add_weighted_edges_from(edges_list)
    normalize_graph(spl)
    in_layer = 1
    out_layer = 3
    pr_layer = len(spl.nodes) - in_layer - out_layer

    for i, node in enumerate(spl.nodes):
        if in_layer > 0:
            spl.nodes[node]['subset'] = 0
            in_layer -= 1
        elif pr_layer > 0:
            spl.nodes[node]['subset'] = 1
            pr_layer -= 1
        else:
            spl.nodes[node]['subset'] = 2

    return spl, spl_labels

#---------------------------------------

spl, spl_labels = setup_spl()

pl = nx.DiGraph()
pl.add_node(0)
short_labels = {0: "0"}

DPI = 192

fig1 = plt.figure(1, figsize=(8, 8))
#fig1 = plt.figure(1, figsize=(480/DPI, 480/DPI), dpi=DPI)
fig1.set_facecolor((30/255, 30/255, 30/255))
fig1.tight_layout()
draw_graph(pl, "planar_layout", short_labels)
ax1 = plt.gca()
ax1.set_facecolor((30/255, 30/255, 30/255))
for pos in ['right', 'top', 'bottom', 'left']: ax1.spines[pos].set_visible(False) 
for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)

fig2 = plt.figure(2, figsize=(8, 8))
#fig2 = plt.figure(2, figsize=(960/DPI, 960/DPI), dpi=DPI)
fig2.set_facecolor((30/255, 30/255, 30/255))
fig2.tight_layout()
draw_graph(spl, "multipartite_layout", spl_labels)
ax2 = plt.gca()
ax2.set_facecolor((30/255, 30/255, 30/255))
for pos in ['right', 'top', 'bottom', 'left']: ax2.spines[pos].set_visible(False) 
for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)

plt.ion()
plt.show()
plt.pause(0.25)

barre_code = ""

eps = 0.1
lda = 0
epoch = 20
e_number = 13 - 1

image_path = 'data/real'
files = [ f for f in sorted(listdir(image_path)) if isfile(join(image_path,f)) ]
files = sorted(files)
images = np.empty(len(files), dtype=object)

for k in range(0, len(files)):
    #print(join(image_path,files[k]))
    images[k] = cv2.imread(join(image_path,files[k]))
    im_g = cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY)
    for j in range(0, epoch):
        if j > e_number: EXPLORE = False
        else: EXPLORE = True
        pl, algs = WFC(spl, EXPLORE)
        for idx in pl:
            exec(algs[idx])
            imS = cv2.resize(im, (480, 360))
            cv2.imshow("Current image", imS)
        if barre_code is None:
            update_SPL(spl, pl, algs, -eps)
        elif len(barre_code) < 13:
            update_SPL(spl, pl, algs, -eps)
        else:
            update_SPL(spl, pl, algs, +eps)
        eps *= (1 - lda)
        normalize_graph(spl)

        short_labels = {i: algs[i].split('(', 1)[0] for i in range(0, len(algs))}
        short_labels = {i: short_labels[i].split(' = ')[1] for i in range(0, len(short_labels))}
        for i in range(0, len(short_labels)):
            if short_labels[i] == "''.join": short_labels[i] = "tesserocr.image_to_text"
        for k in spl_labels:
            if spl_labels[k] == "''.join": spl_labels[k] = "tesserocr.image_to_text"

        plt.figure(fig1.number)
        fig1.clear()
        draw_graph(pl, "planar_layout", short_labels)
        fig1.tight_layout()
        ax1 = plt.gca()
        ax1.set_facecolor((30/255, 30/255, 30/255))
        for pos in ['right', 'top', 'bottom', 'left']: ax1.spines[pos].set_visible(False) 
        for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        plt.figure(fig2.number)
        fig2.clear()
        draw_graph(spl, "multipartite_layout", spl_labels)
        plt.text(5, 5, barre_code, color = "white")
        fig2.tight_layout()
        ax2 = plt.gca()
        ax2.set_facecolor((30/255, 30/255, 30/255))
        for pos in ['right', 'top', 'bottom', 'left']: ax2.spines[pos].set_visible(False)
        for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)
        fig2.canvas.draw()
        fig2.canvas.flush_events()

        #plt.show()