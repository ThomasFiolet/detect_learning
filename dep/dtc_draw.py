import networkx as nx

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager

import cv2 as cv2

import plotly.graph_objects as go
from processing_py import *

from fa2 import ForceAtlas2
import numpy as np
import math

def set_sample_pos(sample, WIN_W, WIN_H):
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

    pos_force = forceatlas2.forceatlas2(sample_numpy, pos=np.asarray(list(pos.values()), dtype=np.float32), iterations = 0)
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

def draw_sample_graph(app, sample, sample_labels):
    app.fill(0)
    for node in sample.nodes:
        app.ellipse(sample.nodes[node]['pos'][0], sample.nodes[node]['pos'][1], 20, 20)

    app.stroke(0, 50)
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

    app.textSize(15)
    app.textAlign(CENTER)
    app.rectMode(CENTER)
    for lbl, node in zip(sample_labels.values(), sample.nodes):
        app.fill(0,0,0,175)
        app.rect(sample.nodes[node]['pos'][0], sample.nodes[node]['pos'][1], len(lbl)*15*0.6, 18)
        app.fill(255)
        app.text(lbl, sample.nodes[node]['pos'][0], sample.nodes[node]['pos'][1] + 4)

#Functions
def labels_comp(pipeline_labels, sample_labels, algs):
    pipeline_labels = {i: algs[i].split('(', 1)[0] for i in range(0, len(algs))}
    pipeline_labels = {i: pipeline_labels[i].split(' = ')[1] for i in range(0, len(pipeline_labels))}
    for i in range(0, len(pipeline_labels)):
        if pipeline_labels[i] == "''.join": pipeline_labels[i] = "tesserocr.image_to_text"
    for k in sample_labels:
        if sample_labels[k] == "''.join": sample_labels[k] = "tesserocr.image_to_text"
    return pipeline_labels, sample_labels


#CHECK NICE GUI