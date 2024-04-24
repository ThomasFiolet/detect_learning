import networkx as nx

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager

import cv2 as cv2

import plotly.graph_objects as go

#Initialization
# fig1 = plt.figure(1, figsize=(8, 8))
# fig1.set_facecolor((30/255, 30/255, 30/255))
# fig1.tight_layout()
# ax1 = plt.gca()
# ax1.set_facecolor((30/255, 30/255, 30/255))
# for pos in ['right', 'top', 'bottom', 'left']: ax1.spines[pos].set_visible(False) 
# for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)

# fig2 = plt.figure(2, figsize=(8, 8))
# #fig2 = plt.figure(2, figsize=(960/DPI, 960/DPI), dpi=DPI)
# fig2.set_facecolor((30/255, 30/255, 30/255))
# fig2.tight_layout()
# ax2 = plt.gca()
# ax2.set_facecolor((30/255, 30/255, 30/255))
# for pos in ['right', 'top', 'bottom', 'left']: ax2.spines[pos].set_visible(False) 
# for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)

plt.ion()
plt.show()
plt.pause(0.25)

#Functions
def labels_comp(pipeline_labels, sample_labels, algs):
    pipeline_labels = {i: algs[i].split('(', 1)[0] for i in range(0, len(algs))}
    pipeline_labels = {i: pipeline_labels[i].split(' = ')[1] for i in range(0, len(pipeline_labels))}
    for i in range(0, len(pipeline_labels)):
        if pipeline_labels[i] == "''.join": pipeline_labels[i] = "tesserocr.image_to_text"
    for k in sample_labels:
        if sample_labels[k] == "''.join": sample_labels[k] = "tesserocr.image_to_text"
    return pipeline_labels, sample_labels
