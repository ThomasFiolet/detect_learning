from os import listdir
from os.path import join, isfile
import random

import cv2 as cv2
import numpy as np
from processing_py import *
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import iter_extract

PIPE = 1
SOURCE = 0
SINK = 2

# WIN_W = 800
# WIN_H = 800

# app = App(WIN_W, WIN_H)
# app.background(255)

source_file = "sources"
pipes_files = "pipes"
sinks_file = "sinks"

conv_net = Conv()

spl = Sample(source_file, sinks_file, pipes_files, conv_net)
# spl.set_pos(WIN_W, WIN_H)
# spl.draw(app)

image_path = 'data/tests'
files = [ f for f in sorted(listdir(image_path)) if isfile(join(image_path,f)) ]
files = sorted(files)
images = np.empty(len(files), dtype=object)
for k in range(0, len(files)):
    images[k] = cv2.imread(join(image_path,files[k]))

# pipeline = Pipeline()
# pipeline.append(spl.current_node)
# print(spl.current_node)

down_width = 512
down_height = 512
down_points = (down_width, down_height)

eps = 0.5

for im_f in images:
    for i in range(0,50):
        pipeline = Pipeline()
        spl.current_node = "im = im_g"
        im_g = cv2.cvtColor(im_f, cv2.COLOR_BGR2GRAY)
        im_g = cv2.resize(im_g, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_g).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        c_im = conv_net.forward(im_t)
        #print("Current node :")
        while(spl.graph.out_degree(spl.current_node) > 0):
            if random.random() > eps:
                idx = torch.argmax(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
                idx = idx.item()
            else:
                idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
                spl.graph.nodes[spl.current_node]['QTable'].FORWARDED = 0
            succ = spl.graph.successors(spl.current_node)
            spl.current_node = iter_extract(succ, idx)
            pipeline.append(spl.current_node)
            #print(spl.current_node) 
        pipeline.browse(im_g)
        #print(pipeline.barre_code)
        pipeline.unsupervised()
        #print("Alg :")
        for alg in pipeline.graph:
            if spl.graph.nodes[alg]['subset'] != SINK and spl.graph.nodes[alg]['QTable'].FORWARDED == 1 :
                #print(alg)
                #print(spl.graph.nodes[alg]['QTable'].last_prediction)
                spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, pipeline.reward)