from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from metrics import reward

def detect_unsupervised(im_g, function_folder):

    PIPE = 1
    SOURCE = 0
    SINK = 2

    source_file, pipes_file, sinks_file, adjacency_file = read_functions(function_folder)
    conv_net = Conv()
    spl = Sample(source_file, sinks_file, pipes_file, adjacency_file, conv_net)
    pipeline = Pipeline()

    down_width = 128
    down_height = 128
    down_points = (down_width, down_height)
    complexity = 2
    rand_eps = 0.0
    score_eps = 0.2
    max_try = 50
    i = 0
    score = 1
    barre_code = None

    while score > score_eps and i < max_try:
        pipeline.zero_data()
        pipeline.complexity = min(complexity, pipeline.horizon)

        spl.current_node = "im = im_g"
        pipeline.append(spl.current_node)

        while spl.graph.nodes[spl.current_node]['subset']  != SINK :
            im_p = pipeline.browse(im_g)
            im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
            im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            c_im = conv_net.forward(im_t)
            if random.random() < rand_eps:
                idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
                spl.graph.nodes[spl.current_node]['QTable'].forward(c_im)
                succ = spl.graph.successors(spl.current_node)
                spl.current_node = iter_extract(succ, idx)
            else:
                idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
                idx = idx.item()
                succ = spl.graph.successors(spl.current_node)
                spl.graph.nodes[spl.current_node]['learner'].choosen_idx = idx
                spl.current_node = iter_extract(succ, idx)
            pipeline.append(spl.current_node)

        pipeline.browse(im_g)

        pipeline.score(None)
        if pipeline.reward < score:
            score = pipeline.reward
            barre_code = pipeline.barre_code
        #print(pipeline.reward)

        #score = reward(pipeline.barre_code, None)

        #print('---pipeline---')
        #for i in range(50):
        for alg in pipeline.graph:
            #print(alg)
            if spl.graph.nodes[alg]['subset'] != SINK :
                spl.graph.nodes[alg]['learner'].train(spl.graph.nodes[alg]['QTable'].last_prediction, pipeline.reward)
        #print('---------')

        i += 1
        if random.random() > 0.5: complexity += 1
        rand_eps += 0.2
        #print(complexity)
        #print(rand_eps)

    return barre_code