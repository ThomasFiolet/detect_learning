import math

from processing_py import *
import networkx as nx
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance
import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import PIL
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp

from utils import zxing
from utils import tesser
from metrics import reward

LAMBDA = 0.2

class Pipeline:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.last_node = ""
        self.reward = 0
        self.barre_code = ""
        self.horizon = 8
        self.complexity = 2

    def zero_data(self):
        self.graph.clear()
        self.last_node = ""
        self.reward = 0
        self.barre_code = ""

    def append(self, alg):
        self.graph.add_node(alg)
        if self.last_node != '':
            self.graph.add_edge(self.last_node, alg)
        self.last_node = alg

    def browse(self, im_g):
        im = im_g
        #barre_code = ""
        for alg in self.graph:
            exec(alg)
        #self.barre_code = barre_code
        return im

    def score(self, ground_truth):
        self.reward = reward(self.barre_code, ground_truth)
        #print('Ground_Truth : ' + str(ground_truth))
        #print('Barre_Code : ' + str(self.barre_code))
        #print('Reward : ' + str(self.reward))
        #print('\n')

    def draw(self, app, WIN_W, WIN_H):
        pos = nx.spring_layout(self.graph)
        nx.set_node_attributes(self.graph, pos, 'pos')

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

        app.fill(0)
        for node in self.graph.nodes:
            app.ellipse(self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1], 20, 20)

        app.stroke(0, 50)
        app.fill(0,0)

        theta = 10
        for edge in self.graph.edges:
            x0 = self.graph.nodes[edge[0]]['pos'][0]
            y0 = self.graph.nodes[edge[0]]['pos'][1]
            x1 = self.graph.nodes[edge[1]]['pos'][0]
            y1 = self.graph.nodes[edge[1]]['pos'][1]
            
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

        # app.textSize(15)
        # app.textAlign(CENTER)
        # app.rectMode(CENTER)
        # for node in self.graph.nodes :
        #     app.fill(0,0,0,175)
        #     app.rect(self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1], len(self.graph.nodes[node]['name'])*15*0.6, 18)
        #     app.fill(255)
        #     app.text(self.graph.nodes[node]['name'], self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1] + 4)

