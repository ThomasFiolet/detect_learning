import networkx as nx
import numpy as np
from fa2 import ForceAtlas2
from processing_py import *

import math

from learn import QSwitch
from learn import Learner

PIPE = 1
SOURCE = 0
SINK = 2

class Sample:
    def __init__(self, sources_file, sinks_file, pipes_file, adjacency_file, conv_net):
        self.current_node = "im = im_g"

        with open(sources_file) as file : sources_string = [line.rstrip() for line in file]
        with open(sinks_file) as file : sinks_string = [line.rstrip() for line in file]
        with open(pipes_file) as file : pipes_string = [line.rstrip() for line in file]
        node_string = sources_string + pipes_string + sinks_string

        self.N_sources = len(sources_string)
        self.N_sinks = len(sinks_string)
        self.N_pipes = len(pipes_string)

        self.N_nodes = self.N_sources + self.N_sinks + self.N_pipes

        self.adjacency = np.loadtxt(adjacency_file)

        self.graph = nx.DiGraph()

        self.edges_list = []
        for i in range(0, self.adjacency.shape[0]):
            for j in range(0, self.adjacency.shape[1]):
                if (self.adjacency[i][j] != 0).all():
                    self.edges_list.append((node_string[i], node_string[j], self.adjacency[i][j]))

        self.graph.add_weighted_edges_from(self.edges_list)

        for i, node in enumerate(self.graph.nodes):
            name = node
            name = name.split('(', 1)[0]
            name = name.split(' = ', 1)[1]
            self.graph.nodes[node]['name'] = name

            self.graph.nodes[node]['loss'] = []
            self.graph.nodes[node]['c_loss'] = 0
            self.graph.nodes[node]['i_loss'] = 0
            
            self.graph.nodes[node]['nuse'] = 0

            if name == "''.join": self.graph.nodes[node]['name'] = "tesserocr.image_to_text"

            n_outputs = sum(1 for _ in self.graph.successors(node))
            if conv_net is None : n_inputs = 8
            else : n_inputs = 29*29
            if n_outputs != 0:
                self.graph.nodes[node]['QTable'] = QSwitch(n_inputs, n_outputs, False)

            for alg in sources_string:
                if node == alg: self.graph.nodes[node]['subset'] = SOURCE

            for alg in sinks_string:
                if node == alg: self.graph.nodes[node]['subset'] = SINK

            for alg in pipes_string:
                if node == alg: self.graph.nodes[node]['subset'] = PIPE

    def find_output_idx(self, output):
        succ = self.graph.successors(self.current_node)
        idx = 0
        for i, v in enumerate(succ):
            if v is output: idx = i
        return idx

    def draw(self, app):
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

        app.textSize(15)
        app.textAlign(CENTER)
        app.rectMode(CENTER)
        for node in self.graph.nodes :
            app.fill(0,0,0,175)
            app.rect(self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1], len(self.graph.nodes[node]['name'])*15*0.6, 18)
            app.fill(255)
            app.text(self.graph.nodes[node]['name'], self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1] + 4)

        print("drawn")

    def set_pos(self, WIN_W, WIN_H):
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

        sample_undirected = self.graph.to_undirected()
        sample_numpy = nx.to_numpy_array(sample_undirected)

        pos = nx.multipartite_layout(self.graph, subset_key='subset', align='vertical', center = np.array([0, 0]), scale = 1)

        pos_force = forceatlas2.forceatlas2(sample_numpy, pos=np.asarray(list(pos.values()), dtype=np.float32), iterations = 1)
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

        nx.set_node_attributes(self.graph, pos_norm, 'pos')

class Map:
    def __init__(self, cities_file, adjacency_file, activation_function):

        self.current_node = ""

        with open(cities_file) as file : cities_string = [line.rstrip() for line in file]
        self.N_cities = len(cities_string)

        self.adjacency = np.loadtxt(adjacency_file, delimiter=",")

        self.graph = nx.DiGraph()

        self.edges_list = []
        for i in range(0, self.adjacency.shape[0]):
            for j in range(0, self.adjacency.shape[1]):
                if (self.adjacency[i][j] != 0).all():
                    self.edges_list.append((cities_string[i], cities_string[j], self.adjacency[i][j]))

        self.graph.add_weighted_edges_from(self.edges_list)

        for i, node in enumerate(self.graph.nodes):
            name = node
            self.graph.nodes[node]['name'] = name

            self.graph.nodes[node]['loss'] = []
            self.graph.nodes[node]['c_loss'] = 0
            self.graph.nodes[node]['i_loss'] = 0

            n_inputs = self.graph.number_of_nodes()
            n_outputs = sum(1 for _ in self.graph.successors(node))
            if n_outputs != 0:
                self.graph.nodes[node]['QTable'] = QSwitch(n_inputs, n_outputs, activation_function)
                self.graph.nodes[node]['learner'] = Learner(None, self.graph.nodes[node]['QTable'].parameters())

    def find_output_idx(self, output):
        succ = self.graph.successors(self.current_node)
        idx = 0
        for i, v in enumerate(succ):
            if v is output: idx = i
        return idx

    def draw(self, app):
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

        app.textSize(15)
        app.textAlign(CENTER)
        app.rectMode(CENTER)
        for node in self.graph.nodes :
            app.fill(0,0,0,175)
            app.rect(self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1], len(self.graph.nodes[node]['name'])*15*0.6, 18)
            app.fill(255)
            app.text(self.graph.nodes[node]['name'], self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1] + 4)

        print("drawn")

    def set_pos(self, WIN_W, WIN_H):
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
                        scalingRatio=5.0,
                        strongGravityMode=True,
                        gravity=5.0,

                        # Log
                        verbose=True)

        sample_undirected = self.graph.to_undirected()
        sample_numpy = nx.to_numpy_array(sample_undirected)

        #pos = nx.multipartite_layout(self.graph, subset_key='subset', align='vertical', center = np.array([0, 0]), scale = 1)
        pos = nx.circular_layout(self.graph)

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

        nx.set_node_attributes(self.graph, pos_norm, 'pos')