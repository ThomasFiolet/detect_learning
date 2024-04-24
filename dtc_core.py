import networkx as nx
import numpy as np

IN_LAYER = 1
IN_LAYER_DIVISION = 3
OUT_LAYER = 3
MAX_LAYER = sum(1 for _ in open('alg_list.txt'))

import dtc_graph
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm != 0: return v/np.linalg.norm(v)
    else: return v

def setup_sample_graph():
    with open("alg_list.txt") as file : alg = [line.rstrip() for line in file]

    sample_labels = {a: a.split('(', 1)[0] for a in alg}
    sample_labels = {a: sample_labels[a].split(' = ', 1)[1] for a in sample_labels}

    sample = nx.DiGraph()

    A = np.ones((MAX_LAYER, MAX_LAYER))

    for i in range(0, IN_LAYER): A[i][0] = 0
    for i in range(MAX_LAYER-OUT_LAYER, MAX_LAYER):
        for j in range(0, MAX_LAYER):
            A[i][j] = 0

    for i in range (IN_LAYER, MAX_LAYER-OUT_LAYER):
        for j in range(MAX_LAYER-OUT_LAYER, MAX_LAYER): A[i][j] = 1000

    edges_list = dtc_graph.edges_list_from_matrix(A, alg)
    sample.add_weighted_edges_from(edges_list)
    dtc_graph.normalize_graph(sample)
    
    pr_layer = len(sample.nodes) - IN_LAYER - OUT_LAYER
    in_layer = IN_LAYER
    for i, node in enumerate(sample.nodes):
        if in_layer > 0:
            sample.nodes[node]['subset'] = 0
            in_layer -= 1
        elif pr_layer > 0:
            sample.nodes[node]['subset'] = 1
            pr_layer -= 1
        else:
            sample.nodes[node]['subset'] = 2

    for k in sample_labels:
        if sample_labels[k] == "''.join": sample_labels[k] = "tesserocr.image_to_text"

    return sample, sample_labels

def update_sample_graph(sample, pipeline, labels, eps):
    for idx in pipeline:
        if idx + 1 >= len(pipeline.nodes): break
        u = labels[idx]
        v = labels[idx + 1]
        sample[u][v]['weight'] = max(sample[u][v]['weight'] + eps,0)

def wave_function_collapse(sample, EXPLORE):
    pipeline = nx.DiGraph()
    nd = list(sample.nodes)[0]
    i = 0
    labels = {i: nd}
    pipeline.add_node(i)
    while list(sample.successors(nd)):
        i = i + 1
        succ = list(sample.successors(nd))
        if not EXPLORE:
            succ_edges = sample.out_edges(nd)
            w = [sample[u][v]['weight'] for u, v in succ_edges]
            nd = random.choices(succ, weights = w, k = 1)[0]
        else:
            nd = random.choice(succ)
        labels[i] = nd
        pipeline.add_node(i)
        pipeline.add_edge(i-1,i)
        pipeline[i-1][i]['weight'] = 1

    return pipeline, labels