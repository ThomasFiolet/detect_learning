import networkx as nx

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager

import cv2 as cv2

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
def draw_graph(fig, G, layout, labels):

    plt.figure(fig.number)
    fig.clear()

    if layout == "planar_layout":
        pos_nodes = nx.planar_layout(G)
    if layout == "multipartite_layout":
        pos_nodes = nx.multipartite_layout(G)
    node_width = [G[u][v]['weight']*3 for u,v in G.edges]
    node_alpha = 1

    colors = [(255/255, 255/255, 255/255)]

    nx.draw_networkx_nodes(G, pos_nodes, node_size=500, node_color = "black", edgecolors = "white")
    nx.draw_networkx_edges(G, pos_nodes, node_size=500, arrows=True, connectionstyle="arc3,rad=0.2", width=node_width, edge_color="white", arrowstyle = "->")

    path = 'SpaceMono-Regular.ttf'
    font_prop = font_manager.FontProperties(fname=path)

    for i, node in enumerate(G.nodes):
        x,y = pos_nodes[node]
        plt.text(x - 0.05, y - 0.06, s = labels[node], font_properties=font_prop, bbox=dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='round, pad=0.2'))
    plt.draw()

    fig.tight_layout()
    ax = plt.gca()
    ax.set_facecolor((30/255, 30/255, 30/255))
    for pos in ['right', 'top', 'bottom', 'left']: ax.spines[pos].set_visible(False) 
    for label_axis in ['x', 'y']: plt.tick_params(axis=label_axis, which='both', bottom=False, top=False, labelbottom=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

def labels_comp(sample_short_labels, spl_labels, algs):
    sample_short_labels = {i: algs[i].split('(', 1)[0] for i in range(0, len(algs))}
    sample_short_labels = {i: sample_short_labels[i].split(' = ')[1] for i in range(0, len(sample_short_labels))}
    for i in range(0, len(sample_short_labels)):
        if sample_short_labels[i] == "''.join": sample_short_labels[i] = "tesserocr.image_to_text"
    for k in spl_labels:
        if spl_labels[k] == "''.join": spl_labels[k] = "tesserocr.image_to_text"
    return sample_short_labels, spl_labels, algs

def draw_current_image(im):
    imS = cv2.resize(im, (480, 360))
    cv2.imshow("Current image", imS)
