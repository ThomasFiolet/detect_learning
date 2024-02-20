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

def labels_comp(pipeline_labels, sample_labels, algs):
    pipeline_labels = {i: algs[i].split('(', 1)[0] for i in range(0, len(algs))}
    pipeline_labels = {i: pipeline_labels[i].split(' = ')[1] for i in range(0, len(pipeline_labels))}
    for i in range(0, len(pipeline_labels)):
        if pipeline_labels[i] == "''.join": pipeline_labels[i] = "tesserocr.image_to_text"
    for k in sample_labels:
        if sample_labels[k] == "''.join": sample_labels[k] = "tesserocr.image_to_text"
    return pipeline_labels, sample_labels

def draw_current_image(im):
    imS = cv2.resize(im, (480, 360))
    cv2.imshow("Current image", imS)
    cv2.waitKey(1)

def plot_graph(G, pos, labels):
    edge_x = []
    edge_y = []
    for k, p in pos.items() : G.nodes[k]['pos'] = p
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    for label, node in zip(labels.items(), G.nodes()): G.nodes[node]['label'] = label

    l=[]
    [l.extend([k,v]) for k,v in labels.items()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=l)
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Sample Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    fig_w = go.FigureWidget(fig)

    return fig_w