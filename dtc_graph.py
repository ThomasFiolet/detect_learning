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