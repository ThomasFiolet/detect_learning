import os
import re
import math

p_min = 4
p_max = 10
for p in range(p_min, p_max + 1):

    graph_size = pow(2, p)
    print(str(graph_size) + '-----------------')
    folder = 'maps/nodes_' + str(graph_size) + '/'

    s = 0
    n_g = 0
    with open(folder + 'railroads_dijsktra') as file : 
        for rail in file:
            n_g = n_g + 1
            towns = (t for t in re.split(',|\n', rail) if t != '')
            for town in towns:
                s = s + 1

    l_g = s/n_g
    print('Mean path length : ' + str(l_g))
    print('Number of paths : ' + str(n_g))
    t_g = ((l_g-1)*n_g)/graph_size
    print('Trainings max per nodes : ' + str(t_g))
    #t_r = math.log2(t_g)
    t_r = int(t_g/graph_size*5)
    print('Trainings per nodes : ' + str(t_r))
    l_r = int(t_r*graph_size)-100
    print('Training set : ' + str(l_r))
    print('\n')