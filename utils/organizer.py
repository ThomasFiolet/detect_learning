from os import listdir
from os.path import join, isfile
import cv2 as cv2
import numpy as np
import random

def read_files(suffix):

    image_path = 'data/' + suffix
    ground_truth_path = 'data/lists/barre-code-list-' + suffix + '.txt'

    files = [ f for f in sorted(listdir(image_path)) if isfile(join(image_path,f)) ]
    files = sorted(files)
    images = np.empty(len(files), dtype=object)
    for k in range(0, len(files)):
        images[k] = cv2.imread(join(image_path,files[k]))

    ground_truth_file = open(ground_truth_path, 'r')
    ground_truth = ground_truth_file.readlines()

    len_files = len(files)

    return images, ground_truth, len_files

def read_functions(folder):
    source_file = "functions/" + folder + "/sources"
    pipes_file = "functions/" + folder + "/pipes"
    sinks_file = "functions/" + folder + "/sinks"

    return source_file, pipes_file, sinks_file

def sort_training_test(training_set_size, images, ground_truth):
    training_set = []
    test_set = []
    training_label = []
    test_label = []

    r = list(range(0, len(images)))
    random.shuffle(r)

    for k, i in zip(r, range(len(images))):
        if i < training_set_size:
            training_set.append(images[k])
            training_label.append(ground_truth[k])
        else:
            test_set.append(images[k])
            test_label.append(ground_truth[k])

    return training_set, test_set, training_label, test_label