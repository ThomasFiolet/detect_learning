import math

from metrics import damerau_levenshtein

def reward(barre_code, ground_truth):

    if barre_code is None or len(barre_code) == 0:
        return 1

    if ground_truth is None:
        if len(barre_code) == 13: dl = 3
        else: dl = 10
    else:
        dl = damerau_levenshtein(barre_code, ground_truth)

    if dl == 0:
        return 0

    log_score = math.log(dl + 1, len(barre_code) + 1)

    #return math.tanh(2*log_score)/2 + 0.5
    return log_score