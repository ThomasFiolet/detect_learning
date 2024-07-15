import math

from metrics import damerau_levenshtein

def reward(barre_code, ground_truth):

    if barre_code is None or len(barre_code) == 0:
        return 1.0

    if ground_truth is None:
        if len(barre_code) == 13: dl = 3
        else: dl = 10
    else:
        dl = damerau_levenshtein(barre_code, ground_truth)

    if dl == 0:
        return 0.0
    else:
        if ground_truth is None:
            log_score = math.log(dl + 1, len(barre_code) + 1)
        else:
            log_score = math.log(dl + 1, max(len(barre_code), len(ground_truth)) + 1)
        return log_score