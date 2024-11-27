import math

from metrics import damerau_levenshtein
from metrics import check_EAN_13

def reward(barre_code, ground_truth):

    if barre_code is None or len(barre_code) == 0:
        return 0.0

    if ground_truth is None:
        if len(barre_code) == 13:
            if check_EAN_13(barre_code):
                dl = 0
            else:
                dl = 3
        else:
            dl = 0.8*len(barre_code)
        log_score = math.log(dl + 1, len(barre_code) + 1)
        return 1 - log_score
    else:
        dl = damerau_levenshtein(barre_code, ground_truth)
        log_score = math.log(dl + 1, max(len(barre_code), len(ground_truth)) + 1)
        return 1 - log_score
