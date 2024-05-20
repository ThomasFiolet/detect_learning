import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm != 0: return v/np.linalg.norm(v)
    else: return v
