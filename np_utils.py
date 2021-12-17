
import numpy as np

def get_colors(src):
    src = np.reshape(src, (-1, 3))
    colors = np.array(list(set([tuple(t) for t in src])))
    return colors
