import numpy as np

def init_block(size, slice:slice, value):
    a = np.zeros(size)
    a[slice]=value
    return a