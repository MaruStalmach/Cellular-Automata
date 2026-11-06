import numpy as np


def init_block(size, start, block_size, value):
    res = np.zeros(size, dtype=np.uint8)
    sel = [
        slice(start[0], start[0] + block_size[0]),
        slice(start[1], start[1] + block_size[1]),
        slice(start[2], start[2] + block_size[2]),
    ]
    res[tuple(sel)] = value
    return res

def random_seeds(size, n, value):
    res = np.zeros(size,dtype=np.uint8)
    pos_x=np.random.randint(size[0], size=(n,))
    pos_y=np.random.randint(size[1], size=(n,))
    res[pos_x,pos_y,0]=value
    return res
        
