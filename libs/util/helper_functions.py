import numpy as np

def init_block(size, start, block_size, value):
    res = np.zeros(size,dtype=np.int16)
    sel = [slice(start[0],start[0]+block_size[0]),slice(start[1],start[1]+block_size[1]),slice(start[2],start[2]+block_size[2])]
    res[tuple(sel)]=value
    return res