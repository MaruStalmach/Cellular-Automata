import numpy as np
import time

state = np.arange(1000).reshape(10,10,-1)
state_padded = np.pad(state,((1,1),(1,1),(1,1)))
mask = np.array([
    [
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ],
    [
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ],
    [
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ],
    ])

def select_neighbors(location):
    state_padded = np.pad(state,((1,1),(1,1),(1,1)))
    
    # loaction = tuple(x,y,z)
    size_diff = state.shape[0] - 1
    
    
    pad_x_left = (location[0])
    pad_x = (pad_x_left, size_diff-pad_x_left)
    
    pad_y_left = (location[1])
    pad_y = (pad_y_left, size_diff-pad_y_left)
    
    pad_z_left = (location[2])
    pad_z = (pad_z_left, size_diff-pad_z_left)
    
    padded_mask = np.pad(mask,(pad_x,pad_y,pad_z))
    
    selection = state_padded * padded_mask
    
    return np.trim_zeros(selection)


def select_neighbors2(location):
    
    state_padded = np.pad(state,((1,1),(1,1),(1,1)))
    
    lx,ly,lz = location
    
    selection = state_padded[lx:lx+3,ly:ly+3,lz:lz+3] * mask
    
    return np.sum(np.trim_zeros(selection))

def select_neighbors3(location):
    
    lx,ly,lz = location
    
    selection = state_padded[lx:lx+3,ly:ly+3,lz:lz+3] * mask
    
    return np.sum(np.trim_zeros(selection))

def select_neighbors4(location):
    
    lx,ly,lz = location
    
    selection = state_padded[lx:lx+3,ly:ly+3,lz:lz+3] * mask
    
    return selection.flatten()

def time_test(n=10000):
    
    
    start = time.time()
    for _ in range(n):
        a = select_neighbors3((0,0,0))
    end = time.time()
    time_1 = (end-start)/n * 1000
    
    start = time.time()
    for _ in range(n):
        a = select_neighbors2((0,0,0))
    end = time.time()
    time_2 = (end-start)/n * 1000
    
    start = time.time()
    for _ in range(n):
        a = select_neighbors4((0,0,0))
    end = time.time()
    time_3 = (end-start)/n * 1000
    
    print(f'time_avg_3 = {time_1}ms, time_avg_2 = {time_2}ms, time_avg_4 = {time_3}ms')