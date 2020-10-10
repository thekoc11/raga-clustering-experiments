import numba
import numpy as np

def gated_selection(chroma):
    shape = chroma.shape
    new_chroma = []
    for i in range(shape[0]):
        time_chroma = chroma[i]
        max = np.max(time_chroma)
        time_chroma = select_max(np.ascontiguousarray(time_chroma), max)
        new_chroma.append(time_chroma)
    return np.array(new_chroma)



@numba.vectorize(['float64(float64, float64)'], target='cuda')
def select_max(array, max):
    if(array < max):
        array = 0
    return array


if __name__ == '__main__':
    chroma = np.random.uniform(0.0, 1.0, (13, 20))
    print(np.ascontiguousarray(chroma).T)
    print(gated_selection(chroma.T))