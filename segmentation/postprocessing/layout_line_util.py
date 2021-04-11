import itertools
from typing import List, Tuple

import numpy as np
from numba import jit


def _build_bl_growth_img(binarized: np.ndarray, points: np.ndarray):
    print(points)
    """
    Build data format to quickly get the position of the next baseline above
    
    Args:
        binarized: Binarized image 
        bl: numpy array containing the baselines with axes (0 = baselines, 1 = points, 2 = xy coords)

    Returns:
        Image, where each pixels tells the distance to the next baseline above (or +MAX_INT) if none
    """
    points = np.transpose(points)
    # transpose the binarized image for more efficient memory access
    bt = np.transpose(binarized)
    # get all baseline points
    # transpose to y,x coordinates
    # sort the points by descending y-axis coords
    points_srtd = np.flip(np.sort(points,axis=0),axis=0) # sorted by y-coordinate high to low
    heights = np.zeros(bt.shape,dtype=np.int32)
    heights[points] = -1
    bin_h = bt.shape[0]
    for pp in points:
        x = pp[0]
        y = pp[1]
        x += 1
        count = np.int32(1)
        while heights[y,x] != -1 and x < bin_h:
            heights[y,x] = count
            #x += 1
            count += 1
    for pp in points:
        x = pp[0]
        y = pp[1]
        if x == 0:
            heights[y,x] = 0
        else:
            heights[y,x] = heights[y,x-1]
    return np.transpose(heights)


class BaselineLookupByPoint:
    def __init__(self, bl: List[List[Tuple[int,int]]]):
        self.baselines = bl
        pos_lookup = dict()
        for i, line in enumerate(bl):
            for p in line:
                pos_lookup[(p[0],p[1])] = i
        self.pos_lookup = pos_lookup

    def get_baseline(self, x,y) -> Tuple[int, List[Tuple[int,int]]]:
        pos = self.pos_lookup.get((x,y))
        if pos:
            return pos, self.baselines[pos]
        else:
            return (None, None)

class BaselineGrowthAcc:

    def __init__(self, binarized: np.ndarray, bl: List[Tuple[int,int]]):
        # get the longest baseline
        pass


