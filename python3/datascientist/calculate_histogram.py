import sys
import numpy as np
from skimage import feature

CELL_SIZE = 4
LBP_POINTS = 24
LBP_RADIUS = 3 

def get_histogram(image):
    lbp = feature.local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, 'uniform')

    bins = LBP_POINTS + 2
    histogram = np.zeros(shape = (image.shape[0] / CELL_SIZE,
                                  image.shape[1] / CELL_SIZE, bins),
                         dtype = np.int)

    for y in range(0, image.shape[0] - CELL_SIZE, CELL_SIZE):
        for x in range(0, image.shape[1] - CELL_SIZE, CELL_SIZE):

            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    histogram[y / CELL_SIZE,
                              x / CELL_SIZE,
                              int(lbp[y+dy, x+dx])] += 1
    return histogram


