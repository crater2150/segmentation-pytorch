import cv2 as cv
import numpy as np


def gauss_threshold(image: np.ndarray, block_size: int = 35, offset: int = 40):
    binary = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, block_size, offset)
    return binary