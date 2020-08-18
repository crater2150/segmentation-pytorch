import numpy as np


def to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)
    else:
        return image.astype(np.uint8) * 255
