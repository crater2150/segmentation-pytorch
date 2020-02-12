import numpy as np
import itertools


def angle_to(p1, p2):
    p2 = p2 - p1
    p1 = p1 - p1
    ang1 = np.arctan2(*p1[::])
    ang2 = np.arctan2(*p2[::])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::])
    ang2 = np.arctan2(*p2[::])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def angle_between3(p1, p2):
    ang1 = np.arctan2(*p1)
    ang2 = np.arctan2(*p2)
    angle_difference = (ang1 - ang2) % (np.pi)
    angle_difference if angle_difference < np.pi else -2 * np.pi + angle_difference
    return np.rad2deg(angle_difference)


def angle_between2(p1, p2):
    import numpy as np
    from numpy.linalg import norm
    v1 = np.array(p1[::-1])
    v2 = np.array(p2[::-1])

    angle_difference = np.arccos((v1 @ v2) / (norm(v1) * norm(v2)))
    return np.rad2deg(angle_difference)


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def gray_to_rgb(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = img[..., np.newaxis]
        return np.concatenate(3 * (img,), axis=-1)
    else:
        return img
