import numpy as np


def scale_baseline(baseline, scale_factor: float = 1):
    if scale_factor == 1 or scale_factor == 1.0:
        return baseline

    return [(int(c[0] * scale_factor), int(c[1] * scale_factor)) for c in baseline]


def make_baseline_continous(bl):
    bl = sorted(bl, key=lambda x: x[0])
    # when downscaling the baseline might not be continuous
    # this method ensures, that we have a  mapping for each x-coordinate to a y-coordinate for a baseline bl
    xstart, xend = bl[0][0], bl[-1][0]
    new_xs = np.arange(xstart, xend + 1)
    xs, ys = zip(*bl)
    new_ys = (np.interp(new_xs, xs, ys) + 0.5).astype(int)
    nl = list(zip(new_xs.tolist(), new_ys.tolist()))
    return nl


def simplify_baseline(bl):
    new_coords = [bl[0]]

    for before, cur, after in zip(bl, bl[1:-1], bl[2:]):
        if before[1] == cur[1] and cur[1] == after[1]:
            continue
        else:
            new_coords.append(cur)
    new_coords.append(bl[-1])
    return new_coords