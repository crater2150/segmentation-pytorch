from sklearn.cluster import DBSCAN
import numpy as np
from segmentation.util import pairwise


def estimate_borders(list_endpoints, max_offset=30):
    m = [[list_endpoints[0]]]
    for tup in list_endpoints[1:]:
        x = tup[0]
        if x - m[-1][0][0] < max_offset:
            m[-1].append(tup)
        else:
            m.append([tup])
    return m


def ret_baselines_by_id(baselines, basline_id_tuple):
    bl = []
    for x in basline_id_tuple:
        l = []
        for _, id in x:
            l.append(baselines[id])
        l.sort(key=lambda z: z[0][0])
        bl.append(l)
    return bl


def split_by_height(border_list):
    list123 = []
    for lines in border_list:
        line_split = []
        height = [np.average(z, axis=0)[0] for z in lines]
        median = min(int(np.median([y - x for x, y in pairwise(height)])), 40)
        start = True
        lines.append(lines[-1])
        height.append(height[-1])
        for prev, cur in pairwise(list(zip(lines, height))):

            prev_line, prev_height = prev
            cur_line, cur_height = cur
            line_split.append(prev_line)
            if cur_height - prev_height >= 3 * median:
                list123.append(line_split)
                line_split = []
            elif start and cur_height - prev_height >= 2 * median:
                list123.append(line_split)
                line_split = []
                start = False
        if len(line_split) > 0:
            list123.append(line_split)

    return list123


def text_border_estimation(baselines):
    left_border_list = []
    right_border_list = []
    for ind, x in enumerate(baselines):
        left_border_list.append((x[0][1], ind))
        right_border_list.append((x[-1][1], ind))

    left_border_list = sorted(left_border_list, key=lambda x: x[0])
    right_border_list = sorted(right_border_list, key=lambda x: x[0])
    left_border = estimate_borders(left_border_list)
    right_border = estimate_borders(right_border_list)

    right_border_ccs = ret_baselines_by_id(baselines, right_border)
    left_border_ccs = ret_baselines_by_id(baselines, left_border)
    right_border_ccs = split_by_height(right_border_ccs)
    left_border_ccs = split_by_height(left_border_ccs)

    left_border_ccs = [x for x in left_border_ccs if len(x) > 2]
    right_border_ccs = [x for x in right_border_ccs if len(x) > 2]

    return (left_border_ccs, right_border_ccs)

    # kt = DBSCAN(eps=100, min_samples=1, metric="euclidean").fit(left_border_list)
    # print(kt)
