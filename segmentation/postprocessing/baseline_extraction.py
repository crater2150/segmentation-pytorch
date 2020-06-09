import numpy as np


def extraxct_baselines_from_probability_map(image_map: np.array, base_line_index=1, base_line_border_index=2,
                                            original=None):
    image = np.argmax(image_map, axis=-1)
    return extract_baselines(image_map=image, base_line_index=base_line_index,
                             base_line_border_index=base_line_border_index, original=original)


def extract_baselines(image_map: np.array, base_line_index=1, base_line_border_index=2, original=None):
    # from skimage import measure
    from scipy.ndimage.measurements import label

    base_ind = np.where(image_map == base_line_index)
    base_border_ind = np.where(image_map == base_line_border_index)

    baseline = np.zeros(image_map.shape)
    baseline_border = np.zeros(image_map.shape)
    baseline[base_ind] = 1
    baseline_border[base_border_ind] = 1
    baseline_ccs, n_baseline_ccs = label(baseline, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    baseline_ccs = [np.where(baseline_ccs == x) for x in range(1, n_baseline_ccs + 1)]
    baseline_border_ccs, n_baseline_border_ccs = label(baseline_border, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    baseline_border_ccs = [np.where(baseline_border_ccs == x) for x in range(1, n_baseline_border_ccs + 1)]

    class Cc_with_type(object):
        def __init__(self, cc, type):
            self.cc = cc
            index_min = np.where(cc[1] == min(cc[1]))  # [0]
            index_max = np.where(cc[1] == max(cc[1]))  # [0]

            if type == 'baseline':
                self.cc_left = (np.mean(cc[0][index_min][0]), cc[1][index_min[0]][0])
                self.cc_right = (np.mean(cc[0][index_max][0]), cc[1][index_max[0]][0])

            else:
                self.cc_left = (np.mean(cc[0]), cc[1][index_min[0]][0])
                self.cc_right = (np.mean(cc[0]), cc[1][index_max[0]][0])

            self.type = type

        def __lt__(self, other):
            return self.cc < other

    baseline_ccs = [Cc_with_type(x, 'baseline') for x in baseline_ccs if len(x[0]) > 10]

    baseline_border_ccs = [Cc_with_type(x, 'baseline_border') for x in baseline_border_ccs if len(x[0]) > 10]

    all_ccs = baseline_ccs + baseline_border_ccs
    from segmentation.util import pairwise, angle_between, angle_to

    def calculate_distance_matrix(ccs, length=50):
        distance_matrix = np.zeros((len(ccs), len(ccs)))
        vertical_distance = 10
        for ind1, x in enumerate(ccs):
            for ind2, y in enumerate(ccs):
                if x is y:
                    distance_matrix[ind1, ind2] = 0
                else:
                    distance = 0
                    same_type = 1 if x.type == y.type else 1000

                    def left(x, y):
                        return x.cc_left[1] > y.cc_right[1]

                    def right(x, y):
                        return x.cc_right[1] < y.cc_left[1]

                    ANGLE = 10
                    if left(x, y):
                        angle = angle_to(np.array(y.cc_right), np.array(x.cc_left))
                        distance = x.cc_left[1] - y.cc_right[1]
                        test_angle = ANGLE if distance > 30 else ANGLE * 3 if distance > 5 else ANGLE * 5
                        if test_angle < angle < (360 - test_angle):
                            distance = 99999
                        else:
                            point_c = y.cc_right
                            point_n = x.cc_left

                            x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                            y_points = np.interp(x_points, [point_c[1], point_n[1]], [point_c[0], point_n[0]]).astype(
                                int)
                            indexes = (y_points, x_points)

                            blackness = np.sum(baseline_border[indexes])
                            # print('left' + str(blackness))
                            distance = distance * (blackness * 5000 + 1)

                    elif right(x, y):
                        angle = angle_to(np.array(x.cc_right), np.array(y.cc_left))
                        distance = y.cc_left[1] - x.cc_right[1]
                        test_angle = ANGLE if distance > 30 else ANGLE * 3 if distance > 5 else ANGLE * 5

                        if test_angle < angle < (360 - test_angle):
                            distance = 99999
                        else:
                            distance = y.cc_left[1] - x.cc_right[1]

                            point_c = x.cc_right
                            point_n = y.cc_left

                            x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                            y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                                 [point_c[0], point_n[0]]).astype(
                                int)
                            indexes = (y_points, x_points)

                            blackness = np.sum(baseline_border[indexes])
                            distance = distance * (blackness * 5000 + 1)
                    else:
                        distance = 99999

                    distance_matrix[ind1, ind2] = distance * same_type
        return distance_matrix

    matrix = calculate_distance_matrix(all_ccs)

    from sklearn.cluster import DBSCAN
    if np.sum(matrix) == 0:
        print("Empty Image")
        return
    t = DBSCAN(eps=100, min_samples=1, metric="precomputed").fit(matrix)
    debug_image = np.zeros(image_map.shape)
    for ind, x in enumerate(all_ccs):
        debug_image[x.cc] = t.labels_[ind]

    ccs = []
    for x in np.unique(t.labels_):
        ind = np.where(t.labels_ == x)
        line = []
        for d in ind[0]:
            if all_ccs[d].type == 'baseline':
                line.append(all_ccs[d])
        if len(line) > 0:
            ccs.append((np.concatenate([x.cc[0] for x in line]), np.concatenate([x.cc[1] for x in line])))

    ccs = [list(zip(x[0], x[1])) for x in ccs]

    from itertools import chain
    from typing import List, Tuple
    from collections import defaultdict
    def normalize_connected_components(cc_list: List[List[Tuple[int, int]]]):
        # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            normalized_cc_list = []
            for cc in point_list:
                cc_dict = defaultdict(list)
                for y, x in cc:
                    cc_dict[x].append(y)
                normalized_cc = []
                # for key, value in cc_dict.items():
                for key in sorted(cc_dict.keys()):
                    value = cc_dict[key]
                    normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
                normalized_cc_list.append(normalized_cc)
            return normalized_cc_list

        return normalize(cc_list)

    ccs = normalize_connected_components(ccs)
    new_ccs = []
    for baseline in ccs:
        new_ccs.append([coord_tup[::-1] for coord_tup in baseline])

    return new_ccs
