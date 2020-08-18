from typing import List, NamedTuple, Tuple, Any

from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN

import numpy as np

from segmentation.postprocessing.data_classes import BboxCluster, BaselineResult


def cluster_1d(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]


def marginalia_detection(bboxs: List[BboxCluster], image, num_border_threshold=5, min_border_length=150,
                         max_gap_length=15, min_segment_length=10, number_of_minimum_baselines_to_count_as_border=2,
                         number_of_minimum_baselines_to_count_as_border2=10, connected_baseline_of_a_border=5):
    change = False
    left_borders = []
    right_borders = []
    index = 0
    for ind, box in enumerate(bboxs):
        for i in box.baselines:
            left_borders.append((box.get_left_x(), index))
            right_borders.append((box.get_right_x(), index))
            index = index + 1
    l = DBSCAN(eps=15, min_samples=1).fit(np.array(left_borders))
    r = DBSCAN(eps=15, min_samples=1).fit(np.array(right_borders))
    from collections import defaultdict
    l_borders = defaultdict(list)
    r_borders = defaultdict(list)

    class Border(NamedTuple):
        border: List[Tuple[Any, Any]]
        bbox_id: int
    index = 0
    for ind, x in enumerate(bboxs):
        l_c = l.labels_[index]
        r_c = r.labels_[index]
        for _ in x.baselines:
            l_borders[l_c].append(Border(border=(x.get_left_x() - 1, x.get_top_y()), bbox_id=ind))
            l_borders[l_c].append(Border(border=(x.get_left_x() - 1, x.get_bottom_y()), bbox_id=ind))
        for _ in x.baselines:
            r_borders[r_c].append(Border(border=(x.get_right_x() + 1, x.get_top_y()), bbox_id=ind))
            r_borders[r_c].append(Border(border=(x.get_right_x() + 1, x.get_bottom_y()), bbox_id=ind))
        for _ in x.baselines:
            index = index + 1

    indices = 0
    border_dict = defaultdict(list)
    for x in l_borders.keys():
        l_med = np.median([l.border[0] for l in l_borders[x]])
        for y in r_borders.keys():
            r_med = np.median([r.border[0] for r in r_borders[y]])

            med = abs(l_med - r_med)
            # mean = abs(l_mean - r_mean)
            if med <= max_gap_length:
                l_border = [x.border for x in l_borders[x]]
                r_border = [x.border for x in r_borders[y]]
                l_id = [x.bbox_id for x in l_borders[x]]
                r_id = [x.bbox_id for x in r_borders[y]]

                min_y_l = min(l_border, key=lambda k: k[1])[1]
                max_y_l = max(l_border, key=lambda k: k[1])[1]
                min_y_r = min(r_border, key=lambda k: k[1])[1]
                max_y_r = max(r_border, key=lambda k: k[1])[1]

                if not (max_y_l < min_y_r or max_y_r < min_y_l) and not bool(set(l_id) & set(r_id)):
                    if min(len(l_border), len(r_border)) >= number_of_minimum_baselines_to_count_as_border and \
                            max(len(l_border), len(r_border)) >= number_of_minimum_baselines_to_count_as_border2 * 2:
                        #print(max(max([r_id.count(x) for x in set(r_id)]), max([l_id.count(x) for x in set(l_id)])))
                        if max(max([r_id.count(x) for x in set(r_id)]), max([l_id.count(x) for x in set(l_id)])) \
                                >= connected_baseline_of_a_border * 2:
                            l_med_b = [(l[0], l[1]) for l in l_border]
                            r_med_b = [(r[0], r[1]) for r in r_border]
                            b = sorted(l_med_b + r_med_b, key=lambda k: k[1])
                            from segmentation.postprocessing.simplify_line import VWSimplifier
                            simplifier = VWSimplifier(np.asarray(b, dtype=np.float64))
                            border_line = simplifier.from_number(2)
                            border_line = border_line.tolist()
                            border_dict[indices] = [[(b[0], b[1]) for b in border_line], x, y]
                            indices = indices + 1
    '''
    indices = 0
    updated_border_lines = {}
    for border_key in border_dict.keys():
        border, x, y = border_dict[border_key]
        l_id = set([x.bbox_id for x in l_borders[x]])
        r_id = set([x.bbox_id for x in r_borders[y]])

        longer_site = l_id if len(l_id) > len(r_id) else r_id
        b_lines = [(bboxs[id].baselines, id) for id in longer_site]
        avg_baseline_height = np.mean([bl_res.height for bl, id in b_lines for bl_res in bl])
        baseline_y = sorted([(bl_res.get_avg_y(), id) for bl, id in b_lines for bl_res in bl],
                            key=lambda k: k[0])
        gaps = []
        for i in range(len(baseline_y) - 1):
            current = baseline_y[i][0]
            after = baseline_y[i + 1][0]
            if after - current > avg_baseline_height * 8:
                gaps.append(baseline_y[i][0])
        print(gaps)
        if len(gaps) != 0:
            previous_gap = 0
            for gap in gaps:
                l_border_segment = [x for x in l_borders[x] if previous_gap < x.border[1] < gap]
                r_border_segment = [x for x in r_borders[y] if previous_gap < x.border[1] < gap]
                l_med = np.median([l.border[0] for l in l_border_segment])
                r_med = np.median([r.border[0] for r in r_border_segment])
                med = abs(l_med - r_med)

                if min(len(l_border_segment), len(r_border_segment)) > number_of_minimum_baselines_to_count_as_border * 2 and \
                        max(len(l_border_segment), len(r_border_segment)) > number_of_minimum_baselines_to_count_as_border2 * 2:
                    l_med_b = [(l.border[0] - med, l.border[1]) for l in l_border_segment]
                    r_med_b = [(r.border[0] + med, r.border[1]) for r in r_border_segment]
                    b = sorted(l_med_b + r_med_b, key=lambda k: k[1])
                    from segmentation.postprocessing.simplify_line import VWSimplifier
                    simplifier = VWSimplifier(np.asarray(b, dtype=np.float64))
                    border_line = simplifier.from_number(2)
                    border_line = border_line.tolist()
                    updated_border_lines[indices] = [[(b[0], b[1]) for b in border_line], x, y]
                    indices = indices + 1

        else:
            updated_border_lines[indices] = [border, x, y]
            indices = indices + 1
    border_dict = updated_border_lines
    '''
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]
    img = image.convert('RGB')
    draw = ImageDraw.Draw(img)
    for ind, x in enumerate(bboxs):
        if x.bbox:
            draw.line(x.bbox + [x.bbox[0]], fill=colors[ind % len(colors)], width=3)
            draw.text((x.bbox[0]), "type:{}".format(x.baselines[0].cluster_type))

    from shapely.geometry import LineString
    for ind, x in enumerate(border_dict.keys()):
        draw.line(border_dict[x][0], fill=(0, 255, 0), width=3)
    # for ind, x in enumerate(l_borders.keys()):
    #    draw.line(l_borders[x].border, fill=(0, 0, 0), width=3)
    # for ind, x in enumerate(r_borders.keys()):
    #    draw.line(r_borders[x].border, fill=(0, 0, 0), width=3)
    for x in border_dict.keys():
        if len(border_dict[x][0]) > 0:
            border = sorted(border_dict[x][0], key=lambda k: k[1])
            for ind, box in enumerate(bboxs):
                top_line = box.get_top_line_of_bbox()
                line1 = LineString(border)
                line2 = LineString(top_line)
                point = line1.intersection(line2)
                if point.geom_type != "LineString":
                    assert point.geom_type == "Point", "no point geom type"
                    x, y = point.xy
                    baselines: List[BaselineResult] = []
                    for ind_1, baseline in enumerate(box.baselines):
                        baseline: BaselineResult = baseline
                        if baseline.baseline[-1][0] > x:
                            for ind2, point in enumerate(baseline.baseline):
                                if point[0] > x:
                                    segment1 = baseline.baseline[:ind2 - 1]
                                    segment2 = baseline.baseline[ind2:]
                                    if len(segment1) > min_segment_length and len(segment2) > min_segment_length:
                                        baselines.append(
                                            BaselineResult(segment1, height=baseline.height,
                                                           font_width=baseline.font_width,
                                                           cluster_type=baseline.cluster_type,
                                                           cluster_location=baseline.cluster_location))
                                        baselines.append(
                                            BaselineResult(segment2, height=baseline.height,
                                                           font_width=baseline.font_width,
                                                           cluster_type=baseline.cluster_type,
                                                           cluster_location=baseline.cluster_location))

                                    else:
                                        baselines.append(baseline)
                                    break

                        else:
                            baselines.append(baseline)
                    bboxs[ind].set_baselines(baselines)
    #from matplotlib import pyplot as plt
    #plt.imshow(np.array(img))
    #plt.show()
    return bboxs
