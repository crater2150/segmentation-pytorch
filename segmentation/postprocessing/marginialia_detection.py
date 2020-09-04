from typing import List

from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN

from segmentation.postprocessing.layout_analysis import BboxCluster, BaselineResult
import numpy as np


def cluster_1d(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def marginalia_detection(bboxs: List[BboxCluster], image, num_border_threshold=5, min_border_length=150):
    change = False
    left_borders = []
    right_borders = []

    for box in bboxs:
        left_borders.append((box.get_left_x(), 0))
        right_borders.append((box.get_right_x(), 0))
    l = DBSCAN(eps=10, min_samples=1).fit(np.array(left_borders))
    r = DBSCAN(eps=10, min_samples=1).fit(np.array(right_borders))
    from collections import defaultdict
    l_borders = defaultdict(list)
    r_borders = defaultdict(list)
    for ind, x in enumerate(bboxs):
        l_c = l.labels_[ind]
        r_c = r.labels_[ind]
        for t in x.baselines:
            l_borders[l_c].append((x.get_left_x() - 1, x.get_top_y()))
            l_borders[l_c].append((x.get_left_x() - 1, x.get_bottom_y()))
        for t in x.baselines:
            r_borders[r_c].append((x.get_right_x() + 1, x.get_top_y()))
            r_borders[r_c].append((x.get_right_x() + 1, x.get_bottom_y()))

        pass

    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]
    # img = Image.fromarray(image)
    img = image.convert('RGB')
    draw = ImageDraw.Draw(img)
    for ind, x in enumerate(bboxs):
        if x.bbox:
            draw.line(x.bbox + [x.bbox[0]], fill=colors[ind % len(colors)], width=3)
            draw.text((x.bbox[0]), "type:{}".format(x.baselines[0].cluster_type))
    for ind, x in enumerate(l_borders.keys()):
        draw.line(l_borders[x], fill=(0, 0, 0), width=3)
    for ind, x in enumerate(r_borders.keys()):
        draw.line(r_borders[x], fill=(0, 0, 0), width=3)
    from matplotlib import pyplot as plt
    from shapely.geometry import LineString
    for x in l_borders.keys():
        border = sorted(l_borders[x], key=lambda k: k[1])
        length = (border[-1][1] - border[0][1])
        if len(l_borders[x]) > num_border_threshold and length > min_border_length:

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
                        for ind2, point in enumerate(baseline.baseline):
                            if baseline.baseline[-1][0] > x:
                                if point[0] > x:
                                    baselines.append(BaselineResult(baseline.baseline[:ind2 - 1], height=baseline.height,
                                                                    font_width=baseline.font_width,
                                                                    cluster_type=baseline.cluster_type,
                                                                    cluster_location=baseline.cluster_location))
                                    baselines.append(BaselineResult(baseline.baseline[ind2:], height=baseline.height,
                                                                    font_width=baseline.font_width,
                                                                    cluster_type=baseline.cluster_type,
                                                                    cluster_location=baseline.cluster_location))
                                    break

                            else:
                                baselines.append(baseline)
                    bboxs[ind].set_baselines(baselines)


    plt.imshow(np.array(img))
    plt.show()
    return bboxs
    pass
