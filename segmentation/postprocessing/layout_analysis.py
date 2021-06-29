import itertools
import math
import multiprocessing
from collections.abc import Collection, Iterable
from functools import partial
from typing import List, Callable
from PIL import Image, ImageDraw

import numpy as np
from sklearn.cluster import DBSCAN

from segmentation.postprocessing.data_classes import BboxCluster, BaselineResult, MovedBaselineTop
from segmentation.postprocessing.baselines_util import make_baseline_continous
'''
Todo: Refactor file
'''


class BaselineCollection(Collection):
    def __init__(self, baselines=[]):
        self.baselines = baselines

    def __iter__(self):
        return iter(self.baselines)

    def __len__(self):
        return len(self.baselines)

    def __contains__(self, bl):
        return bl in self.baselines  # TODO: maybe compare by points, not by reference?


class Baseline(Iterable):
    def __init__(self, points):
        self.points = points

    def unzip(self):
        return zip(*self.points)

    def as_array(self):
        return np.array(self.points)

    def __iter__(self):
        return iter(self.points)

    def continuous(self) -> 'Baseline':
        return Baseline(make_baseline_continous(self.points))


class BaselinePrediction:
    def __init__(self, baselines: BaselineCollection, upper_baselines: BaselineCollection = BaselineCollection([])):
        self.baselines = baselines
        self.upper_baselines = upper_baselines

    def to_json(self):
        pass


def is_below(b1: BboxCluster, b2: BboxCluster, gap_padding_factor=0.5):
    """
    Checks if b2 is above b1
    """
    height = b2.get_average_height()
    return _is_in_direction(b1.get_bottom_line_of_bbox(), b2.get_top_line_of_bbox(),
                            lambda b2y1, b1y1: b2y1 < b1y1 + gap_padding_factor * height)


def is_above(b1: BboxCluster, b2: BboxCluster, gap_padding_factor=0.5):
    """
    Checks if b2 is below b1
    """
    height = b2.get_average_height()
    return _is_in_direction(b1.get_bottom_line_of_bbox(), b2.get_top_line_of_bbox(),
                            lambda b2y1, b1y1: b2y1 + gap_padding_factor * height > b1y1)


def _is_in_direction(b1line: BboxCluster, b2line: BboxCluster, compare: Callable[[float, float], bool]):
    (b1x1, b1y1), (b1x2, b1y2) = b1line
    (b2x1, b2y1), (b2x2, b2y2) = b2line
    if b2x1 <= b1x1 <= b2x2 or b2x1 <= b1x2 <= b2x2 or (b2x1 >= b1x1 and b2x2 <= b1x2) or (
            b2x1 <= b1x1 and b2x2 >= b1x2):
        if compare(b2y1, b1y1):  # (0,0) is top left
            return True

    return False


def is_between(b1: BboxCluster, b2: BboxCluster, b3: BboxCluster, gap_padding_factor=0.5):
    """
    checks if b1 is between b2 and b3
    """
    return (is_below(b1, b2, gap_padding_factor) and is_above(b1, b3, gap_padding_factor)) or \
           (is_below(b1, b3, gap_padding_factor) and is_above(b1, b2, gap_padding_factor))


def get_bboxs_between(bbox: BboxCluster, bbox2: BboxCluster, bbox_cluster: List[BboxCluster]):
    return [x for x in _different_bboxes(bbox, bbox_cluster)
            if is_between(x, bbox, bbox2)]


def _different_bboxes(bbox: BboxCluster, bbox_cluster: List[BboxCluster]):
    for x in bbox_cluster:
        # TODO: is this really what we want, or should the chain.from_iterable be removed?
        if set(itertools.chain.from_iterable(x.bbox)) != set(itertools.chain.from_iterable(bbox.bbox)):
            yield x


def get_bboxs_above(bbox: BboxCluster, bbox_cluster: List[BboxCluster]):
    result = []
    for x in _different_bboxes(bbox, bbox_cluster):
        if is_above(x, bbox):
            difference = bbox.get_top_line_of_bbox()[0][1] - x.get_bottom_line_of_bbox()[1][1]
            height = bbox.get_average_height()
            if difference <= height:
                result.append(x)

    # only return b_boxes which are directly above a specific bbox
    # thus filtering b_boxes which doesnt fulfill this requirement
    # is needed when pages are not deskewed
    return [x for x in result
            if all(not is_above(x, z) for z in result)]


def get_bboxs_below(bbox: BboxCluster, bbox_cluster: List[BboxCluster]):
    result = []
    for x in _different_bboxes(bbox, bbox_cluster):
        if is_below(x, bbox):
            difference = x.get_top_line_of_bbox()[1][1] - bbox.get_bottom_line_of_bbox()[0][1]
            height = bbox.get_average_height()
            if difference <= height:
                result.append(x)

    # only return b_boxes which are directly above a specific bbox
    # thus filtering b_boxes which doesnt fulfill this requirement
    # is needed when pages are not deskewed
    return [x for x in result
            if all(not is_below(x, z) for z in result)]


def analyse(baselines, image, use_improved_tops=True):
    result = []
    heights = []
    length = []
    if baselines is None:
        return
    if use_improved_tops:
        improved_tops = get_top_of_baselines_improved(baselines, image)
    else:
        improved_tops = get_top_of_baselines(baselines, image)

    for vec in improved_tops:
        result.append(vec)
        heights.append(vec[2])
        length.append(vec[0][-1][1])

    img = Image.fromarray((1 - image) * 255).convert('RGB')
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    from segmentation.postprocessing.util import baseline_to_bbox, crop_image_by_polygon
    result_dict = {}
    for ind, x in enumerate(result):
        pol = baseline_to_bbox(x[0],
                               margin_top=x[2],
                               margin_bot=0,
                               margin_left=0,
                               margin_right=0)
        cut = crop_image_by_polygon(polygon=pol, image=image)
        from segmentation.postprocessing.util import get_stroke_width
        if cut[0] is not None:
            score1, _ = get_stroke_width(cut[0])
            baseline = x[0]
            p1 = baseline[0]
            p2 = baseline[-1]
            vector = [x[2] / np.max(heights), score1]
            vector2 = [p1[0] / max(length), p2[0] / max(length)]
            result_dict[ind] = [baseline, score1, x[2], vector, vector2]
            draw.line(list(itertools.chain.from_iterable(baseline)), fill=colors[ind % len(colors)], width=2)

    if len(result_dict) == 0: return []

    inds = result_dict.keys()
    vectors = [result_dict[indice][3] for indice in inds]
    h, s = zip(*vectors)
    vectors = [[h_s, 0] for h_s, s_s in list(zip(h, s))]
    vectors2 = [result_dict[indice][4] for indice in inds]
    t = DBSCAN(eps=0.3, min_samples=1).fit(np.array(vectors))
    e = DBSCAN(eps=0.03, min_samples=1).fit(np.array(vectors2))

    cluster_results = []
    for ind, x in enumerate(inds):
        meta = result_dict[x]
        pol = baseline_to_bbox(meta[0],
                               margin_top=meta[2],
                               margin_bot=0,
                               margin_left=0,
                               margin_right=0)
        # draw.polygon(pol, outline=colors[ind % len(colors)])
        draw.text((pol[0]), "w:{},h:{},l:{} l:{}".format(round(meta[1], 3), meta[2], t.labels_[ind],
                                                         e.labels_[ind]),
                  fill=(14, 183, 242))  # ), font=ImageFont.truetype("font_path123"))
        cluster_results.append(BaselineResult(baseline=meta[0],
                                              height=meta[2],
                                              font_width=meta[1],
                                              cluster_type=t.labels_[ind],
                                              cluster_location=e.labels_[ind]))
    cluster_results = [x for x in cluster_results if x.height > 5]
    clusterd = generate_clustered_lines(cluster_results)
    bboxes = generate_bounding_box_cluster(clustered=clusterd)
    return bboxes


def connect_bounding_box(bboxes: [List[BboxCluster]]):
    bboxes_clone = bboxes.copy()
    clusters = []
    cluster = []

    def alpha_shape_from_list_of_bboxes(clusters):
        def merge_points_to_box(point_list):
            if len(point_list) == 1:
                return point_list[0]

            list1 = []
            list2 = []
            for x in point_list:
                x = sorted(x, key=lambda k: (k[0], k[1]))
                list1.append(x[:2])
                list2.append(list(reversed(x[2:])))
            array = list(itertools.chain.from_iterable(list1 + list(reversed(list2))))
            return array

        bboxes = []
        for item in clusters:
            item = sorted(item, key=lambda k: k.bbox[0][1])
            y = [x.bbox for x in item]
            # points = list(itertools.chain.from_iterable([x.bbox for x in item]))
            array = merge_points_to_box(point_list=y)

            baselines = []
            for x in item:
                baselines = baselines + x.baselines
            bboxes.append(BboxCluster(baselines=baselines, bbox=array))
        return bboxes

    skip = False
    while len(bboxes_clone) != 0:
        for ind, x in reversed(list(enumerate(bboxes_clone))):
            if len(cluster) == 0:
                cluster.append(x)
                del bboxes_clone[ind]
                break
            b1p1, b1p2 = cluster[-1].get_bottom_line_of_bbox()
            b2p1, p2p2 = x.get_top_line_of_bbox()
            b1x1, b1y1 = b1p1
            b1x2, b1y2 = b1p2
            b2x1, b2y1 = b2p1
            b2x2, b2y2 = p2p2

            height = min(cluster[-1].baselines[0].height, x.baselines[0].height)
            type1 = cluster[-1].baselines[0].cluster_type
            type2 = x.baselines[0].cluster_type
            if len(get_bboxs_above(x, bboxes)) > 1 or len(get_bboxs_below(cluster[-1], bboxes)) > 1:
                clusters.append(cluster)
                cluster = []
                break

            if type1 == type2:
                if is_above(cluster[-1], x) and (abs(b1x1 - b2x1) < 150 or abs(b1x2 - b2x2) < 150):  ##check between
                    if len(get_bboxs_between(x, cluster[-1], bboxes)) == 0:
                        if abs(b2y1 - b1y1) < height * 1.5:
                            box = None
                            pointer = 1
                            while True:
                                if ind - pointer >= 0:
                                    if is_above(bboxes_clone[ind - pointer], cluster[-1]):
                                        box = bboxes_clone[ind - pointer]
                                        break
                                else:
                                    break
                                if pointer > 5:
                                    break
                                pointer = pointer + 1
                            if box is not None:
                                b3p1, b3p2 = box.get_bottom_line_of_bbox()
                                b4p1, b4p2 = bboxes_clone[ind].get_bottom_line_of_bbox()

                                b4x1, b4y1 = b4p2
                                b3x2, b3y2 = b3p2
                                type3 = box.baselines[0].cluster_type
                                if type3 == type2 and abs(b4y1 - b3y2) < height:
                                    clusters.append(cluster)
                                    cluster = []

                            cluster.append(x)
                            del bboxes_clone[ind]
                            break
            if ind == 0:
                clusters.append(cluster)
                cluster = []
                break
        if len(bboxes_clone) == 0:
            clusters.append(cluster)
        pass

    return alpha_shape_from_list_of_bboxes(clusters)


def generate_bounding_box_cluster(clustered: List[List[BaselineResult]]):
    boxes = []

    def get_border(cluster: List[BaselineResult]):

        x_min = math.inf
        x_max = 0
        y_min = math.inf
        y_max = 0
        height = 0
        for item in cluster:
            before = x_min
            x, y = list(zip(*item.baseline))
            x = list(x)
            y = list(y)
            x_min = min(x + [x_min])
            y_min = min(y + [y_min])
            x_max = max(x + [x_max])
            y_max = max(y + [y_max])
            if before != x_min:
                height = item.height
        return BboxCluster(baselines=cluster,
                           bbox=[(x_min, y_min - height), (x_max, y_min - height), (x_max, y_max), (x_min, y_max)])

    for t in clustered:
        boxes.append(get_border(t))
    return boxes


def generate_clustered_lines(cluster_results: List[BaselineResult]):
    clone = sorted(cluster_results, key=lambda t: t.baseline[0][1])
    clustered = []
    cluster = []
    while len(clone) > 0:
        for ind, x in enumerate(reversed(clone)):

            if len(cluster) == 0:
                cluster.append(x)
                del clone[len(clone) - 1 - ind]
                break
            start_x = cluster[-1].baseline[0][0]
            end_x = cluster[-1].baseline[-1][0]
            start_x_2 = x.baseline[0][0]
            end_x_2 = x.baseline[-1][0]
            if abs(cluster[-1].baseline[0][1] - x.baseline[0][1] > cluster[-1].height * 3):
                clustered.append(cluster)
                cluster = []
                break
            if cluster[-1].cluster_type == x.cluster_type and \
                    cluster[-1].cluster_location == x.cluster_location:
                cluster.append(x)
                del clone[len(clone) - 1 - ind]
                break
            if (start_x <= start_x_2 and end_x >= end_x_2) \
                    or (start_x_2 <= start_x <= end_x_2) \
                    or (start_x_2 <= end_x <= end_x_2):
                clustered.append(cluster)
                cluster = []
                break
            if ind == len(clone) - 1:
                clustered.append(cluster)
                cluster = []
                break
        if len(clone) == 0:
            clustered.append(cluster)
    return clustered


def get_top_wrapper(baseline, image=None, threshold=0.2):
    top_border, height = get_top(image, baseline, threshold=threshold)
    return MovedBaselineTop(baseline, top_border, height)


def get_top(image, baseline, threshold=0.2, disable_now=False, max_steps=None):
    if max_steps is None:
        max_steps = int(image.shape[0])
    x, y = zip(*baseline)
    indexes = (np.array(y), np.array(x))
    max_black_pixels = 0
    height = 0
    for i in range(min(np.min(indexes[0]), max_steps)):  # do at most min_height steps, so that min(y) == 0
        indexes = (indexes[0] - 1, indexes[1])
        if np.min(indexes[0]) == 0:
            height = height + 1
            return list(zip(indexes[1], indexes[0])), height  # we have to stop right now
        now = np.sum(image[indexes])
        if (max_black_pixels * threshold > now or (now <= 5 and not disable_now)) and height > 5:
            break
        height = height + 1
        max_black_pixels = now if now > max_black_pixels else max_black_pixels
    return list(zip(indexes[1].tolist(), indexes[0].tolist())), height


def get_top_of_baselines(baselines, image=None, threshold=0.2, process_pool: multiprocessing.Pool = None) -> List[
MovedBaselineTop]:
    p_get_top = partial(get_top_wrapper, image=image, threshold=threshold)
    if process_pool:
        out = list(process_pool.map(p_get_top, baselines))
    else:
        out = list(map(p_get_top, baselines))

    return out


# def match_baselines_by_starting_point

def get_top_of_baselines_improved(baselines, image: np.ndarray = None, threshold=0.2,
                                  process_pool: multiprocessing.Pool = None):
    # check for each bl if its continous
    for bl in baselines:
        assert len(bl) == (bl[-1][0] - bl[0][0] + 1), "Baseline must be continuous for get_top_of_baselines"
    baseline_ys = [list(zip(*bl))[1] for bl in baselines]
    DIST_INF = 1000000
    HEIGHT_MULTP = 1.5
    HEIGHT_MULTP_MATCHING = 3  # this should be calculated as "avg line spacing"

    # for each baseline, match a baseline that is above
    # this matching is horribly inefficient and can be improved significantly
    matches = []
    for bl_id, bl in enumerate(baselines):
        best_dist = DIST_INF
        for match_id, match_cand in enumerate(baselines):
            if bl_id == match_id: continue
            # the beginning and end coordinates are not allowed to deviate a lot
            if abs(bl[0][0] - match_cand[0][0]) > image.shape[1] * 0.05: continue

            # right side must not be much shorter, but can be much longer
            if bl[-1][0] > match_cand[-1][0]:
                if abs(bl[-1][0] - match_cand[-1][0]) > image.shape[1] * 0.05: continue

            # remove points in the match candidate, that are further to the right than our baseline
            bl_mx = bl[-1][0]
            # trunc_match_cand_ys = [c[1] for c in match_cand if c[0] <= bl_mx]

            # do fast truncation
            start_diff = max(bl[0][0] - match_cand[0][0], 0)
            trunc_match_cand_ys = baseline_ys[match_id][:len(bl) + start_diff]

            # calculate avg height
            bl_h = sum(c[1] for c in bl) / len(bl)
            # mh = sum(c[1] for c in match_cand if c[0] <= bl[-1][0]) / len (match_cand) # do not do this
            # only take the avg height until the short bl ends
            mh = sum(trunc_match_cand_ys) / len(trunc_match_cand_ys)
            if mh > bl_h: continue  # match candidate lies below
            # score is height difference
            dist = abs(mh - bl_h)
            if dist < best_dist:
                best_dist = dist

        matches.append((bl, best_dist, match_id))

    out = []
    for match in matches:
        bl, tb, height = get_top_wrapper(match[0], image, threshold=threshold)
        if match[1] > height * HEIGHT_MULTP_MATCHING or match[1] == DIST_INF:
            perfect_height_max = height * HEIGHT_MULTP_MATCHING
            # without max steps, this can take way too long for broken inputs
            _, perfect_height = get_top(image, match[0], threshold=0.001, disable_now=True,
                                        max_steps=int(perfect_height_max + 1))
            if perfect_height < perfect_height_max:
                height = perfect_height
            else:
                height = int(height * HEIGHT_MULTP)
        else:
            height = int(min(match[1] - 1, (height * HEIGHT_MULTP + match[1]) / 2))
            """
            # we still have room, increase the height
            if height * 1.5 > match[1]:
                height = int(match[1]) - 1
            else:
                height = int(height * 1.5)
            """

        # prevent growing lines out of the image
        heighest = min(c[1] for c in bl)
        height = min(height, heighest - 1)

        out.append(MovedBaselineTop(bl, [(b[0], b[1] - height) for b in bl], height))

    return out
