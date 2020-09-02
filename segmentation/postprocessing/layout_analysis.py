import glob
import itertools
import math
from typing import NamedTuple, List, Tuple, Union
from collections import namedtuple
from matplotlib import pyplot
from skimage.morphology import skeletonize
from sklearn.cluster import dbscan
from segmentation.util import previous_and_next
from segmentation.network import Network
from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
from segmentation.settings import PredictorSettings
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont

'''
Todo: Refactor file
'''


class BaselineResult(NamedTuple):
    baseline: List
    height: int
    font_width: float
    cluster_type: int
    cluster_location: int

    def scale(self, scale_factor):
        baseline = [(x[0] * scale_factor, x[1] * scale_factor) for x in self.baseline]
        return BaselineResult(baseline=baseline,
                              height=self.height * scale_factor,
                              font_width=self.font_width,
                              cluster_location=self.cluster_location,
                              cluster_type=self.cluster_type
                              )


class BboxCluster(NamedTuple):
    baselines: List[BaselineResult]
    bbox: List[Tuple[any, any]]

    def scale(self, scale_factor):
        return BboxCluster(baselines=[x.scale(scale_factor) for x in self.baselines],
                           bbox=[(x[0] * scale_factor, x[1] * scale_factor) for x in self.bbox])

    def get_average_height(self):
        return np.mean([x.height for x in self.baselines])

    def get_char_cluster_type(self):
        return self.baselines[0].cluster_type

    def get_location_cluster_type(self):
        return self.baselines[0].cluster_location

    def number_of_baselines_in_cluster(self):
        return len(self.baselines)

    def get_bottom_line_of_bbox(self):
        bbox_sorted = sorted(self.bbox, key=lambda k: (k[1], k[0]))
        return bbox_sorted[-2], bbox_sorted[-1]

    def get_top_line_of_bbox(self):
        bbox_sorted = sorted(self.bbox, key=lambda k: (k[1], k[0]))

        return bbox_sorted[0], bbox_sorted[1]


def is_above(b1: BboxCluster, b2: BboxCluster, gap_padding_factor=0.5):
    b1p1, b1p2 = b1.get_top_line_of_bbox()
    b2p1, p2p2 = b2.get_bottom_line_of_bbox()
    b1x1, b1y1 = b1p1
    b1x2, b1y2 = b1p2
    b2x1, b2y1 = b2p1
    b2x2, b2y2 = p2p2
    height = b2.get_average_height()
    if b2x1 <= b1x1 <= b2x2 or b2x1 <= b1x2 <= b2x2 or (b2x1 >= b1x1 and b2x2 <= b1x2) or (
            b2x1 <= b1x1 and b2x2 >= b1x2):
        #print(b2y1)
        #print(b1y1)
        #print(b2y1 < b1y1)
        if b2y1 < b1y1 + gap_padding_factor * height: # (0,0) is top left
            return True

    return False

def is_below(b1: BboxCluster, b2: BboxCluster, gap_padding_factor=0.5):
    b1p1, b1p2 = b1.get_bottom_line_of_bbox()
    b2p1, p2p2 = b2.get_top_line_of_bbox()
    b1x1, b1y1 = b1p1
    b1x2, b1y2 = b1p2
    b2x1, b2y1 = b2p1
    b2x2, b2y2 = p2p2
    height = b2.get_average_height()
    if b2x1 <= b1x1 <= b2x2 or b2x1 <= b1x2 <= b2x2 or (b2x1 >= b1x1 and b2x2 <= b1x2) or (
            b2x1 <= b1x1 and b2x2 >= b1x2):
        #print(b2y1)
        #print(b1y1)
        #print(b2y1 < b1y1)
        if b2y1 + gap_padding_factor * height > b1y1: # (0,0) is top left
            return True

    return False


def get_bboxs_above(bbox: BboxCluster, bbox_cluster: List[BboxCluster], height_threshold=100):
    result = []
    for x in bbox_cluster:
        if set(sorted(list(itertools.chain.from_iterable(x.bbox)))) != set(
                sorted(list(itertools.chain.from_iterable(bbox.bbox)))):
            if is_above(bbox, x):
                b1p1, b1p2 = bbox.get_top_line_of_bbox()
                b2p1, p2p2 = x.get_bottom_line_of_bbox()
                b1x1, b1y1 = b1p1
                b2x2, b2y2 = p2p2
                height = bbox.get_average_height()
                difference = b1y1 - b2y2
                if difference <= height:
                    result.append(x)

    # only return b_boxes which are directly above a specific bbox
    # thus filtering b_boxes which doesnt fulfill this requirement
    # is needed when pages are not deskewed
    result_direct = []
    for x in result:
        direct_above = True
        for z in result:
            if is_above(x, z):
                direct_above = False
                break
        if direct_above:
            result_direct.append(x)
    return result_direct


def get_bboxs_below(bbox: BboxCluster, bbox_cluster: List[BboxCluster], height_threshold=100):
    result = []
    for x in bbox_cluster:
        if set(sorted(list(itertools.chain.from_iterable(x.bbox)))) != set(
                sorted(list(itertools.chain.from_iterable(bbox.bbox)))):
            if is_below(bbox, x):
                b1p1, b1p2 = bbox.get_bottom_line_of_bbox()
                b2p1, p2p2 = x.get_top_line_of_bbox()
                b1x1, b1y1 = b1p1
                b2x2, b2y2 = p2p2
                height = bbox.get_average_height()
                difference = b2y2 - b1y1
                if difference <= height:
                    result.append(x)

    # only return b_boxes which are directly above a specific bbox
    # thus filtering b_boxes which doesnt fulfill this requirement
    # is needed when pages are not deskewed
    result_direct = []
    for x in result:
        direct_below = True
        for z in result:
            if is_below(x, z):
                direct_below = False
                break
        if direct_below:
            result_direct.append(x)
    return result_direct


def analyse(baselines, image, image2):
    image = 1 - image
    result = []
    heights = []
    length = []
    img = image2.convert('RGB')
    if baselines is None:
        array = np.array(img)
        pyplot.imshow(array)
        pyplot.show()
        return
    for baseline in baselines:
        index, height = get_top(image=image, baseline=baseline)
        result.append((baseline, index, height))
        heights.append(height)
        length.append(baseline[-1][1])
    img = image2.convert('RGB')
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    from segmentation.postprocessing.util import baseline_to_bbox, crop_image_by_polygon
    height, width = image.shape
    result_dict = {}
    for ind, x in enumerate(result):
        pol = baseline_to_bbox(x[0],
                               margin_top=x[2],
                               margin_bot=0,
                               margin_left=0,
                               margin_right=0)
        cut = crop_image_by_polygon(polygon=pol, image=image)
        from segmentation.postprocessing.util import get_stroke_width
        score1, score2 = get_stroke_width(cut[0])
        baseline = x[0]
        p1 = baseline[0]
        p2 = baseline[-1]
        vector = [x[2] / np.max(heights), score1]
        vector2 = [p1[0] / max(length), p2[0] / max(length)]
        result_dict[ind] = [baseline, score1, x[2], vector, vector2]
        draw.line(list(itertools.chain.from_iterable(baseline)), fill=colors[ind % len(colors)], width=2)

    inds = result_dict.keys()
    vectors = [result_dict[indice][3] for indice in inds]
    h, s = zip(*vectors)
    vectors = [[h_s, 0] for h_s, s_s in list(zip(h, s))]
    vectors2 = [result_dict[indice][4] for indice in inds]
    t = DBSCAN(eps=0.1, min_samples=1).fit(np.array(vectors))
    e = DBSCAN(eps=0.01, min_samples=1).fit(np.array(vectors2))

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
    bboxes = connect_bounding_box(bboxes)
    for ind, x in enumerate(bboxes):
        if x.bbox:
            draw.line(x.bbox + [x.bbox[0]], fill=colors[ind % len(colors)], width=3)

    array = np.array(img)
    # pyplot.imshow(array)
    # pyplot.show()
    return bboxes


def connect_bounding_box(bboxes: [List[BboxCluster]]):
    bboxes_clone = bboxes.copy()
    clusters = []
    cluster = []

    def alpha_shape_from_list_of_bboxes(clusters):
        def merge_ponts_to_box(point_list):
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
            array = merge_ponts_to_box(point_list=y)

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
            if len(get_bboxs_above(x, bboxes)) > 1 or (len(clusters) != 0 and len(clusters[-1]) != 0 and
                                                       len(get_bboxs_below(clusters[-1][-1], bboxes)) > 1):
                print("43444")

                print(x.bbox)
                for t in get_bboxs_above(x, bboxes):
                    print(t.bbox)
                print("43444")

                clusters.append(cluster)
                cluster = []
                break

            if type1 == type2:
                if is_above(x, cluster[-1]) and (abs(b1x1 - b2x1) < 150 or abs(b1x2 - b2x2) < 150):
                    if abs(b2y1 - b1y1) < height:
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
                            print(abs(b3y2 - b4y1))
                            if type3 == type2 and abs(b3y2 - b4y1) < height / 2:
                                print("31223123123")
                                clusters.append(cluster)
                                cluster = []

                        cluster.append(x)
                        del bboxes_clone[ind]
                        break
            if ind == 0:
                print("444")

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
            elif (start_x <= start_x_2 and end_x >= end_x_2) \
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


def get_top(image, baseline, threshold=0.2):
    x, y = zip(*baseline)
    indexes = (np.array(y), np.array(x))
    before = 0
    height = 0
    while True:
        indexes = (indexes[0] - 1, indexes[1])
        now = np.sum(image[indexes])
        if before * threshold > now and height > 5:
            break
        height = height + 1
        before = now if now > before else before
    return list(zip(indexes[1], indexes[0])), height


if __name__ == '__main__':
    files = [
        "/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/OCR-D/images/reinkingk_policey_1653_0016.png"]
    model_paths = ["/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/adam_unet_efficientnet-b3_40_1.torch"]
    ''',
     "/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/adam_unet_efficientnet-b5_20_1.torch",
     "/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/adam_unet_efficientnet-b4_20_1.torch",
     "/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/adam_unet_inceptionresnetv2_20_1.torch",
     "/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/adam_unet_resnet50_20_1.torch"]
    '''
    networks = []
    from segmentation.scripts.predict import Ensemble
    from PIL import Image, ImageDraw, ImageFont

    files0 = list(itertools.chain.from_iterable(
        [glob.glob("/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/OCR-D/images/*.png")]))
    files1 = list(itertools.chain.from_iterable(
        [glob.glob("/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/norbert_fischer/lgt/bin/*.png")]))
    files2 = list(itertools.chain.from_iterable(
        [glob.glob("/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/narren/GW5049/images/*.png")]))
    for x in model_paths:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for file in files0:
        p_map, scale_factor = ensemble(file, scale_area=1000000)
        baselines = extraxct_baselines_from_probability_map(p_map)

        image = Image.open(file)
        image = image.resize((int(scale_factor * image.size[0]), int(scale_factor * image.size[1])))

        from segmentation.preprocessing.basic_binarizer import gauss_threshold
        from segmentation.preprocessing.util import to_grayscale

        grayscale = to_grayscale(np.array(image))
        binary = gauss_threshold(image=grayscale) / 255

        analyse(baselines=baselines, image=binary, image2=image)
