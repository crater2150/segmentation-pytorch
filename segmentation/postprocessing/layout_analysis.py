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



def analyse(baselines, image, image2):
    image = 1 - image
    result = []
    heights = []
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
        vector = [x[2] / max(heights), score1]
        vector2 = [p1[0] / width, p2[0] / width]
        result_dict[ind] = [baseline, score1, x[2], vector, vector2]
        draw.line(list(itertools.chain.from_iterable(baseline)), fill=colors[ind % len(colors)], width=2)

    inds = result_dict.keys()
    vectors = [result_dict[indice][3] for indice in inds]
    vectors2 = [result_dict[indice][4] for indice in inds]

    t = DBSCAN(eps=0.08, min_samples=1).fit(np.array(vectors))
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
    bboxes = get_bbounding_box_of_cluster(clustered=clusterd)
    bboxes = connect_bounding_box(bboxes)
    for ind, x in enumerate(bboxes):
        if x.bbox:
            draw.line(x.bbox + [x.bbox[0]], fill=colors[ind % len(colors)], width=3)

    array = np.array(img)
    # pyplot.imshow(array)
    # pyplot.show()
    return bboxes


def remove_small_baselines(bboxes: List[BboxCluster]):
    for x in previous_and_next(bboxes):
        pass


def connect_bounding_box(bboxes: [List[BboxCluster]]):
    bboxes_clone = bboxes.copy()
    clusters = []
    cluster = []

    def alpha_shape_from_list_of_bboxes(clusters):
        def merge_ponts_to_box(points):
            if len(points) == 4:
                return points

            def split_list_equally(list):
                list1 = []
                list2 = []
                for ind, x in enumerate(list):
                    if ind % 2 == 0:
                        list1.append(x)
                    else:
                        list2.append(x)
                return list1, list2

            # nb = len(points) / 4
            points = sorted(points, key=lambda k: (k[1], k[0]))
            l_1, l_2 = split_list_equally(points)
            array = l_1 + list(reversed(l_2))
            return array

        bboxes = []
        for item in clusters:
            points = list(itertools.chain.from_iterable([x.bbox for x in item]))
            array = merge_ponts_to_box(points=points)

            baselines = []
            for x in item:
                baselines = baselines + x.baselines
            bboxes.append(BboxCluster(baselines=baselines, bbox=array))
        return bboxes

    while len(bboxes_clone) != 0:
        for ind, x in reversed(list(enumerate(bboxes_clone))):
            if len(cluster) == 0:
                cluster.append(x)
                del bboxes_clone[ind]
                break
            bbox = cluster[-1].bbox
            height = min(cluster[-1].baselines[0].height, x.baselines[0].height)
            x1, y1 = zip(*bbox)
            type1 = cluster[-1].baselines[0].cluster_type
            x2, y2 = zip(*x.bbox)
            type2 = x.baselines[0].cluster_type

            if type1 == type2:
                if abs(min(x1)-min(x2)) < 20 or abs(max(x1)-max(x2)) < 20:
                    if abs(max(y1) - min(y2)) < height or abs(min(y1) - max(y2)) < height:
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


def get_bbounding_box_of_cluster(clustered: List[List[BaselineResult]]):
    boxes = []

    def get_border(cluster: List[BaselineResult]):

        xmin = math.inf
        xmax = 0
        ymin = math.inf
        ymax = 0
        height = 0
        for item in cluster:
            before = xmin
            x, y = list(zip(*item.baseline))
            x = list(x)
            y = list(y)
            xmin = min(x + [xmin])
            ymin = min(y + [ymin])
            xmax = max(x + [xmax])
            ymax = max(y + [ymax])
            if before != xmin:
                height = item.height
        return BboxCluster(baselines=cluster,
                           bbox=[(xmin, ymin - height), (xmax, ymin - height), (xmax, ymax), (xmin, ymax)])

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
