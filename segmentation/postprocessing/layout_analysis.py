import glob
import itertools
import math
from typing import NamedTuple, List, Tuple
from collections import namedtuple
from matplotlib import pyplot
from skimage.morphology import skeletonize
from sklearn.cluster import dbscan

from segmentation.network import Network
from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
from segmentation.settings import PredictorSettings
import numpy as np
from sklearn.cluster import DBSCAN

'''
Todo: Refactor file
'''


class ClusterResult(NamedTuple):
    baseline: List
    height: int
    font_width: float
    cluster_type: int
    cluster_location: int


class BboxBaselines(NamedTuple):
    baselines: List[ClusterResult]
    bbox: List[Tuple[int]]


def analyse(baselines, image, image2):
    image = 1 - image
    result = []
    heights = []
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
        #draw.polygon(pol, outline=colors[ind % len(colors)])
        draw.text((pol[0]), "w:{},h:{},l:{} l:{}".format(round(meta[1], 3), meta[2], t.labels_[ind],
                                                         e.labels_[ind]), fill=(14, 183, 242))  # ), font=ImageFont.truetype("font_path123"))
        cluster_results.append(ClusterResult(baseline=meta[0],
                                             height=meta[2],
                                             font_width=meta[1],
                                             cluster_type=t.labels_[ind],
                                             cluster_location=e.labels_[ind]))

    clusterd = generate_clustered_lines(cluster_results)
    bboxes = get_bbounding_box_of_cluster(clustered=clusterd)
    for ind, x in enumerate(bboxes):
        draw.line(x + [x[0]], fill=colors[ind % len(colors)], width=3)

    array = np.array(img)
    pyplot.imshow(array)
    pyplot.show()


def get_bbounding_box_of_cluster(clustered: List[List[ClusterResult]]):
    boxes = []

    def get_border(cluster: List[ClusterResult]):
        xmin = math.inf
        xmax = 0
        ymin = math.inf
        ymax = 0
        for item in cluster:
            x, y = item.baseline[0]
            x2, y2 = item.baseline[-1]
            xmin = min(x, x2, xmin)
            ymin = min(y - item.height, y2 - item.height, ymin)
            xmax = max(x, x2, xmax)
            ymax = max(y, y2, ymax)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    for t in clustered:
        boxes.append(get_border(t))
    return boxes


def generate_clustered_lines(cluster_results: List[ClusterResult]):
    clone = sorted(cluster_results, key=lambda t: t.baseline[0][1])
    clustered = []
    cluster = []
    while len(clone) > 0:
        for ind, x in enumerate(reversed(clone)):

            if x.cluster_type == 2 and x.cluster_location == 7:
                pass
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

    files = list(itertools.chain.from_iterable(
        [glob.glob("/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/OCR-D/images/*.png")]))
    files = list(itertools.chain.from_iterable(
        [glob.glob("/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/norbert_fischer/lgt/bin/*.png")]))
    for x in model_paths:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for file in files:
        p_map, scale_factor = ensemble(file, scale_area=1000000)
        baselines = extraxct_baselines_from_probability_map(p_map)

        image = Image.open(file)
        image = image.resize((int(scale_factor * image.size[0]), int(scale_factor * image.size[1])))

        from segmentation.preprocessing.basic_binarizer import gauss_threshold
        from segmentation.preprocessing.util import to_grayscale

        grayscale = to_grayscale(np.array(image))
        binary = gauss_threshold(image=grayscale) / 255

        analyse(baselines=baselines, image=binary, image2=image)
