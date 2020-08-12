import argparse
import json
from os import path
from typing import List
import warnings
import glob

from matplotlib import pyplot

from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
from segmentation.settings import PredictorSettings
import torch
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
from segmentation.postprocessing import baseline_extraction
from PIL import Image, ImageDraw


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def scale_baselines(baselines, scale_factor=1.0):
    for b_idx, bline in enumerate(baselines):
        for c_idx, coord in enumerate(bline):
            coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
            baselines[b_idx][c_idx] = coord


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        scale_factor = None
        for m in self.models:
            p_map, s_factor = m.predict_single_image_by_path(x, rgb=True, preprocessing=True)
            scale_factor = s_factor
            res.append(p_map)
        if len(res) == 1:
            return res[0], scale_factor
        res = np.stack(res, axis=0)
        return np.mean(res, axis=0), scale_factor


def main():
    from segmentation.network import TrainSettings, dirs_to_pandaframe, load_image_map_from_file, MaskSetting, MaskType, PCGTSVersion, XMLDataset, Network, compose, MaskGenerator, MaskDataset
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--image_path", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    args = parser.parse_args()
    files = list(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    networks = []
    for x in args.load:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for x in files:
        p_map, scale_factor = ensemble(x)
        baselines = extraxct_baselines_from_probability_map(p_map)
        if baselines is not None and len(baselines) > 0:
            scale_baselines(baselines, 1 / scale_factor)
            img = Image.open(x)  # open image
            draw = ImageDraw.Draw(img)
            for ind, x in enumerate(baselines):
                t = list(itertools.chain.from_iterable(x))
                a = t[::]
                draw.line(a, fill=colors[ind % len(colors)], width=4)
        array = np.array(img)
        pyplot.imshow(array)
        pyplot.show()


if __name__ == "__main__":
    main()