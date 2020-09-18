import argparse
import json
from os import path
from typing import List
import warnings

from segmentation.settings import PredictorSettings

warnings.simplefilter(action='ignore', category=FutureWarning)


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
    from segmentation.network import TrainSettings, dirs_to_pandaframe, load_image_map_from_file, MaskSetting, MaskType, PCGTSVersion, XMLDataset, Network, compose, MaskGenerator, MaskDataset
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default=None,
                        help="load an existing model and continue training")
    parser.add_argument("--test_input", type=dir_path, nargs="*", default=[], help="Path to folder(s) containing test images")
    parser.add_argument("--test_mask", type=dir_path, nargs="+", default=[], help="Path to folder(s) containing test xmls")
    parser.add_argument("--color-map", dest="map", type=str, required=True,
                        help="path to color map to load")

    args = parser.parse_args()

    test = dirs_to_pandaframe(args.test_input, args.test_mask)
    i_map = load_image_map_from_file(args.map)
    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    d_test = XMLDataset(test, i_map,
                        mask_generator=MaskGenerator(settings=settings))
    p_setting = PredictorSettings(PREDICT_DATASET=d_test,
                                  MODEL_PATH=args.load)
    network = Network(p_setting, color_map=i_map)

    accuracy, loss = network.eval()

    return accuracy, loss


if __name__ == "__main__":
    main()