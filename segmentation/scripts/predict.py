import argparse
import multiprocessing
from os import path
import warnings
import glob
import os

import scipy
from skimage.filters import try_all_threshold, threshold_local
from PIL import Image, ImageDraw
from segmentation.postprocessing.baseline_extraction import extract_baselines_from_probability_map
from segmentation.postprocessing.layout_analysis import analyse, connect_bounding_box, get_top_of_baselines, get_top_of_baselines_improved
from segmentation.settings import PredictorSettings
from segmentation.util import PerformanceCounter
from segmentation.dataset import get_rescale_factor, rescale_pil
from segmentation.preprocessing.ocrupus import binarize

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
from segmentation.util import logger
from functools import cached_property


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def scale_baselines(baselines, scale_factor=1.0):
    if baselines is not None:
        for b_idx, bline in enumerate(baselines):
            for c_idx, coord in enumerate(bline):
                coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
                baselines[b_idx][c_idx] = coord

def simplify_baseline(bl):
    new_coords = []
    new_coords.append(bl[0])

    for before, cur, after in zip(bl, bl[1:-1], bl[2:]):
        if before[1] == cur[1] and cur[1] == after[1]:
            continue
        else:
            new_coords.append(cur)
    new_coords.append(bl[-1])
    return new_coords


class SourceImage:
    @staticmethod
    def load(filename):
        img = Image.open(filename)
        return SourceImage(img)

    def __init__(self, img: Image, scale_factor=1):
        self.pil_image = img
        self.binarized_cache = None
        self.array_cache = None
        self.scale_factor = scale_factor

    def scaled(self, scale_factor) -> 'SourceImage':
        return SourceImage(rescale_pil(self.pil_image, scale_factor), scale_factor=scale_factor)

    def scale_area(self, max_area, additional_scale_factor=None) -> 'SourceImage':

        rescale_factor = get_rescale_factor(self.pil_image, scale_area=max_area)

        if additional_scale_factor is not None:
            rescale_factor = rescale_factor * additional_scale_factor

        return self.scaled(rescale_factor)

    def binarized(self):
        if self.binarized_cache is None:
            self.binarized_cache = binarize(self.array().astype("float64")).astype("uint8")
        return self.binarized_cache

    def array(self):
        if self.array_cache is None:
            self.array_cache = np.array(self.pil_image, dtype=np.uint8)
        return self.array_cache

    def is_rescaled(self):
        return self.scale_factor != 1

class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x, scale_area, additional_scale_factor=None):
        raise DeprecationWarning()
        res = []
        scale_factor = None
        for m in self.models:
            p_map, s_factor = m.predict_single_image_by_path(x, rgb=True, preprocessing=True, scale_area=scale_area,
                                                             additional_scale_factor=additional_scale_factor)
            scale_factor = s_factor
            res.append(p_map)
        if len(res) == 1:
            return res[0], scale_factor
        res = np.stack(res, axis=0)
        return np.mean(res, axis=0), scale_factor

    def predict_image(self, source_image: SourceImage):
        def predict(m):
            # tta_aug=None means default augmentation
            return m.predict_single_image(source_image.array(), rgb=True, preprocessing=True, tta_aug=None)

        if len(self.models) == 1:
            return predict(self.models[0])
        else:
            res = np.zeros(shape=source_image.array().shape, dtype=np.float32)
            for m in self.models:
                res += predict(m)
            return res / len(self.models) # TODO: check if this is equivalent

class DebugDrawDummy:
    def __init__(self, *args, **kwargs):
        pass
    def draw_bboxs(self, bboxs):
        pass
    def draw_baselines(self, baselines):
        pass
    def image(self):
        raise NotImplementedError("requesting image but drawing is disables")


class DebugDraw:
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    def __init__(self, source_image: SourceImage):
        self.img = source_image.pil_image.copy().convert('RGB')
        self.draw = ImageDraw.Draw(self.img)

    def draw_bboxs(self, bboxs):
        for ind, x in enumerate(bboxs):
            if x.bbox:
                self.draw.line(x.bbox + [x.bbox[0]], fill=DebugDraw.colors[ind % len(DebugDraw.colors)], width=3)
                self.draw.text((x.bbox[0]), "type:{}".format(x.baselines[0].cluster_type))

    def draw_baselines(self, baselines):
        if baselines is None or len(baselines) == 0:
            return
        for ind, x in enumerate(baselines):
            t = list(itertools.chain.from_iterable(x))
            a = t[::]
            self.draw.line(a, fill=DebugDraw.colors[ind % len(DebugDraw.colors)], width=4)

    def image(self):
        return self.img






def main():
    from segmentation.network import TrainSettings, dirs_to_pandaframe, load_image_map_from_file, MaskSetting, MaskType, \
        PCGTSVersion, XMLDataset, Network, compose, MaskGenerator, MaskDataset
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS


    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--image_path", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--scale_area", type=int, default=1000000,
                        help="max pixel amount of an image")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")
    parser.add_argument("--layout_prediction", action="store_true", help="Generates Layout of the page "
                                                                         "based on the baselines")
    parser.add_argument("--show_baselines", action="store_true", help="Draws baseline to the debug image")
    parser.add_argument("--show_layout", action="store_true", help="Draws layout regions to the debug image")
    parser.add_argument("--output_xml", action="store_true", help="Outputs Xml Files")
    parser.add_argument("--output_xml_path", type=str, default=None, help="Directory of the XML output")
    parser.add_argument("--max_line_height", type=int, default=None,
                        help="If the average line_height of an document is bigger then the specified value, "
                             "the document is scaled down an processed again on the new resolution. "
                             "Proposed Value == 22")
    parser.add_argument("--min_line_height", type=int, default=None,
                        help="If the average line_height of an document is smaller then the specified value, "
                             "the document is scaled up an processed again on the new resolution")
    parser.add_argument("--marginalia_postprocessing", action="store_true", help="Enables marginalia postprocessing")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--improved_top_detection", action="store_true", help="Use improved baseline top detection")
    parser.add_argument("--invertmatch", action="store_true", help="Using inverted matching for line height prediction")
    parser.add_argument("--print_xml", action="store_true", help="Print XML to stdout")

    args = parser.parse_args()
    files = list(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    networks = []
    bboxs = None
    for x in args.load:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)

    with multiprocessing.Pool() as process_pool:
        for file in files:
            baselines = None
            logger.info("Processing: {} \n".format(file))
            source_image = SourceImage.load(file)

            scale_factor_multiplier = 1
            while True:
                scaled_image = source_image.scale_area(args.scale_area,scale_factor_multiplier)
                p_map = ensemble.predict_image(scaled_image)
                baselines = extract_baselines_from_probability_map(p_map, process_pool=process_pool)

                scale_factor = scaled_image.scale_factor

                # image = img.resize((int(scale_factor * img.size[0]), int(scale_factor * img.size[1])))
                # img = img.convert('RGB')
                # draw = ImageDraw.Draw(img)
                #from matplotlib import pyplot as plt
                #f, ax = plt.subplots(1, 3, True, True)
                #ax[0].imshow(image)
                #map = scipy.special.softmax(p_map, axis=-1)
                #ax[1].imshow(map[:,:,1])
                #ax[2].imshow(map[:,:,2])

                #plt.show()
                if args.show_baselines or args.show_layout:
                    debug_draw = DebugDraw(source_image)
                else:
                    debug_draw = DebugDrawDummy()

                if baselines is not None:

                    binary = scaled_image.binarized()
                    with PerformanceCounter(function_name="Baseline Height Calculation mp"):
                        out = get_top_of_baselines(baselines, image=1 - binary, process_pool=None) # No MP is faster here (avoid image copy)
                    heights = [x[2] for x in out]

                    if (args.max_line_height is not None or args.min_line_height is not None) \
                            and scale_factor_multiplier == 1:

                        if (args.max_line_height is not None and np.median(heights) > args.max_line_height) or \
                                (args.min_line_height is not None and np.median(heights) < args.min_line_height):
                            scale_factor_multiplier = (args.max_line_height - 7) / np.median(heights)
                            logger.info("Resizing image Avg:{}, Med:{} \n".format(np.mean(heights), np.median(heights)))
                            continue

                     # Deactivate this layout prediction and instead use simple heuristics to get individual textlines
                    if args.layout_prediction:
                        with PerformanceCounter(function_name="Layout Analysis"):
                            bboxs = analyse(baselines=baselines, image=(1 - binary), image2=scaled_image.array())
                        from segmentation.postprocessing.marginialia_detection import marginalia_detection
                        if args.marginalia_postprocessing:
                            bboxs = marginalia_detection(bboxs, scaled_image.array())
                            baselines = [bl.baseline for cluster in bboxs for bl in cluster.baselines]
                            bboxs = analyse(baselines=baselines, image=(1 - binary), image2=scaled_image.array())
                        bboxs = connect_bounding_box(bboxs)
                        bboxs = [x.scale(1 / scale_factor) for x in bboxs]
                        if args.show_layout:
                            debug_draw.draw_bboxs(bboxs)

                scale_baselines(baselines, 1 / scale_factor)

                if args.show_baselines:
                    debug_draw.draw_baselines(baselines)

                if args.output_path_debug_images:
                    basename = "debug_" + os.path.basename(file)
                    file_path = os.path.join(args.output_path_debug_images, basename)
                    debug_draw.image().save(file_path)

                if args.print_xml or (args.output_xml is not None and args.output_xml_path is not None):
                    from segmentation.gui.xml_util import TextRegion, BaseLine, TextLine, XMLGenerator
                    regions = []

                    if bboxs is not None:
                        # Layout segmentation is done, save baselines inside teh regions
                        for box in bboxs:
                            text_lines = []
                            for b_line in box.baselines:
                                text_region_coord = b_line.baseline + list(reversed(
                                    [(x, y - b_line.height) for x, y in b_line.baseline]))
                                text_lines.append(TextLine(coords=text_region_coord, baseline=BaseLine(simplify_baseline(b_line.baseline))))
                            regions.append(TextRegion(text_lines, coords=box.bbox))

                    elif baselines is not None:
                        # no layout segmentation is done, create text regions for each baseline
                        text_lines = []
                        baseline_tops = get_top_of_baselines_improved(baselines, source_image.binarized())

                        for bl, bl_top, _ in baseline_tops:
                            bl = simplify_baseline(bl)
                            bl_top = simplify_baseline(bl_top)
                            text_region_coord = bl + list(reversed(bl_top))
                            text_lines.append(TextLine(coords=text_region_coord, baseline=BaseLine(bl)))
                        w,h = source_image.array().shape[1], source_image.array().shape[0]
                        regions.append(TextRegion(text_lines, coords=[(0,0), (w,0), (w,h), (0,h)]))

                    xml_gen = XMLGenerator(source_image.pil_image.size[0], source_image.pil_image.size[1], os.path.basename(file), regions=regions)
                    if (args.print_xml):
                        print(xml_gen.baselines_to_xml_string())
                    else:
                        xml_gen.save_textregions_as_xml(args.output_xml_path)

                if args.debug:
                    from matplotlib import pyplot
                    pyplot.imshow(SourceImage(debug_draw.image()).array())
                    pyplot.show()
                break
                pass


if __name__ == "__main__":
    main()
