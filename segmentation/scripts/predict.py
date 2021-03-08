import argparse
import glob
import multiprocessing
import os
import warnings

import torch

from segmentation.postprocessing.baseline_extraction import extract_baselines_from_probability_map
from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.postprocessing.layout_analysis import get_top_of_baselines
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.layout import process_layout, LayoutProcessingSettings, layout_debugging
from segmentation.settings import PredictorSettings
from segmentation.util import PerformanceCounter

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
from segmentation.util import logger
from typing import NamedTuple, List

from segmentation.network import Network

from segmentation.postprocessing.baselines_util import simplify_baseline


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x, scale_area, additional_scale_factor=None):
        raise DeprecationWarning()

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
            return res / len(self.models)  # TODO: check if this is equivalent (it probably is)


class PredictionSettings(NamedTuple):
    model_paths: List[str]
    scale_area: int = None
    min_line_height: int = None
    max_line_height: int = None


class Predictor:
    def __init__(self, settings: PredictionSettings):
        self.settings = settings
        self.networks = [Network(PredictorSettings(MODEL_PATH=x)) for x in self.settings.model_paths]
        self.ensemble = Ensemble(self.networks)

    def predict_image(self, source_image: SourceImage,
                      process_pool: multiprocessing.Pool = multiprocessing.Pool(1)) -> (PredictionResult, SourceImage):
        scale_factor_multiplier = 1
        while True:
            scaled_image = source_image.scale_area(self.settings.scale_area, scale_factor_multiplier)
            with PerformanceCounter("Prediction"):
                p_map = self.ensemble.predict_image(scaled_image)
                baselines = extract_baselines_from_probability_map(p_map, process_pool=process_pool)

            if baselines is not None:
                binary = scaled_image.binarized()
                with PerformanceCounter(function_name="Baseline Height Calculation mp"):
                    out = get_top_of_baselines(baselines, image=1 - binary,
                                               process_pool=None)  # No MP is faster here (avoid image copy)
                heights = [x[2] for x in out]
                med_height = np.median(heights)

                if (self.settings.max_line_height is not None and med_height > self.settings.max_line_height) \
                        or (self.settings.min_line_height is not None and med_height < self.settings.min_line_height) \
                        and scale_factor_multiplier == 1:
                    scale_factor_multiplier = (self.settings.max_line_height - 7) / med_height
                    logger.info("Resizing image Avg:{}, Med:{} \n".format(np.mean(heights), med_height))
                    continue
            break

        # simplify the baselines
        baselines = [simplify_baseline(bl) for bl in baselines or []]
        # now we have the baselines extracted
        return PredictionResult(baselines=baselines,
                                prediction_scale_factor=scaled_image.scale_factor,
                                prediction_shape=list(scaled_image.array().shape)), scaled_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--image_path", type=str, nargs="*", default=[],
                        help="Specify image glob pattern")
    parser.add_argument('files', metavar='FILE', type=str, nargs='*',
                        help='Source files to process')
    parser.add_argument("--scale_area", type=int, default=1000000,
                        help="max pixel amount of an image")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")
    parser.add_argument("--layout_prediction", action="store_true", help="Generates Layout of the page "
                                                                         "based on the baselines")
    parser.add_argument("--show_baselines", action="store_true", help="Draws baseline to the debug image")
    parser.add_argument("--show_lines", action="store_true", help="Draws line polygons to the debug image")
    parser.add_argument("--show_layout", action="store_true", help="Draws layout regions to the debug image")
    parser.add_argument("--show_fix_line_endings", action="store_true",
                        help="Show debug information for the line endings fix")
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
    parser.add_argument("--export_path", type=str, default=None, help="Export Predictions as JSON to given path")
    parser.add_argument("--simplified_xml", action="store_true", help="Output simplified PageXML for LAREX")
    parser.add_argument("--height_diff_factor", type=int, default=-2,
                        help="line height factor for SchnipSchnip. Use more negative value, if detected lines are not high enough")
    parser.add_argument("--schnipschnip", action="store_true",
                        help="Use SchnipSchnip Algorithm to cut Regions into lines")
    parser.add_argument("--twosteps", action="store_true", help="Run two step prediction")

    return parser.parse_args()


def two_step_file_func(data):
    args, prediction, file = data
    source_image = SourceImage.load(file)
    scaled_image = source_image.scaled(prediction.prediction_scale_factor)

    if args.export_path:
        bname_json = os.path.splitext(os.path.basename(file))[0] + ".blp.json"
        with open(os.path.join(args.export_path, bname_json), "w") as f:
            f.write(prediction.to_json())

    scale_factor = prediction.prediction_scale_factor

    layout_settings = LayoutProcessingSettings.from_cmdline_args(args)

    analyzed_content = process_layout(prediction, scaled_image, None, layout_settings)

    # layout_debugging(args, analyzed_content, scaled_image, file)

    # convert this back to the original image space
    analyzed_content = analyzed_content.to_pagexml_space(scale_factor)

    # debugging

    layout_debugging(args, analyzed_content, source_image, file)

    if args.print_xml or (args.output_xml is not None and args.output_xml_path is not None):
        xml_gen = analyzed_content.export(source_image, file, simplified_xml=args.simplified_xml)
        if args.print_xml:
            print(xml_gen.baselines_to_xml_string())
        else:
            xml_gen.save_textregions_as_xml(args.output_xml_path)


def main():
    args = parse_args()
    if args.image_path:
        if args.files is not None and args.files != []:
            logger.error(f"Cannot specify --image_path and positional arguments at the same time")

        logger.warn(f"Using glob filenames: {args.image_path}.")
        logger.warn("Glob might silently skip unreadable or unaccessable files.")
        files = sorted(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    else:
        files = args.files
    settings = PredictionSettings(args.load, args.scale_area, args.min_line_height, args.max_line_height)

    torch.set_num_threads(multiprocessing.cpu_count())
    # TODO: On Ryzen 5600X this does not improve performance
    # To improve CPU prediction performance, maybe prefetch img load and run distance_matrix on multiple cores

    nn_predictor = Predictor(settings)
    if not args.twosteps:
        with multiprocessing.Pool(args.processes) as process_pool:
            for file in files:
                logger.info("Processing: {} \n".format(file))
                source_image = SourceImage.load(file)

                prediction, scaled_image = nn_predictor.predict_image(source_image, process_pool=process_pool)

                if args.export_path:
                    bname_json = os.path.splitext(os.path.basename(file))[0] + ".blp.json"
                    with open(os.path.join(args.export_path, bname_json), "w") as f:
                        f.write(prediction.to_json())

                scale_factor = prediction.prediction_scale_factor

                layout_settings = LayoutProcessingSettings.from_cmdline_args(args)

                analyzed_content = process_layout(prediction, scaled_image, process_pool, layout_settings)

                # layout_debugging(args, analyzed_content, scaled_image, file)

                # convert this back to the original image space
                analyzed_content = analyzed_content.to_pagexml_space(scale_factor)

                # debugging

                layout_debugging(args, analyzed_content, source_image, file)

                if args.print_xml or (args.output_xml is not None and args.output_xml_path is not None):
                    xml_gen = analyzed_content.export(source_image, file, simplified_xml=args.simplified_xml)
                    if args.print_xml:
                        print(xml_gen.baselines_to_xml_string())
                    else:
                        xml_gen.save_textregions_as_xml(args.output_xml_path)
    else:  # two step prediction
        predictions = []
        with multiprocessing.Pool() as pool:
            for file in files:
                logger.info("Processing: {} \n".format(file))
                source_image = SourceImage.load(file)

                prediction, _ = nn_predictor.predict_image(source_image, process_pool=pool)
                predictions.append(prediction)

            data = [(args, pred, file) for pred, file in zip(predictions, files)]

            for _ in pool.imap_unordered(two_step_file_func, data):
                pass


if __name__ == "__main__":
    with PerformanceCounter("Total running time"):
        main()
