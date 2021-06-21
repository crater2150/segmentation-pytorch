import argparse
import glob
import itertools
import multiprocessing
import os

from tqdm import tqdm

from segmentation.postprocessing.baselines_util import scale_baseline, make_baseline_continous, simplify_baseline
from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.postprocessing.debug_draw import DebugDraw
from segmentation.postprocessing.layout_analysis import get_top_of_baselines_improved
from segmentation.postprocessing.layout_processing import process_layout
from segmentation.postprocessing.layout_settings import LayoutProcessingSettings
from segmentation.preprocessing.source_image import SourceImage
from segmentation.util import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, required=True, help="Glob for .blp.json files containing the baselines")
    parser.add_argument("--image_path", type=str, help="Path to where the image files are stored",
                        required=True)

    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--improved_top_detection", action="store_true", help="Use improved baseline top detection")
    parser.add_argument("--marginalia_postprocessing", action="store_true", help="Enables marginalia postprocessing")

    parser.add_argument("--debug", action="store_true", help="Show a debug image")
    parser.add_argument("--show_baselines", action="store_true", help="Draws baseline to the debug image")
    parser.add_argument("--show_lines", action="store_true", help="Draws line polygons to the debug image")
    parser.add_argument("--show_layout", action="store_true", help="Draws layout regions to the debug image")
    parser.add_argument("--lines_only", action="store_true", help="Only do simple line heuristic")
    parser.add_argument("--schnipschnip", action="store_true", help="Use SchnipSchnip Alg") # use the schnipschnip algorithm
    parser.add_argument("--print_xml", action="store_true", help="Print XML to stdout Files")
    parser.add_argument("--simplified_xml", action="store_true", help="Write simplified XML")
    parser.add_argument("--output_xml", action="store_true", help="Outputs Xml Files")
    parser.add_argument("--output_xml_path", type=str, default=None, help="Directory of the XML output")
    parser.add_argument("--layout_prediction", action="store_true", help="Generates Layout of the page "
                                                                         "based on the baselines")
    parser.add_argument("--layout_method", type=str, choices=["linesonly", "analyse", "analyse+schnipschnip", "full"])

    parser.add_argument("--fix_baseline_points", action="store_true",help="Remove Baseline Points which are outsite the \"legal\" image area")
    parser.add_argument("--assert_binarized", action="store_true", help="Do not allow binarization of the image file")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")
    parser.add_argument("--show_fix_line_endings", action="store_true", help="Show debug information for the line endings fix")
    parser.add_argument("--height_diff_factor", type=int, default=-2,
                        help="line height factor for SchnipSchnip. Use more negative value, if detected lines are not high enough")
    parser.add_argument("--single_threaded", action="store_true", help="Do not use multiprocessing")
    return parser.parse_args()


def layout_debugging(args, analyzed_content, source_image, image_filename):
    if args.show_baselines or args.show_layout or args.show_lines:
        debug_draw = DebugDraw(source_image)
        if args.show_baselines:
            debug_draw.draw_baselines(analyzed_content.baselines)
        if args.show_lines:
            if analyzed_content.regions:
                for reg in analyzed_content.regions:
                    debug_draw.draw_polygons(reg.lines_polygons)
            elif analyzed_content.lines_polygons:
                debug_draw.draw_polygons(analyzed_content.lines_polygons)
            else:
                baselines = list(map(make_baseline_continous, analyzed_content.baselines))
                baseline_tops = get_top_of_baselines_improved(baselines, 1 - source_image.binarized())
                # draw the polygons as well
                polys = []
                for bl, bl_top, _ in baseline_tops:
                    bl = simplify_baseline(bl)
                    bl_top = simplify_baseline(bl_top)
                    text_region_coord = bl + list(reversed(bl_top))
                    polys.append(text_region_coord)
                debug_draw.draw_polygons(polys)
        if args.show_layout and analyzed_content.bboxs or analyzed_content.regions:
            if analyzed_content.bboxs:
                debug_draw.draw_bboxs(analyzed_content.bboxs) # TODO: should this draw the BBoxs or the region polygons
            elif analyzed_content.regions:
                debug_draw.draw_polygons([x.region_polygon for x in analyzed_content.regions])


        if args.output_path_debug_images:
            basename = "debug_" + os.path.basename(image_filename)
            file_path = os.path.join(args.output_path_debug_images, basename)
            debug_draw.image().save(file_path)

        if args.debug:
            from matplotlib import pyplot
            pyplot.imshow(SourceImage(debug_draw.image()).array())
            pyplot.show()


def mp_process(args):
    pred_file, args = args

    with open(pred_file) as f:
        prediction = PredictionResult.from_json(f.read())
        # find the image file
    img_filename = os.path.join(args.image_path, os.path.basename(pred_file).split(".")[0] + ".png")
    source_image = SourceImage.load(img_filename)

    # The following process converts the Source Image to PageXML Space and
    # calculates the layout in PageXML space
    # When using already binarized Images, we would rather want to process the layout
    # In image Space, in order to use
    # correctly scale it
    """
    if prediction.prediction_scale_factor != 1:
        logger.info(f"Using PSF: {prediction.prediction_scale_factor} for {pred_file}")
        scaled_image = source_image.scaled(prediction.prediction_scale_factor)
        scale_factor = prediction.prediction_scale_factor
    elif source_image.array().shape[1] != prediction.prediction_resolution[0] or \
        source_image.array().shape[0] != prediction.prediction_resolution[1]:
        # determine the scale factor
        sf = (source_image.array().shape[0] / prediction.prediction_resolution[1])
        scaled_image = source_image.scaled(sf)
        logger.info(f"Rescaling to: {prediction.prediction_resolution} for {pred_file}")
    else:
        scaled_image = source_image
        logger.info(f"No scaling required for {pred_file}")
    """
    # scale the Baselines to the binarized image's size and do the processing in image space
    scale_factor = (source_image.array().shape[0] / prediction.prediction_shape[0])
    scaled_prediction = PredictionResult(baselines=[scale_baseline(bl, scale_factor) for bl in prediction.baselines],
                                         prediction_shape=list(source_image.array().shape),
                                         prediction_scale_factor=1)

    if args.fix_baseline_points:
        def allowed_func(p):
            return 0 <= p[0] < scaled_prediction.prediction_shape[1] \
                   and 0 <= p[1] < scaled_prediction.prediction_shape[0]
        new_baselines = [list(filter(allowed_func, bl)) for bl in scaled_prediction.baselines]
        scaled_prediction.baselines = [bl for bl in new_baselines if bl]

    layout_settings = LayoutProcessingSettings.from_cmdline_args(args)

    analyzed_content = process_layout(scaled_prediction, source_image, None, layout_settings)

    # layout_debugging(args, analyzed_content, scaled_image, file)

    # convert this back to the original image space
    # analyzed_content = analyzed_content.to_pagexml_space(scaled_image.scale_factor)

    # debugging

    layout_debugging(args, analyzed_content, source_image, img_filename)

    if args.print_xml or (args.output_xml is not None and args.output_xml_path is not None):
        xml_gen = analyzed_content.export(source_image, img_filename, simplified_xml=args.simplified_xml)
        if args.print_xml:
            print(xml_gen.baselines_to_xml_string())
        else:
            xml_gen.save_textregions_as_xml(args.output_xml_path)


def main():
    args = parse_args()

    if args.assert_binarized:
        logger.warning("Assert source images are already binarized.")
        SourceImage.fail_on_binarize=True

    data = list(zip(sorted(glob.glob(args.prediction)), itertools.repeat(args)))
    if args.show_layout or args.show_lines or args.show_baselines or args.single_threaded or args.processes == 1:
        for _ in tqdm(map(mp_process, data), total=len(data)):
            pass
    else:
        with multiprocessing.Pool() as p:
            for _ in tqdm(p.imap_unordered(mp_process, data), total=len(data)):
                pass


if __name__ == "__main__":
    main()
