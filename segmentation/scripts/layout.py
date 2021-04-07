import argparse
import glob
import itertools
import multiprocessing
import os
from dataclasses import dataclass
from typing import NamedTuple, List, Dict
from enum import Enum
import numpy as np
from functools import partial

from segmentation.gui.xml_util import TextRegion, BaseLine, TextLine, XMLGenerator
from segmentation.postprocessing.data_classes import PredictionResult, BboxCluster
from segmentation.postprocessing.debug_draw import DebugDraw
from segmentation.postprocessing.layout_analysis import get_top_of_baselines, get_top_of_baselines_improved, analyse, connect_bounding_box
from segmentation.postprocessing.layout_line_segment import schnip_schnip_algorithm, cutout_to_polygon, PageContours, \
    fix_coutout_lineendings, LinePoly, CutoutElem, schnip_schnip_algorithm_old
from segmentation.preprocessing.source_image import SourceImage
from segmentation.postprocessing.baselines_util import scale_baseline, make_baseline_continous, simplify_baseline, flip_baseline
from segmentation.util import PerformanceCounter, logger
from segmentation.postprocessing.layout_settings import LayoutProcessingSettings

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
    parser.add_argument("--fix_baseline_points", action="store_true",help="Remove Baseline Points which are outsite the \"legal\" image area")
    parser.add_argument("--assert_binarized", action="store_true", help="Do not allow binarization of the image file")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")
    parser.add_argument("--show_fix_line_endings", action="store_true", help="Show debug information for the line endings fix")
    return parser.parse_args()




# this structure should contain the finished content information of the page in PageXML coordinate space
@dataclass
class AnalyzedRegion:
    bbox: BboxCluster
    baselines: List
    lines_polygons: List
    cutouts: List

    def scale(self, scale_factor:float = 1):
        baselines = [scale_baseline(bl, scale_factor) for bl in self.baselines]
        lines_polygons = [scale_baseline(lp.poly if type(lp) is not list else lp, scale_factor) for lp in self.lines_polygons]
        bbx = self.bbox.scale(scale_factor)
        return AnalyzedRegion(bbox=bbx, baselines=baselines, lines_polygons=lines_polygons, cutouts=None) # todo scale cutouts


@dataclass
class AnalyzedContent:
    baselines: List = None
    lines_polygons: List = None
    bboxs: List = None
    regions: List[AnalyzedRegion] = None

    def to_pagexml_space(self, scale_factor: float) -> 'AnalyzedContent':
        if scale_factor == 1 or scale_factor == 1.0:
            return self

        undo_scale_factor = 1 / scale_factor
        baselines = [scale_baseline(bl, undo_scale_factor) for bl in self.baselines]
        if self.regions:
            reg = [r.scale(undo_scale_factor) for r in self.regions]
        else:
            reg = None
        if self.lines_polygons:
            lp = [scale_baseline(bl, undo_scale_factor) for bl in self.lines_polygons]
        else:
            lp = None
        if self.bboxs:
            bbx = [x.scale(undo_scale_factor) for x in self.bboxs]
        else:
            bbx = None

        return AnalyzedContent(baselines, lp, bbx, regions=reg)

    def export(self, source_image, source_filename, simplified_xml=False) -> XMLGenerator:
        regions = []
        if self.regions is not None:
            for ib, reg in enumerate(self.regions):
                text_lines = []
                for bline, tline in zip(reg.baselines, reg.lines_polygons):
                    text_lines.append(TextLine(coords=tline, baseline=BaseLine(bline)))
                regions.append(TextRegion(text_lines, coords=reg.bbox.bbox))

        elif self.bboxs is not None:
            # Layout segmentation is done, save baselines inside the regions
            for box in self.bboxs:
                text_lines = []
                for b_line in box.baselines:
                    text_region_coord = b_line.baseline + list(reversed(
                        [(x, y - b_line.height) for x, y in b_line.baseline]))
                    text_lines.append(TextLine(coords=text_region_coord,
                                               baseline=BaseLine(simplify_baseline(b_line.baseline))))
                regions.append(TextRegion(text_lines, coords=box.bbox))

        elif self.lines_polygons is not None and simplified_xml:
            # no layout segmentation is done, create text regions for each baseline
            text_lines = []
            for bl, text_region_coord in zip(self.baselines, self.lines_polygons):
                text_lines.append(TextLine(coords=text_region_coord, baseline=BaseLine(bl)))
            w, h = source_image.array().shape[1], source_image.array().shape[0]
            regions.append(TextRegion(text_lines, coords=[(0, 0), (w, 0), (w, h), (0, h)]))
        elif self.lines_polygons is not None and not simplified_xml:
            # no layout segmentation is done, create text regions for each baseline
            for bl, text_region_coord in zip(self.baselines, self.lines_polygons):
                tl = TextLine(coords=text_region_coord, baseline=BaseLine(bl))
                regions.append(TextRegion([tl], coords=text_region_coord))

        xml_gen = XMLGenerator(source_image.pil_image.size[0], source_image.pil_image.size[1],
                               os.path.basename(source_filename), regions=regions)
        return xml_gen


def complex_layout(prediction: PredictionResult, scaled_image: SourceImage, settings: LayoutProcessingSettings, process_pool) -> List: # returns a page xml
    baselines = list(map(make_baseline_continous, prediction.baselines))
    if not settings.source_scale:
        raise NotImplementedError()
    with PerformanceCounter(function_name="Layout Analysis"):
        bboxs = analyse(baselines=baselines, image=(1 - scaled_image.binarized()), use_improved_tops=False)
    from segmentation.postprocessing.marginialia_detection import marginalia_detection

    if settings.marginalia_postprocessing:
        bboxs = marginalia_detection(bboxs, scaled_image.array())
        baselines = [bl.baseline for cluster in bboxs for bl in cluster.baselines]
        bboxs = analyse(baselines=baselines, image=(1 - scaled_image.binarized()), use_improved_tops=False)
    return connect_bounding_box(bboxs)


def generate_lines_polygons(prediction: PredictionResult, scaled_image: SourceImage, process_pool) -> List:
    baselines = list(map(make_baseline_continous, prediction.baselines))
    baseline_tops = list(map(lambda blt: blt.top, get_top_of_baselines_improved(baselines, 1 - scaled_image.binarized(), process_pool=process_pool)))
    middle_lines = [
                [
                    (t[0], b[1] + (t[1] - b[1]) // 5) for b, t in zip(base, top)
                ] for base, top in zip(baselines, baseline_tops)
            ]
    flip = partial(flip_baseline, image_shape=scaled_image.array().shape)
    baseline_bottoms = get_top_of_baselines(
            list(map(flip, middle_lines)),
            1 - np.flip(scaled_image.binarized()),
            process_pool=process_pool)
    baseline_bottoms = [ flip(b.top) for b in baseline_bottoms ]

    cutouts = [ CutoutElem(bl, tc, bc) for bl, tc, bc in zip(baselines, baseline_tops, baseline_bottoms) ]
    contours = PageContours(scaled_image, dilation_amount=1)
    polys = [fix_coutout_lineendings(co, contours, i).poly for i, co in enumerate(cutouts)]
    return polys


def process_layout(prediction, scaled_image: SourceImage, process_pool, settings:LayoutProcessingSettings) -> AnalyzedContent:
    if settings.lines_only:
        return AnalyzedContent(baselines=prediction.baselines, lines_polygons=generate_lines_polygons(prediction, scaled_image, process_pool))
    else:
        analyzed_content = AnalyzedContent(baselines=prediction.baselines, bboxs=complex_layout(prediction, scaled_image, settings, process_pool))
        if settings.schnip_schnip:
            bbox: BboxCluster
            analyzed_content.regions = []
            all_lines: List[LinePoly] = []

            with PerformanceCounter("SchnipSchnip"):
                for bbox in analyzed_content.bboxs:
                    cutouts = schnip_schnip_algorithm_old(scaled_image, prediction, bbox, settings)
                    if settings.fix_line_endings:
                        contours = PageContours(scaled_image, dilation_amount=1)
                        lines = [fix_coutout_lineendings(co,contours, i ) for i, co in enumerate(cutouts)]
                    else:
                        cutout_polys = [cutout_to_polygon(co, scaled_image) for co in cutouts]
                        lines = [LinePoly(poly, co) for poly,co in zip(cutout_polys, cutouts)]
                    all_lines.extend(lines)

                    reg = AnalyzedRegion(bbox, [simplify_baseline(co.bl) for co in cutouts], [line.poly for line in lines], cutouts)
                    analyzed_content.regions.append(reg)

                if settings.debug_show_fix_line_endings:
                    from matplotlib import pyplot as plt
                    fig, ax = plt.subplots(1,2)

                    ax[0].imshow(scaled_image.binarized())
                    for poly in all_lines:
                        x, y = [x[0] for x in poly.poly], [x[1] for x in poly.poly]
                        x.append(x[0])
                        y.append(y[0])
                        ax[0].plot(x, y, color='#ff3333', alpha=1, linewidth=3, solid_capstyle='round')
                        cutout_poly = cutout_to_polygon(poly.cutout, None)
                        cx, cy = [x[0] for x in cutout_poly], [x[1] for x in cutout_poly]
                        cx.append(cx[0])
                        cy.append(cy[0])
                        ax[0].plot(cx, cy, color="#6699cc", alpha=1, linewidth=3)
                    ax[1].imshow(contours.labeled)
                    fig.show()
                    plt.show()
        return analyzed_content


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
        if args.show_layout and analyzed_content.bboxs:
            debug_draw.draw_bboxs(analyzed_content.bboxs)

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
    scaled_prediction = PredictionResult([scale_baseline(bl, scale_factor) for bl in prediction.baselines],
                                         [source_image.array().shape[1], source_image.array().shape[0]], 1)

    if args.fix_baseline_points:
        def allowed_func(p):
            return 0 <= p[0] < scaled_prediction.prediction_shape[1] \
                   and 0 <= p[1] < scaled_prediction.prediction_shape[0]
        new_baselines = []
        for bl in scaled_prediction.baselines:
            bl = list(filter(allowed_func,bl))
            new_baselines.append(bl)
        scaled_prediction.baselines = new_baselines

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


    data = zip(sorted(glob.glob(args.prediction)), itertools.repeat(args))
    if args.show_layout or args.show_lines or args.show_baselines:
        for _ in map(mp_process, data):
            pass
    else:
        with multiprocessing.Pool() as p:
            for _ in p.imap(mp_process, data):
                pass


if __name__ == "__main__":
    main()
