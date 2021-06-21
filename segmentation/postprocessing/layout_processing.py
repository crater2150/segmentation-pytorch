import itertools
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List
from typing import NamedTuple

import numpy as np
import shapely.ops
from shapely.geometry import Polygon

from segmentation.gui.xml_util import TextRegion, BaseLine, TextLine, XMLGenerator
from segmentation.postprocessing.baselines_util import scale_baseline, make_baseline_continous, simplify_baseline, \
    flip_baseline
from segmentation.postprocessing.data_classes import PredictionResult, BboxCluster
from segmentation.postprocessing.layout_analysis import get_top_of_baselines, get_top_of_baselines_improved, analyse, \
    connect_bounding_box
from segmentation.postprocessing.layout_line_segment import schnip_schnip_algorithm, cutout_to_polygon, PageContours, \
    fix_cutout_lineendings, LinePoly, CutoutElem, schnip_schnip_algorithm_old, BaselineHashedLookup, BBox
from segmentation.postprocessing.layout_settings import LayoutProcessingSettings, LayoutProcessingMethod
from segmentation.postprocessing.util import UnionFind
from segmentation.preprocessing.source_image import SourceImage
from segmentation.util import PerformanceCounter
from segmentation.util import logger

# this structure should contain the finished content information of the page in PageXML coordinate space
@dataclass
class AnalyzedRegion:
    bbox: BboxCluster
    baselines: List
    lines_polygons: List
    cutouts: List
    region_polygon: List = None

    def scale(self, scale_factor:float = 1):
        baselines = [scale_baseline(bl, scale_factor) for bl in self.baselines]
        lines_polygons = [scale_baseline(lp.poly if type(lp) is not list else lp, scale_factor) for lp in self.lines_polygons]
        if self.bbox:
            bbx = self.bbox.scale(scale_factor)
        else:
            bbx = None
        if self.region_polygon:
            rp = scale_baseline(self.region_polygon, scale_factor)
        else:
            rp = None
        return AnalyzedRegion(bbox=bbx, baselines=baselines, lines_polygons=lines_polygons, cutouts=None, region_polygon=rp) # TODO: scale cutouts

    def get_region_polygon(self):
        if self.region_polygon:
            return self.region_polygon
        else:
            return self.bbox.bbox

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
        if self.baselines:
            baselines = [scale_baseline(bl, undo_scale_factor) for bl in self.baselines]
        else:
            baselines = None
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
                regions.append(TextRegion(text_lines, coords=reg.get_region_polygon()))

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

    cutouts = [CutoutElem(bl, tc, bc) for bl, tc, bc in zip(baselines, baseline_tops, baseline_bottoms)]
    contours = PageContours(scaled_image, dilation_amount=1)
    polys = [co.poly for co  in fix_cutout_lineendings(cutouts, contours)]
    return polys

def show_fix_line_endings(scaled_image, lines: List[LinePoly], contours: PageContours):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(scaled_image.binarized())
    for poly in lines:
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

def lines_to_analysed_content(lines: List[LinePoly], scaled_image: SourceImage) -> AnalyzedContent:
    cutouts = [l.cutout for l in lines]
    return AnalyzedContent(baselines=[simplify_baseline(co.bl) for co in cutouts],
                         lines_polygons=[line.poly for line in lines])

def process_layout_linesonly(prediction: PredictionResult, scaled_image: SourceImage, process_pool, settings:LayoutProcessingSettings) -> AnalyzedContent:
    return AnalyzedContent(baselines=prediction.baselines, lines_polygons=generate_lines_polygons(prediction, scaled_image, process_pool))


def process_layout_analyse(prediction: PredictionResult, scaled_image: SourceImage, process_pool, settings:LayoutProcessingSettings) -> AnalyzedContent:
    analyzed_content = AnalyzedContent(baselines=prediction.baselines,
                                       bboxs=complex_layout(prediction, scaled_image, settings, process_pool))
    if settings.layout_method == LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP or settings.layout_method == LayoutProcessingMethod.SCHNIPSCHNIP_REGIONSONLY:
        bbox: BboxCluster
        analyzed_content.regions = []
        all_lines: List[LinePoly] = []

        with PerformanceCounter("SchnipSchnip"):
            for bbox in analyzed_content.bboxs:
                cutouts = schnip_schnip_algorithm_old(scaled_image, prediction, bbox, settings)
                if settings.fix_line_endings:
                    contours = PageContours(scaled_image, dilation_amount=1)
                    lines = fix_cutout_lineendings(cutouts, contours)
                else:
                    cutout_polys = [cutout_to_polygon(co, scaled_image) for co in cutouts]
                    lines = [LinePoly(poly, co) for poly, co in zip(cutout_polys, cutouts)]
                all_lines.extend(lines)

                reg = AnalyzedRegion(bbox, [simplify_baseline(co.bl) for co in cutouts],
                                     [line.poly for line in lines], cutouts)
                analyzed_content.regions.append(reg)

        if settings.debug_show_fix_line_endings:
            show_fix_line_endings(scaled_image, all_lines, contours)

    return analyzed_content



def process_layout_full(prediction: PredictionResult, scaled_image: SourceImage, process_pool, settings:LayoutProcessingSettings) -> AnalyzedContent:

    cutouts = schnip_schnip_algorithm(scaled_image, prediction, settings)

    if settings.fix_line_endings:
        contours = PageContours(scaled_image, dilation_amount=1)
        lines = fix_cutout_lineendings(cutouts, contours)
    else:
        cutout_polys = [cutout_to_polygon(co, scaled_image) for co in cutouts]
        lines = [LinePoly(poly, co) for poly, co in zip(cutout_polys, cutouts)]

    if settings.debug_show_fix_line_endings:
        show_fix_line_endings(scaled_image, lines, contours or PageContours(scaled_image, dilation_amount=1))
    reg = lines_to_analysed_content(lines, scaled_image=scaled_image)


    return reg

    new_prediction = PredictionResult([c.bl for c in cutouts], prediction.prediction_shape, prediction.prediction_scale_factor)

    bboxs = complex_layout(new_prediction,scaled_image,settings)

    # analyse the layout
    bboxs = complex_layout(new_prediction, scaled_image, settings, process_pool)

    bl_lookup = BaselineHashedLookup(prediction.baselines)


    for bbox in analyzed_content.bboxs:
        bbox_bl_indices = [bl_lookup.get_index(bl.baseline) for bl in bbox.baselines]
        bbox_cutouts = [cutouts[idx] for idx in bbox_bl_indices]
        polygons = [cutout_to_polygon(c) for c in bbox_cutouts]
        analyzed_content.regions.append(
            AnalyzedRegion(bbox, [c.bl for c in bbox_cutouts],
                           polygons, bbox_cutouts)
        )
        all_lines.extend([LinePoly(p, c) for p, c in zip(polygons, cutouts)])


def make_multipolygon_solid(mp: shapely.geometry.MultiPolygon):
        if not isinstance(mp, list):
            if mp.geom_type == 'Polygon':
                mp = [mp]
        while True:
            for i, geom in enumerate(mp):
                if len(geom.interiors) > 0:
                    # then cut it
                    nearest_points = shapely.ops.nearest_points(geom.interiors[0], geom.exterior)
                    # splitter = shapely.geometry.LineString([geom.interiors[0].coords[0], geom.exterior.coords[0]]).buffer(1)
                    splitter = shapely.geometry.LineString(list(nearest_points)).buffer(0.1)
                    # split_geom = shapely.ops.split(geom, splitter)
                    split_geom = geom.difference(splitter)
                    # TODO: shoudn't this be
                    # shapely.ops.unary_union mp[0:i] + [split_geom] + mp[i+1:] ?!?!
                    mp = shapely.ops.unary_union([x for x in [mp[0:i], split_geom, mp[i + 1:]] if not x == []])
                    if mp.geom_type == 'Polygon':
                        mp = [mp]
                    break
            else:
                break

        # either this will not halt or we will have properly splitted polygons
        for geom in mp:
            assert len(geom.interiors) == 0, "Something was not right during the splitting"
        out_geoms = []
        for geom in mp:
            if len(geom.exterior.coords) < 3:
                continue
            else:
                out_geoms.append(geom)
        return out_geoms

def merge_regions(content: AnalyzedContent) -> AnalyzedContent:
    @dataclass
    class DetectedLine:
        id: int
        polygon: Polygon
        polygon_coords: list
        baseline: list
        bbox: BBox
        cutout: list
        buffered_polygon: Polygon = None
        buffered_bbox = None

    buffer_amount = 2
    all_lines = []
    for r in content.regions:
        for bl, lp, co in zip(r.baselines, r.lines_polygons, r.cutouts):
            all_lines.append(DetectedLine(len(all_lines) + 1,Polygon(lp),lp,bl, BBox.from_polygon(lp), co))

    # buffer the polygon
    for r in all_lines:
        r.buffered_polygon = r.polygon.buffer(buffer_amount)
        r.buffered_bbox = BBox.from_polygon(r.buffered_polygon.exterior.coords)

    # make a union find data structure to find the regions
    uf = UnionFind.from_count(len(all_lines))

    # find all the intersections
    for ia, a in enumerate(all_lines):
        for ib, b in enumerate(all_lines):
            if uf.find(ia) == uf.find(ib): continue
            #if a.buffered_bbox.intersects(b.buffered_bbox):
            if a.buffered_polygon.intersects(b.buffered_polygon): # Only use b.polygon to prevent touching polygons from unioning
                uf.union2(ia, ib)

    lines_grouped = []
    for g in uf.get_sets().values():
        lg = [all_lines[i] for i in g]
        lines_grouped.append(lg)

    final_regions = []
    for g in lines_grouped:
        region = AnalyzedRegion(bbox=None,
                                baselines=[x.baseline for x in g],
                                region_polygon=None,
                                lines_polygons=[x.polygon_coords for x in g],
                                cutouts=[x.cutout for x in g])
        # make the unified region polygon
        if len(g) == 1:
            merged_poly = g[0].buffered_polygon
        else:
            merged_poly = shapely.ops.unary_union([Polygon(x.buffered_polygon.exterior).buffer(0) for x in g])
            merged_poly = merged_poly.buffer(0)
            if merged_poly.geom_type == "MultiPolygon":
                p1,p2 = shapely.ops.nearest_points(merged_poly.geoms[0],merged_poly.geoms[1])
                print(p1,p2)
            while isinstance(merged_poly, shapely.geometry.MultiPolygon):
                merged_poly = shapely.ops.unary_union(merged_poly)

        pc = [(x[0], x[1]) for x in merged_poly.exterior.coords]
        assert len(pc) > 2
        region.region_polygon = pc
        final_regions.append(region)

    return AnalyzedContent(regions=final_regions,
                           baselines=list(itertools.chain.from_iterable([x.baselines for x in final_regions])),
                           lines_polygons=list(itertools.chain.from_iterable(x.lines_polygons for x in final_regions)))


def process_layout(prediction: PredictionResult, scaled_image: SourceImage, process_pool, settings:LayoutProcessingSettings) -> AnalyzedContent:
    if settings.layout_method == LayoutProcessingMethod.LINES_ONLY:
        return process_layout_linesonly(prediction, scaled_image, process_pool, settings)
    elif settings.layout_method in {LayoutProcessingMethod.ANALYSE, LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP, LayoutProcessingMethod.SCHNIPSCHNIP_REGIONSONLY}:
        analysed_content = process_layout_analyse(prediction, scaled_image, process_pool, settings)
        if settings.layout_method == LayoutProcessingMethod.SCHNIPSCHNIP_REGIONSONLY:
            return merge_regions(analysed_content)
        else:
            return analysed_content
    elif settings.layout_method in {LayoutProcessingMethod.FULL, LayoutProcessingMethod.FULL_REGIONSONLY}:
        analyzed_content = process_layout_full(prediction, scaled_image, process_pool, settings)
        if settings.layout_method == LayoutProcessingMethod.FULL_REGIONSONLY:
            return merge_regions(analyzed_content)
        else:
            return analyzed_content
