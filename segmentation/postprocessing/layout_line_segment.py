import itertools
import heapq
import itertools
import json
import multiprocessing.sharedctypes
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Set

import numpy as np
import scipy
import scipy.ndimage.measurements
import skimage.draw
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union

from segmentation.postprocessing.baseline_graph import BaselineGraph, LabeledBaseline
from segmentation.postprocessing.baselines_util import make_baseline_continous, simplify_baseline
from segmentation.postprocessing.data_classes import PredictionResult, BboxCluster
from segmentation.postprocessing.debug_draw import DebugDraw
from segmentation.postprocessing.layout_analysis import get_top_of_baselines
from segmentation.postprocessing.layout_line_util import _build_bl_growth_img
from segmentation.postprocessing.util import NewImageReconstructor, show_images
from segmentation.preprocessing.source_image import SourceImage
# find a path that divides the two baselines
from segmentation.postprocessing.layout_settings import LayoutProcessingSettings
from segmentation.util import logger

QueueElem = namedtuple("QueueElem", "f d point parent")
# f is dist + heur, d is distance, n is node, p is parent

seq_number = 0
def extend_baselines(line_a: List, line_b: List) -> Tuple[List,List]:
    start_dif = line_b[0][0] - line_a[0][0]
    end_swap = False
    if start_dif < 0:
        start_dif = -start_dif
        end_swap = not end_swap
        line_a, line_b = line_b, line_a

    if start_dif > 0: # line_a is longer, extend line_b
        y = line_b[0][1]
        x_start = line_b[0][0] - start_dif
        line_b = [(x_start + i ,y) for i in range(start_dif)] + line_b



    end_dif = line_b[-1][0] - line_a[-1][0]
    if end_dif < 0:
        end_dif = -end_dif
        end_swap = not end_swap
        line_a, line_b = line_b, line_a

    if end_dif > 0: # line_b is longer, extend line a
        end_y = line_a[-1][1]
        end_start_x = line_a[-1][0]
        line_a = line_a + [(end_start_x + i + 1, end_y) for i in range(end_dif)]

    if end_swap:
        line_a, line_b = line_b, line_a

    if not (line_a[0][0] == line_b[0][0] and line_a[-1][0] == line_b[-1][0]):
        line_a_r = list(reversed(line_a))
        line_b_r = list(reversed(line_b))
        raise RuntimeError("oops")
    return line_a, line_b

def make_path(target_node:QueueElem, start_node: QueueElem) -> List:
    path_reversed = []
    node = target_node
    while node != start_node:
        #print(node.f, node.d, node.point)
        path_reversed.append(node.point)
        # if we are on home position, no need to move vertically
        if node.point[0] == start_node.point[0] + 1:
            break
        node = node.parent
    return list(reversed(path_reversed))

class DividingPathStartingBias(Enum):
    MID = 0
    TOP = 1
    BOTTOM = 2

def find_dividing_path(inv_binary_img: np.ndarray, cut_above, cut_below, starting_bias = DividingPathStartingBias.MID) -> List:
    # assert, that both cut_baseline is a list of lists and cut_topline is also a list of lists

    tl, bl = extend_baselines(cut_above, cut_below)
    # see if there is a corridor, if not, push top baseline up by 1 px
    min_channel = 3
    for x, points in enumerate(zip(tl,bl)):
        p1, p2 = points
        if p1[1] > p2[1]:
            tl[x], bl[x] = bl[x],tl[x]
            p1, p2 = tl[x], bl[x]

        if abs(p1[1] - p2[1]) <= min_channel:
            if tl[x][1] > min_channel:
                tl[x] = (tl[x][0], tl[x][1] - min_channel)
            elif bl[x][1] + min_channel < inv_binary_img.shape[0]:
                bl[x] = (bl[x][0], bl[x][1] + min_channel)
            else:
                assert False, "Unlucky"


    """
    def find_children(cur_node, x_start):
        x = cur_node[0]
        xi = x - x_start
        y1, y2 = tl[xi][1], bl[xi][1] # TODO: shouldn't this be +1 ?
        if y1 > y2: y1, y2 = y2, y1
        #for y in range(max(tl[xi][1], cur_node[1] - 1), min(cur_node[1]+2, bl[xi][1] + 1)):
        for y in range(y1,y2+1):
            yield (x + 1,y)
    """
    def find_children_rect(cur_node, x_start):
        x = cur_node[0]
        y = cur_node[1]
        xi = x - x_start
        y1, y2 = tl[xi][1], bl[xi][1]  # TODO: shouldn't this be +1 ?
        # if y1 > y2: y1, y2 = y2, y1
        # if tl[xi+1][1] <= y <= bl[xi+1][1]: yield (x+1,y)
        if tl[xi + 1][1] <= y <= bl[xi + 1][1]: yield (x + 1, y)
        if y > y1: yield (x, y-1)
        if y < y2: yield (x, y+1)

    # use Dijkstra's algorithm to find the shortest dividing path
    # dummy start point
    end_x = int(bl[-1][0])
    assert end_x == int(tl[-1][0]) and end_x == int(bl[-1][0])

    # adjust the constant factor for different cutting behaviours
    def dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # return 1+ abs( p2[1] - p1[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # bottom one is ~ 4% faster

    def H_fn(p):
        return end_x - p[0]
    #H_fn = lambda p: end_x - p[0]
    #nodeset = dict()
    if starting_bias == DividingPathStartingBias.MID:
        start_point = (bl[0][0] - 1, int(abs(bl[0][1] + tl[0][1]) / 2))
    elif starting_bias == DividingPathStartingBias.TOP:
        start_point = (bl[0][0] - 1, tl[0][1] + 1)
    elif starting_bias == DividingPathStartingBias.BOTTOM:
        start_point = (bl[0][0] - 1, bl[0][1] - 1)
    else:
        raise NotImplementedError()
    start_x = bl[0][0] # start_point[0] + 1

    y1, y2 = tl[0][1], bl[0][1]+ 1
    if y1 > y2: y1, y2 = y2,y1
    source_points = [(start_x, y) for y in range(y1, y2)]

    start_elem = QueueElem(0, 0, start_point, None)
    Q: List[QueueElem] = []

    for p in source_points:
        heapq.heappush(Q,QueueElem(d=dist_fn(start_point, p), f=dist_fn(start_point,p ) + H_fn(p),point=p,parent=start_elem))

    #distance = defaultdict(lambda: 2147483647)
    visited = set()
    shortest_found_dist = defaultdict(lambda: 2147483647)
    for elem in Q:
        shortest_found_dist[elem.point] = elem.d
    while len(Q) > 0:
        node = heapq.heappop(Q)
        if node.point in visited: continue # if we already visited this node
        visited.add(node.point)
        if node.point[0] == end_x:
            path = make_path(node, start_elem)
            if False:
                dd = DebugDraw(SourceImage.from_numpy(np.array(255*(1 - inv_binary_img),dtype=np.uint8)))
                dd.draw_baselines([tl, bl, path])
                img = dd.image()
                #global seq_number
                #img.save(f"/tmp/seq{seq_number}.png")
                #seq_number += 1
                #dd.show()
            return path
        for child in find_children_rect(node.point, start_x):
            if child in visited: continue
            # calculate distance and heur
            p1, p2 = node.point, child
            # not having to call a function is ~10 % faster

            d = node.d + abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
            #d = dist_fn(node.point,child) + node.d

            if shortest_found_dist[child] <= d:
                continue  # we already found it

            """
            found = nodeset.get(child)
            if found is None:
                nodeset[child] = child
            else:
                child = found
            """

            # is this path to this node shorter?
            shortest_found_dist[child] = d


            #h = H_fn(child)
            h = end_x - child[0]

            heapq.heappush(Q, QueueElem(f=h+d, d=d, point=child, parent=node))
    logger.error("Cannot run A*")
    logger.error(f"Cut above: {cut_above}")
    logger.error(f"Cut below: {cut_below}")

    # raise RuntimeError("Unreachable")
    # Just use the middle line, to avoid crashing
    logger.error("Using fill path to avoid crashing")
    fill_path = []
    for pa, pb in zip(cut_above, cut_below):
        fill_path.append((pa[0],round((pa[1] + pb[1]) / 2)))
    return fill_path



QuadrupletElem = namedtuple("QuadruppletEleem", "bl_top tl_cur bl_cur tl_bot")

CutoutElem = namedtuple("CutoutElem", "bl tc bc")

def shorten_cutline(cutout_line: List[Tuple], bl : List[Tuple]):
    minb = bl[0][0]
    maxb = bl[-1][0]
    return [co for co in cutout_line if minb <= co[0] <= maxb]

def schnip_schnip_algorithm_old(scaled_image: SourceImage, prediction: PredictionResult, bbox: BboxCluster, settings: LayoutProcessingSettings) -> List[CutoutElem]:
    #himg = _build_bl_growth_img(scaled_image.binarized(), np.array(list(itertools.chain.from_iterable(prediction.baselines))))
    #show_images([himg])
    baselines_cont = [make_baseline_continous(bl) for bl in prediction.baselines]
    if prediction.toplines is not None:
        toplines_cont = [make_baseline_continous(bl) if bl is not None else None for bl in prediction.toplines]
    else:
        toplines_cont = [None for bl in baselines_cont]
    if len(bbox.baselines) == 0: return []

    def blhash(bl):
        return json.dumps([(int(a[0]),int(a[1])) for a in bl])
    inv_binary =  1 - scaled_image.binarized()
    inv_binary_dilated = scipy.ndimage.binary_dilation(inv_binary)
    calculated_tops = [np.array(x[1]).tolist() for x in get_top_of_baselines(baselines_cont,inv_binary, process_pool=None)]

    #with PerformanceCounter("Hashing"):
    bl_to_idx = dict((blhash(bl), i) for (i, bl) in enumerate(baselines_cont))

    pairs = []
    # make sure, that for each baseline, we do have a topline
    for bl in bbox.baselines:
        bl = bl.baseline
        bl_i = bl_to_idx[blhash(bl)]
        # is there a matching topline?
        if False and len(prediction.matching[bl_i]) > 0: # todo remove
            topline = toplines_cont[prediction.matching[bl_i][0]]
        else:
            topline = calculated_tops[bl_i]
        pairs.append((baselines_cont[bl_i], topline))

    # sort the baselines from top to bottom by their average height
    pairs = sorted(pairs, key= lambda pair: sum(p[1] for p in pair[0]) / len(pair[0]))

    # to calculate a line polyon, we need: baseline from the top line, a topline, a baseline, and the topline from the line below
    # because we don't have that for the first and last line, we fake this, for all inner lines, pair it up now

    bl_cutouts = []

    cuts = []
    def draw_bls(bls):
        dd = DebugDraw(scaled_image)
        dd.draw_baselines(bls)

        from matplotlib import pyplot
        pyplot.imshow(SourceImage(dd.image()).array())
        pyplot.show()

    #with PerformanceCounter("FindCuts"):
    # doing MP here is probably not faster
    for pt, pb in zip(pairs, pairs[1:]):
        # draw_bls([pt[0],pb[1]])
        cuts.append(find_dividing_path(inv_binary_dilated,pt[0], pb[1]))
        # draw_bls([pt[0], pb[1], cuts[-1]])
    for pair, tc, bc in zip(pairs[1:], cuts, cuts[1:]):
        bl_cutouts.append(CutoutElem(bl=pair[0], tc=shorten_cutline(tc, pair[0]), bc=shorten_cutline(bc, pair[0])))

    """
    for pair_top, pair_cur, pair_bottom in zip(pairs, pairs[1:], pairs[2:]):
        quad = QuadrupletElem(pair_top[0], pair_cur[1], pair_cur[0], pair_bottom[1])

        top_cutout = find_dividing_path(inv_binary,quad.bl_top, quad.tl_cur)
        bottom_cutout = find_dividing_path(inv_binary, quad.bl_cur, quad.tl_bot)

        bl_cutouts.append(CutoutElem(bl=quad.bl_cur,tc=top_cutout, bc=bottom_cutout))        
    """



    #draw_bls([quad.bl_top, quad.tl_cur, top_cutout])
    #draw_bls([quad.bl_cur, quad.tl_bot, bottom_cutout])

    def moveline(bl, x, img_height):
        min_y = min(p[1] for p in bl)
        max_y = max(p[1] for p in bl)
        if min_y + x < 0:
            x = -min_y
        if max_y + x >= img_height:
            x = (img_height - max_y - 1)

        return [(p[0], p[1] + x) for p in bl]

    def calculate_height_diff(a,b):
        avg_height_topline = sum(p[1] for p in a) / len(a)
        avg_height_baseline = sum(p[1] for p in b) / len(b)
        return round(abs(avg_height_topline - avg_height_baseline))

    # fix first line
    height_diff = calculate_height_diff(pairs[0][0], pairs[0][1])
    first_tc = find_dividing_path(inv_binary_dilated,
                                  cut_above=moveline(pairs[0][0], (settings.schnip_schnip_height_diff_factor)*height_diff, int(inv_binary.shape[0])),
                                  cut_below=moveline(pairs[0][1], int(height_diff * 0.3), int(inv_binary.shape[0])),
                                  starting_bias=DividingPathStartingBias.BOTTOM)
    if len(pairs) > 1:
        #first_bc = find_dividing_path(inv_binary, pairs[0][0], pairs[1][1])
        first_bc = cuts[0]
    else:
        first_bc = find_dividing_path(inv_binary_dilated,
                                      pairs[0][0], moveline(pairs[0][0], height_diff, int(inv_binary.shape[0])),
                                      starting_bias=DividingPathStartingBias.TOP)

    bl_cutouts = [CutoutElem(pairs[0][0], tc=shorten_cutline(first_tc, pairs[0][0]), bc=shorten_cutline(first_bc, pairs[0][0]))] + bl_cutouts

    if len(pairs) > 1:

        # find a bottom line for the last baseline
        #bot_tc = find_dividing_path(inv_binary, pairs[-2][0], pairs[-1][1])
        bot_tc = cuts[-1]
        height_diff = calculate_height_diff(pairs[-1][0], pairs[-1][1])
        bot_bc = find_dividing_path(inv_binary_dilated,
                                    pairs[-1][0], moveline(pairs[-1][0], height_diff, int(inv_binary.shape[0])),
                                    starting_bias=DividingPathStartingBias.TOP)
        bl_cutouts = bl_cutouts + [CutoutElem(pairs[-1][0],tc=shorten_cutline(bot_tc, pairs[-1][0]), bc=shorten_cutline(bot_bc,pairs[-1][0]))]

    def fix_cutout_overgrow(cutout: CutoutElem):
        # calculate the median cutout height
        heights = [abs(a[1] - b[1]) for (a,b) in zip(cutout.bc, cutout.tc)]
        tc_off = [abs(a[1] - b[1]) for (a,b) in zip(cutout.tc, cutout.bl)]
        bc_off = [abs(a[1] - b[1]) for (a, b) in zip(cutout.bc, cutout.bl)]
        med_tc_off = float(np.median(tc_off))
        med_bc_off = float(np.median(bc_off))

        med_height = float(np.median(heights))
        ntc = []
        nbc = []
        for tc, bl, bc, height, tco, bco in zip(cutout.tc, cutout.bl, cutout.bc, heights, tc_off, bc_off):
            if height > med_height * 1.25:
                # check if we broke out to the top or the bottom
                if abs(tc[1] - bl[1]) - med_tc_off > abs(bc[1] - bl[1]) - med_bc_off:
                    # we broke from the tc
                    ntc.append((tc[0],bl[1] - med_tc_off))
                    nbc.append(bc)
                else:
                    ntc.append(tc)
                    nbc.append((bc[0],bl[1] + med_bc_off))
            else:
                ntc.append(tc)
                nbc.append(bc)
        return CutoutElem(cutout.bl, ntc, nbc)

    #bl_cutouts = [fix_cutout_overgrow(c) for c in bl_cutouts]



    return bl_cutouts

def schnip_schnip_algorithm(scaled_image: SourceImage, prediction: PredictionResult, bbox: BboxCluster, settings: LayoutProcessingSettings) -> List[CutoutElem]:
    blg = BaselineGraph.build_graph(prediction.baselines, scaled_image.array().shape[1], scaled_image.array().shape[0])
    blg.visualize(scaled_image.array())

    baselines_cont = [x.baseline.bl for x in blg.nodes]

    inv_binary =  1 - scaled_image.binarized()
    inv_binary_dilated = scipy.ndimage.binary_dilation(inv_binary)
    calculated_tops = [np.array(x[1]).tolist() for x in get_top_of_baselines(baselines_cont,inv_binary, process_pool=None)]
    

    #with PerformanceCounter("Hashing"):
    bl_to_idx = dict((blhash(bl), i) for (i, bl) in enumerate(baselines_cont))

    pairs = []
    # make sure, that for each baseline, we do have a topline
    for bl in bbox.baselines:
        bl = bl.baseline
        bl_i = bl_to_idx[blhash(bl)]
        # is there a matching topline?
        if False and len(prediction.matching[bl_i]) > 0: # todo remove
            topline = toplines_cont[prediction.matching[bl_i][0]]
        else:
            topline = calculated_tops[bl_i]
        pairs.append((baselines_cont[bl_i], topline))

    # sort the baselines from top to bottom by their average height
    pairs = sorted(pairs, key= lambda pair: sum(p[1] for p in pair[0]) / len(pair[0]))

    # to calculate a line polyon, we need: baseline from the top line, a topline, a baseline, and the topline from the line below
    # because we don't have that for the first and last line, we fake this, for all inner lines, pair it up now

    bl_cutouts = []

    cuts = []
    def draw_bls(bls):
        dd = DebugDraw(scaled_image)
        dd.draw_baselines(bls)

        from matplotlib import pyplot
        pyplot.imshow(SourceImage(dd.image()).array())
        pyplot.show()

    #with PerformanceCounter("FindCuts"):
    # doing MP here is probably not faster
    for pt, pb in zip(pairs, pairs[1:]):
        # draw_bls([pt[0],pb[1]])
        cuts.append(find_dividing_path(inv_binary_dilated,pt[0], pb[1]))
        # draw_bls([pt[0], pb[1], cuts[-1]])
    for pair, tc, bc in zip(pairs[1:], cuts, cuts[1:]):
        bl_cutouts.append(CutoutElem(bl=pair[0], tc=shorten_cutline(tc, pair[0]), bc=shorten_cutline(bc, pair[0])))

    """
    for pair_top, pair_cur, pair_bottom in zip(pairs, pairs[1:], pairs[2:]):
        quad = QuadrupletElem(pair_top[0], pair_cur[1], pair_cur[0], pair_bottom[1])

        top_cutout = find_dividing_path(inv_binary,quad.bl_top, quad.tl_cur)
        bottom_cutout = find_dividing_path(inv_binary, quad.bl_cur, quad.tl_bot)

        bl_cutouts.append(CutoutElem(bl=quad.bl_cur,tc=top_cutout, bc=bottom_cutout))        
    """



    #draw_bls([quad.bl_top, quad.tl_cur, top_cutout])
    #draw_bls([quad.bl_cur, quad.tl_bot, bottom_cutout])

    def moveline(bl, x, img_height):
        min_y = min(p[1] for p in bl)
        max_y = max(p[1] for p in bl)
        if min_y + x < 0:
            x = -min_y
        if max_y + x >= img_height:
            x = (img_height - max_y - 1)

        return [(p[0], p[1] + x) for p in bl]

    def calculate_height_diff(a,b):
        avg_height_topline = sum(p[1] for p in a) / len(a)
        avg_height_baseline = sum(p[1] for p in b) / len(b)
        return round(abs(avg_height_topline - avg_height_baseline))

    # fix first line
    height_diff = calculate_height_diff(pairs[0][0], pairs[0][1])
    first_tc = find_dividing_path(inv_binary_dilated,
                                  moveline(pairs[0][0], (settings.schnip_schnip_height_diff_factor)*height_diff, int(inv_binary.shape[0])), pairs[0][1],
                                  starting_bias=DividingPathStartingBias.BOTTOM)
    if len(pairs) > 1:
        #first_bc = find_dividing_path(inv_binary, pairs[0][0], pairs[1][1])
        first_bc = cuts[0]
    else:
        first_bc = find_dividing_path(inv_binary_dilated,
                                      pairs[0][0], moveline(pairs[0][0], height_diff, int(inv_binary.shape[0])),
                                      starting_bias=DividingPathStartingBias.TOP)

    bl_cutouts = [CutoutElem(pairs[0][0], tc=shorten_cutline(first_tc, pairs[0][0]), bc=shorten_cutline(first_bc, pairs[0][0]))] + bl_cutouts

    if len(pairs) > 1:

        # find a bottom line for the last baseline
        #bot_tc = find_dividing_path(inv_binary, pairs[-2][0], pairs[-1][1])
        bot_tc = cuts[-1]
        height_diff = calculate_height_diff(pairs[-1][0], pairs[-1][1])
        bot_bc = find_dividing_path(inv_binary_dilated,
                                    pairs[-1][0], moveline(pairs[-1][0], height_diff, int(inv_binary.shape[0])),
                                    starting_bias=DividingPathStartingBias.TOP)
        bl_cutouts = bl_cutouts + [CutoutElem(pairs[-1][0],tc=shorten_cutline(bot_tc, pairs[-1][0]), bc=shorten_cutline(bot_bc,pairs[-1][0]))]

    return bl_cutouts


def cutout_to_polygon(cutout: CutoutElem, scaled_image: SourceImage = None) -> List:
    def draw_bls(bls):
        dd = DebugDraw(scaled_image)
        dd.draw_baselines(bls)
        dd.show()

    bl, tc, bc = cutout
    tc = [x for x in tc if x[0] >= bl[0][0] and x[0] <= bl[-1][0]]
    bc = [x for x in bc if x[0] >= bl[0][0] and x[0] <= bl[-1][0]]
    stc = simplify_baseline(tc)
    sbc = simplify_baseline(bc)

    poly = stc + [bl[-1]] + list(reversed(sbc)) + [bl[0]]
    #draw_bls([bl, tc, bc])
    #draw_bls([poly])
    return poly


@dataclass
class LinePoly:
    __slots__ = ["poly", "cutout"]
    poly: List
    cutout: CutoutElem

    #def __iter__(self):
    #    return iter(self.poly)

@dataclass
class BBox:
    __slots__ = ["x1", "y1", "x2", "y2"]
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class Contour:
    __slots__ = ["label", "bbox", "height", "width"]
    label: int
    bbox: BBox
    height: int
    width: int


class PageContours:
    def __init__(self, image: SourceImage, dilation_amount=0):
        self.binarized = image.binarized() == 0
        if dilation_amount > 0:
            #self.binarized = scipy.ndimage.binary_dilation(self.binarized)
            self.binarized = scipy.ndimage.binary_closing(self.binarized)
        labled, count = scipy.ndimage.measurements.label(self.binarized,np.array([[1,1,1]]*3)) # 8 connectivity
        self.labeled = labled
        self.count = int(count)

        objs = scipy.ndimage.measurements.find_objects(labled)
        contours = [Contour(0,BBox(0,0,0,0),0,0)] # Background
        for label, (ys,xs) in enumerate(objs, start=1):
            bbox = BBox(x1=int(xs.start), y1=int(ys.start), x2=int(xs.stop), y2=int(ys.stop)) # TODO: stop + 1?
            contours.append(Contour(label,bbox,bbox.y2 - bbox.y1, bbox.x2 - bbox.x1))
        self.contours = contours

    def __iter__(self):
        return iter(self.contours)

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, item) -> Contour:
        return self.contours.__getitem__(item)

    def get_labeled_slice_for_contour(self, label: int) -> np.ndarray:
        cc = self[label]
        bb = cc.bbox
        return self.labeled[bb.y1 : bb.y2, bb.x1 : bb.x2]

    def find_labels_intersecting_cutout(self, co: CutoutElem):
        poly = cutout_to_polygon(co)
        """
        r = []
        c = []
        r += [b[1] for b in co.bc]
        c += [b[0] for b in co.bc]
        r += [t[1] for t in co.tc]
        c += [t[0] for t in co.tc]
        """
        r = [p[1] for p in poly]
        c = [p[0] for p in poly]
        return self.find_labels_intersecting_polygon(r,c)

    def find_labels_intersecting_polygon(self, r, c):
        points = skimage.draw.polygon(r, c)
        # find all labels at these points
        labels = np.unique(self.labeled[points])
        return set(int(x) for x in labels if x > 0)



IntersectionResult = namedtuple("IntersectionResult", "line_id label")

LineSegment = namedtuple("LineSegment", "x1 y1 x2 y2")


def find_intersecting_labels(lines: List[LineSegment], contours: PageContours) -> Set[int]:
    # create a new image and put the
    intersections = set()
    for li, l in enumerate(lines):
        intersects = contours.labeled[skimage.draw.line(l.y1, l.x1, l.y2, l.x1)]
        for i in intersects:
            if i > 0: intersections.add(i)

    return intersections


"""def find_intersecting_labels(l: LineSegment, contours:PageContours):
    intersects = contours.labeled[skimage.draw.line(l.y1, l.x1, l.y2, l.x1)]
    return set(int(i) for i in intersects)
"""


def cutout_get_matched_pairs(cutout: CutoutElem):
    top_lku = dict(cutout.tc)
    bot_lku = dict(cutout.bc)

    x_intersect = sorted(list(set(top_lku.keys()).intersection(set(bot_lku.keys()))))
    return [(x, top_lku[x], bot_lku[x]) for x in x_intersect]


def cutout_average_height(trips: List[Tuple]) -> float:
    if len(trips) == 0:
        return 1
    sum_h = sum(abs(x[2] - x[1]) for x in trips)
    return sum_h / len(trips)


def get_contour_convex(relevant: List[int], contours: PageContours) -> List[Tuple]:
    points = []
    for cc in relevant:
        slc = contours.get_labeled_slice_for_contour(cc)
        contour = contours[cc]
        arr_y, arr_x = (slc == cc).nonzero()
        arr_y += contour.bbox.y1
        arr_x += contour.bbox.x1
        points.extend(zip(arr_x.tolist(),arr_y.tolist()))
    hull = ConvexHull(points)
    poly = list(points[i] for i in hull.vertices)
    return poly


def fix_cutout_lineendings(cutouts: List[CutoutElem], contours: PageContours) -> List[LinePoly]:
    line_polys = []

    # make a dict that tells us which line_ids are covering (partly?) a contour
    contour_hit_by = defaultdict(set)
    for line_id, co in enumerate(cutouts):
        hit_by = contours.find_labels_intersecting_cutout(co)
        for contour in hit_by:
            contour_hit_by[int(contour)].add(line_id)

    contours_used_up = set()

    for line_id, cutout in enumerate(cutouts):
        cutout_poly = cutout_to_polygon(cutout, None)
        fallback = LinePoly(cutout_poly, cutout)

        try: # TODO: bad hgack
            beg_lines = [LineSegment(cutout.bc[0][0], cutout.bc[0][1], cutout.bl[0][0], cutout.bl[0][1]),
                         LineSegment(cutout.bl[0][0], cutout.bl[0][1], cutout.tc[0][0], cutout.tc[0][1])]
            end_lines = [LineSegment(cutout.bc[-1][0], cutout.bc[-1][1], cutout.bl[-1][0], cutout.bl[-1][1]),
                         LineSegment(cutout.bl[-1][0], cutout.bl[-1][1], cutout.tc[-1][0], cutout.tc[-1][1])]
            logger.info(f"Beg Lines: {beg_lines}\n")
            logger.info(f"End Lines: {end_lines}\n")
            int_labels_beg = find_intersecting_labels(beg_lines, contours)
            int_labels_end = find_intersecting_labels(end_lines, contours)
            logger.info(f"Beg Labels: {int_labels_beg}\n")
            logger.info(f"End Labels: {int_labels_end}\n")
            # if the label is approximately the height of the line
            accepted_labels_beg = []
            accepted_labels_end = []
            # calculate average cutout height
            avg_line_height = cutout_average_height(cutout_get_matched_pairs(cutout))

            accepted_labels_beg = []
            int_labels_beg = list(int_labels_beg)
            int_labels_end = list(int_labels_end)
            for cc in map(lambda x: contours[x], int_labels_beg):
                cc_hit_by = contour_hit_by[cc.label]
                if len(cc_hit_by) > 1 or line_id not in cc_hit_by: continue # do not use this label, it's cursed
                if cc.label in contours_used_up: continue  # dont use this, because it has already been used
                if cc.height <= avg_line_height * 1.1 and \
                  cc.bbox.y2 >= cutout.bl[0][1] - (avg_line_height / 2) and \
                  cc.bbox.y2 <= cutout.bc[0][1] + (avg_line_height) * 0.2: # TODO: do the same for the top cutout (but probably with more offset)
                    logger.info(f"Accepted: {cc} for {line_id}\n")
                    accepted_labels_beg.append(cc.label)

            for cc in map(lambda x: contours[x], int_labels_end):
                if cc.label == 259:
                    a = 1
                cc_hit_by = contour_hit_by[int(cc.label)]
                if len(cc_hit_by) > 1 or line_id not in cc_hit_by: continue  # do not use this label, it's cursed
                if cc.label in contours_used_up: continue # dont use this, because it has already been used
                if cc.height <= avg_line_height * 1.1 and \
                  cc.bbox.y2 >= cutout.bl[-1][1] - (avg_line_height / 2) and \
                  cc.bbox.y2 <= cutout.bc[-1][1] + (avg_line_height) * 0.2:
                    logger.info(f"Accepted end: {cc} for {line_id}\n")
                    accepted_labels_end.append(cc.label)

            if len(accepted_labels_end) == 0 and len(accepted_labels_beg) == 0:
                raise Exception("early stopping")# early stopping

            # add the accepted labels to the list of labels that cannot be acced anymore
            contours_used_up = contours_used_up.union(set(accepted_labels_beg).union(accepted_labels_end))

            pll = [Polygon(cutout_poly)]
            assert pll[0].is_valid
            if len(accepted_labels_beg) > 0:
                ch_beg = get_contour_convex(list(set(accepted_labels_beg)), contours)
                pll.append(Polygon(ch_beg).buffer(1,cap_style=2,join_style=3))
            if len(accepted_labels_end) > 0:
                ch_end = get_contour_convex(list(set(accepted_labels_end)), contours)
                pll.append(Polygon(ch_end).buffer(1,cap_style=2,join_style=3))

            poly_union = unary_union(pll)
            if poly_union.geom_type != "Polygon":
                logger.warning(f"Result is not a polygon. Is: {poly_union.geom_type}")
                raise Exception("early stopping")
            points = [(round(x[0]), round(x[1])) for x in poly_union.exterior.coords]

            x,y = poly_union.exterior.xy
            rec = NewImageReconstructor(contours.labeled, contours.count + 1, background_color=(255, 255, 255),
                                        undefined_color=(0, 0, 0))
            for lbl in itertools.chain(int_labels_beg, int_labels_end):
                if lbl in set(accepted_labels_beg).union(set(accepted_labels_end)):
                    rec.label(lbl, (50, 0, 0))
                else:
                    rec.label(lbl, (0, 0, 100))

            """
            from matplotlib import pyplot as plt
            plt.imshow(rec.get_image())
            plt.plot(x,y,color='#ff3333', alpha=1, linewidth=3, solid_capstyle='round')
            cx, cy = [x[0] for x in cutout_poly], [x[1] for x in cutout_poly]
            cx.append(cx[0])
            cy.append(cy[0])
            plt.plot(cx, cy, color="#6699cc", alpha=1, linewidth=3)
            plt.show()
            """
            line_polys.append(LinePoly(points,cutout))

        except:
            line_polys.append(fallback)

    return line_polys









