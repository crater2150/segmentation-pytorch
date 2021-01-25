import ctypes
import itertools
import json
import multiprocessing
import multiprocessing.sharedctypes

from collections import namedtuple, defaultdict, deque
from typing import List, Tuple
import heapq

from segmentation.postprocessing.baselines_util import make_baseline_continous, simplify_baseline
from segmentation.postprocessing.data_classes import PredictionResult, BaselineResult, BboxCluster
from segmentation.postprocessing.debug_draw import DebugDraw
from segmentation.postprocessing.layout_analysis import get_top_of_baselines_improved, get_top, get_top_of_baselines
from segmentation.preprocessing.source_image import SourceImage

import numpy as np

# find a path that divides the two baselines
from segmentation.util import PerformanceCounter, logger

QueueElem = namedtuple("QueueElem", "f d point parent")
# f is dist + heur, d is distance, n is node, p is parent

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
        node = node.parent
    return list(reversed(path_reversed))


def find_dividing_path(inv_binary_img: np.ndarray, cut_above, cut_below) -> List:
    # assert, that both cut_baseline is a list of lists and cut_topline is also a list of lists

    tl, bl = extend_baselines(cut_above, cut_below)

    def find_children(cur_node, x_start):
        x = cur_node[0]
        xi = x - x_start
        y1, y2 = tl[xi][1], bl[xi][1] # TODO: shouldn't this be +1 ?
        if y1 > y2: y1, y2 = y2, y1
        #for y in range(max(tl[xi][1], cur_node[1] - 1), min(cur_node[1]+2, bl[xi][1] + 1)):
        for y in range(y1,y2+1):
            yield (x + 1,y)

    # use dijkstras algorithm to find the shortest dividing path
    # dummy start point
    end_x = int(bl[-1][0])

    # adjust the constant factor for different cutting behaviours
    def dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # return 1+ abs( p2[1] - p1[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # bottom one is ~ 4% faster

    def H_fn(p):
        return end_x - p[0]
    #H_fn = lambda p: end_x - p[0]
    nodeset = dict()

    start_point = (bl[0][0] - 1, int(abs(bl[0][1] + tl[0][1]) / 2))
    start_elem = QueueElem(H_fn(start_point), 0, start_point, None)
    Q: List[QueueElem] = [start_elem]

    #distance = defaultdict(lambda: 2147483647)
    visited = set()
    while len(Q) > 0:
        node = heapq.heappop(Q)
        if node.point in visited: continue # if we already visited this node
        visited.add(node.point)
        if node.point[0] == end_x:
            return make_path(node, start_elem)
        for child in find_children(node.point, start_point[0]):
            if child in visited: continue
            # calculate distance and heur
            d = dist_fn(node.point,child) + node.d
            h = H_fn(child)

            found = nodeset.get(child)
            if found is None:
                nodeset[child] = child
            else:
                child = found

            heapq.heappush(Q, QueueElem(f=h+d, d=d, point=child, parent=node))
    logger.error("Cannot run A*")
    logger.error(f"Cut above: {cut_above}")
    logger.error(f"Cut below: {cut_below}")

    raise RuntimeError("Unreachable")

QuadrupletElem = namedtuple("QuadruppletEleem", "bl_top tl_cur bl_cur tl_bot")

CutoutElem = namedtuple("CutoutElem", "bl tc bc")

def schnip_schnip_algorithm(scaled_image: SourceImage, prediction: PredictionResult, bbox: BboxCluster, process_pool :multiprocessing.Pool = None) -> List[CutoutElem]:
    baselines_cont = [make_baseline_continous(bl) for bl in prediction.baselines]
    if prediction.toplines is not None:
        toplines_cont = [make_baseline_continous(bl) if bl is not None else None for bl in prediction.toplines]
    else:
        toplines_cont = [None for bl in baselines_cont]
    if len(bbox.baselines) == 0: return []

    def blhash(bl):
        return json.dumps([(int(a[0]),int(a[1])) for a in bl])
    inv_binary =  1 - scaled_image.binarized()
    calculated_tops = [np.array(x[1]).tolist() for x in get_top_of_baselines(baselines_cont,inv_binary, process_pool=process_pool)]

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
        cuts.append(find_dividing_path(inv_binary,pt[0], pb[1]))
        # draw_bls([pt[0], pb[1], cuts[-1]])
    for pair, tc, bc in zip(pairs[1:], cuts, cuts[1:]):
        bl_cutouts.append(CutoutElem(bl=pair[0], tc=tc, bc=bc))

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
    first_tc = find_dividing_path(inv_binary, moveline(pairs[0][0], (-2)*height_diff, int(inv_binary.shape[0])), pairs[0][1])
    if len(pairs) > 1:
        #first_bc = find_dividing_path(inv_binary, pairs[0][0], pairs[1][1])
        first_bc = cuts[0]
    else:
        first_bc = find_dividing_path(inv_binary, pairs[0][0], moveline(pairs[0][0], height_diff, int(inv_binary.shape[0])))

    bl_cutouts = [CutoutElem(pairs[0][0], tc=first_tc, bc=first_bc)] + bl_cutouts

    if len(pairs) > 1:

        # find a bottom line for the last baseline
        #bot_tc = find_dividing_path(inv_binary, pairs[-2][0], pairs[-1][1])
        bot_tc = cuts[-1]
        height_diff = calculate_height_diff(pairs[-1][0], pairs[-1][1])
        bot_bc = find_dividing_path(inv_binary, pairs[-1][0], moveline(pairs[-1][0], height_diff, int(inv_binary.shape[0])))
        bl_cutouts = bl_cutouts + [CutoutElem(pairs[-1][0],tc=bot_tc, bc=bot_bc)]

    return bl_cutouts


def cutout_to_polygon(cutout, scaled_image):
    def draw_bls(bls):
        dd = DebugDraw(scaled_image)
        dd.draw_baselines(bls)

        from matplotlib import pyplot
        pyplot.imshow(SourceImage(dd.image()).array())
        pyplot.show()

    bl, tc, bc = cutout
    tc = [x for x in tc if x[0] >= bl[0][0] and x[0] <= bl[-1][0]]
    bc = [x for x in bc if x[0] >= bl[0][0] and x[0] <= bl[-1][0]]
    stc = simplify_baseline(tc)
    sbc = simplify_baseline(bc)

    poly = stc + [bl[-1]] + list(reversed(sbc)) + [bl[0]]
    #draw_bls([bl, tc, bc])
    #draw_bls([poly])
    return poly











