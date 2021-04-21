from dataclasses import dataclass

from PIL import ImageDraw
from numba import jit

from segmentation.postprocessing.baselines_util import make_baseline_continous
from segmentation.postprocessing.data_classes import PredictionResult
import numpy as np
from typing import List, Tuple

from segmentation.postprocessing.util import show_images


@dataclass
class LabeledBaseline:
    bl: List
    label: int
    topline: List

@dataclass
class BaselinePlot:
    baselines: List[LabeledBaseline]
    labeled: np.ndarray

    @staticmethod
    def from_baselines(baselines: List, image_w: int, image_h: int):
        labeled = np.zeros(shape=(image_h, image_w),dtype=np.int)
        labeled_baselines: List[LabeledBaseline] = []
        for i, bl in enumerate(baselines, start=1):
            labeled_baselines.append(LabeledBaseline(bl, i))
            bl_c = make_baseline_continous(bl)
            for x,y in bl_c:
                labeled[y,x] = i
        return BaselinePlot(labeled_baselines, labeled)

    def find_first_below(self, x: int, y: int) -> Tuple[List, Tuple[int, int]]:
        labeled: np.ndarray = self.labeled
        # look down starting from y+1,x to find first baseline and return that line and the intersection point
        for y in range(y+1, self.labeled.shape[0]):
            if labeled[y,x] != 0:
                return self.baselines[labeled[y,x]-1], (x,y)
        return None, None

    def find_first_above(self, x: int, y: int) -> Tuple[List, Tuple[int, int]]:
        labeled: np.ndarray = self.labeled
        # look down starting from y+1,x to find first baseline and return that line and the intersection point
        for y in range(y -1, -1, -1):
            if labeled[y, x] != 0:
                return self.baselines[labeled[y, x] - 1], (x, y)
        return None, None

    def find_baselines_above(self, bl: LabeledBaseline, max_sep: int):
        above = []
        for x,y in bl.bl:
            fbl, pos = self.find_first_above(x,y)
            if fbl and abs(y - pos[1]) <= max_sep:
                above.append(fbl)
                continue # we do not want to add this baseline multiple times
        return above

    def find_baselines_below(self, bl: LabeledBaseline, max_sep: int):
        below = []
        for x, y in bl.bl:
            fbl, pos = self.find_first_below(x, y)
            if fbl and abs(y - pos[1]) <= max_sep:
                below.append(fbl)
                continue # dont add baseline multiple times
        return below


@dataclass
class BaselineGraphNode:
    baseline: LabeledBaseline
    above: List['BaselineGraphNode']
    below: List['BaselineGraphNode']


    def _interpolate_merged_line(self, ref_line, tl_xs, tl_ys) -> List[Tuple[int,int]]:
        bl_xs = [r[0] for r in ref_line]
        tl_ys_c = np.interp(bl_xs, tl_xs, tl_ys).tolist()  # do slope compensation at front / end
        return [(x, y) for x, y in zip(bl_xs, tl_ys_c)]

    def get_merged_line_above(self, ref_line: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        # we assume that ref_line is continous
        max_x = max(ref_line[-1][0], max(l.baseline.bl[-1][0] for l in self.above))
        ys = np.full(-1, shape=(len(self.above), max_x))
        for i, al in enumerate(self.above):
            ys[i,al.baseline.bl[0][0]:] = [p[1] for p in al.baseline.bl]
        ym = np.max(ys,axis=0)
        tlx = ym[ref_line[0][0]:]  # cut leading bit
        tl_xs = np.where(tlx >= 0)
        tl_ys = tlx[tl_xs]
        tl_xs += ref_line[0][0]
        return self._interpolate_merged_line(ref_line, tl_xs, tl_ys)

    def get_merged_topline_below(self, ref_line: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        MAX_INT = np.int32(2147483647)
        max_x = max(ref_line[-1][0], max(l.baseline.topline[-1][0] for l in self.above))
        ys = np.full(MAX_INT, shape=(len(self.below), max_x))
        for i, al in enumerate(self.below):
            ys[i, al.baseline.topline[0][0]:] = [p[1] for p in al.baseline.topline]
        ym = np.min(ys, axis=0)
        tlx = ym[ref_line[0][0]:]  # cut leading bit
        tl_xs = np.where(tlx < MAX_INT)

        tl_ys = tlx[tl_xs]
        tl_xs += ref_line[0][0]
        return self._interpolate_merged_line(ref_line, tl_xs, tl_ys)









@dataclass
class BaselineGraph:
    nodes: List[BaselineGraphNode]
    baseline_plot: BaselinePlot
    @staticmethod
    def build_graph(baselines: List[List[Tuple[int, int]]], img_w: int, img_h: int):
        gp = BaselinePlot.from_baselines(baselines, img_w, img_h)
        nodes = []

        for bl in gp.baselines:
            above = gp.find_baselines_above(bl, 100)
            below = gp.find_baselines_below(bl, 100)
            nodes.append(BaselineGraphNode(bl, above, below))
        return BaselineGraph(nodes, gp)

    def visualize(self, base_img):
        # make the labeled image rgb
        if base_img is not None:
            rgb = base_img
            if len(rgb.shape) == 2:
                rgb = np.dstack([rgb] * 3)
        else:
            rgb = np.where(self.baseline_plot.labeled > 0, np.uint8(0), np.uint8(255))
            rgb = np.dstack([rgb]*3)
        from PIL import Image
        im = Image.fromarray(rgb)

        draw = ImageDraw.Draw(im)
        for node in self.nodes:
            for bel in node.below:
                draw.line((node.baseline.bl[0][0], node.baseline.bl[0][1], bel.bl[0][0], bel.bl[0][1]),fill=(255,0,0),width=4)
                for p1,p2 in zip(node.baseline.bl, node.baseline.bl[1:]):
                    draw.line((p1[0],p1[1],p2[0],p2[1]), fill=(0,0,255),width=4)
                    for p1, p2 in zip(node.baseline.topline, node.baseline.topline[1:]):
                        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 255, 255), width=4)
        show_images([np.array(im)], interpolation="bilinear")
        #return rgb








