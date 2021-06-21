from dataclasses import dataclass, field

from PIL import ImageDraw

from segmentation.postprocessing.baselines_util import make_baseline_continous
from segmentation.postprocessing.data_classes import PredictionResult, MovedBaselineTop
import numpy as np
from typing import List, Tuple, Set, Optional

from segmentation.postprocessing.layout_processing import AnalyzedContent
from segmentation.postprocessing.util import show_images

from segmentation.preprocessing.source_image import SourceImage
from segmentation.util import PerformanceCounter

Point = Tuple[int,int]
LinePoints = List[Point]

@dataclass
class LabeledLine:
    points: LinePoints
    label: int

class LabeledLineNFindAcc:
    def __init__(self, lines: List[LabeledLine], dist_image: np.ndarray):
        self.lines = lines
        self.dist_image = dist_image

    @staticmethod
    def from_lines(lines: List[LinePoints], image_w: int, image_h: int):
        dist_image = np.zeros(shape=(image_h, image_w),dtype=np.int32)
        labeled_lines: List[LabeledLine] = []
        for i, bl in enumerate(lines, start=1):
            labeled_lines.append(LabeledLine(bl, i))
            bl_c = make_baseline_continous(bl)
            for x, y in bl_c:
                dist_image[y,x] = i
        return LabeledLineNFindAcc(labeled_lines, dist_image)

    def find_first_below(self, x: int, y: int) -> Tuple[LabeledLine, Tuple[int, int]]:
        labeled: np.ndarray = self.dist_image
        # look down starting from y+1,x to find first baseline and return that line and the intersection point
        for y in range(y+1, self.dist_image.shape[0]):
            if labeled[y,x] != 0:
                return self.lines[labeled[y, x] - 1], (x, y)
        return None, None

    def find_first_above(self, x: int, y: int) -> Tuple[LabeledLine, Tuple[int, int]]:
        labeled: np.ndarray = self.dist_image
        # look down starting from y+1,x to find first baseline and return that line and the intersection point
        for y in range(y -1, -1, -1):
            if labeled[y, x] != 0:
                return self.lines[labeled[y, x] - 1], (x, y)
        return None, None

    def find_baselines_above(self, bl: LabeledLine, max_sep: int) -> List[LabeledLine]:
        above_labels = set()
        for x,y in bl.points:
            fbl, pos = self.find_first_above(x,y)
            if fbl and abs(y - pos[1]) <= max_sep:
                above_labels.add(fbl.label)
                continue # we do not want to add this baseline multiple times
        return [self.lines[i-1] for i in above_labels]

    def find_baselines_below(self, bl: LabeledLine, max_sep: int) -> List[LabeledLine]:
        below_labels = set()
        for x, y in bl.points:
            fbl, pos = self.find_first_below(x, y)
            if fbl and abs(y - pos[1]) <= max_sep:
                below_labels.add(fbl.label)
                continue # dont add baseline multiple times
        return [self.lines[i-1] for i in below_labels]

    def get_top_sensitive(self, baseline: List[Tuple[int,int]], inverted_binary, threshold=0.2, max_steps=None, disable_now=False) -> MovedBaselineTop:
        if max_steps is None:
            max_steps = int(inverted_binary.shape[0])
        x, y = zip(*baseline)
        indices = (np.array(y), np.array(x))
        max_black_pixels = 0
        height = 0
        for i in range(min(np.min(indices[0]), max_steps)):  # do at most min_height steps, so that min(y) == 0
            indices = (indices[0] - 1, indices[1])
            # if we hit another baseline, or if we are too high, abort!
            if np.any(self.dist_image[indices[0],indices[1]] != 0) or \
                    np.min(indices[0]) == 0:
                height = height + 1
                return MovedBaselineTop(baseline, list(zip(indices[1].tolist(), indices[0].tolist())), height)
            now = np.sum(inverted_binary[indices])
            if (max_black_pixels * threshold > now or (now <= 5 and not disable_now)) and height > 5:
                break
            height = height + 1
            max_black_pixels = now if now > max_black_pixels else max_black_pixels
        return MovedBaselineTop(baseline, list(zip(indices[1].tolist(), indices[0].tolist())), height)



@dataclass
class BaselineGraphNode:
    baseline: LabeledLine
    topline: LabeledLine = None
    label : int = None
    above: List['BaselineGraphNode'] = field(default_factory=list)
    below: List['BaselineGraphNode'] = field(default_factory=list)

    def get_above_labels_set(self):
        return set(x.label for x in self.above)

    def get_below_label_set(self):
        return set(x.label for x in self.below)

    @staticmethod
    def _interpolate_merged_line(ref_line, tl_xs, tl_ys) -> List[Tuple[int, int]]:
        bl_xs = [r[0] for r in ref_line]
        tl_ys_c = np.interp(bl_xs, tl_xs, tl_ys).tolist()  # do slope compensation at front / end
        return [(round(x), round(y)) for x, y in zip(bl_xs, tl_ys_c)]

    def get_merged_line_above(self, ref_line: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        # we assume that ref_line is continous
        max_x = max(ref_line[-1][0], max(l.baseline.points[-1][0] for l in self.above))
        ys = np.full(shape=(len(self.above), max_x+1), fill_value=-1)
        for i, al in enumerate(self.above):
            ys[i, [p[0] for p in al.baseline.points]] = [p[1] for p in al.baseline.points]
        ym = np.max(ys,axis=0)
        tlx = ym[ref_line[0][0]:]  # cut leading bit
        tl_xs = np.where(tlx >= 0)[0]
        tl_ys = tlx[tl_xs]
        tl_xs += ref_line[0][0]
        return BaselineGraphNode._interpolate_merged_line(ref_line, tl_xs, tl_ys)

    def get_merged_topline_below(self, ref_line: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        MAX_INT = np.int32(2147483647)
        max_x = max(ref_line[-1][0], max(l.topline.points[-1][0] for l in self.below))
        ys = np.full(shape=(len(self.below), max_x+1), fill_value=MAX_INT)
        for i, al in enumerate(self.below):
            ys[i, [p[0] for p in al.baseline.points]] = [p[1] for p in al.topline.points]
        ym = np.min(ys, axis=0)
        tlx = ym[ref_line[0][0]:]  # cut leading bit
        tl_xs = np.where(tlx < MAX_INT)[0]

        tl_ys = tlx[tl_xs]
        tl_xs += ref_line[0][0]
        return BaselineGraphNode._interpolate_merged_line(ref_line, tl_xs, tl_ys)



@dataclass
class BaselineGraph:
    nodes: List[BaselineGraphNode]
    baseline_acc: LabeledLineNFindAcc
    topline_acc: LabeledLineNFindAcc

    def node_by_label(self, label:int ):
        return self.nodes[label-1]

    @staticmethod
    def build_graph(baselines: List[LinePoints], toplines: Optional[List[LinePoints]], img_w: int, img_h: int):
        bl_acc = LabeledLineNFindAcc.from_lines(baselines, img_w, img_h)
        tl_acc = LabeledLineNFindAcc.from_lines(toplines, img_w, img_h)

        nodes = [BaselineGraphNode(baseline=bl, topline=tl, label=bl.label) for bl, tl in zip(bl_acc.lines, tl_acc.lines)]
        assert all(i == n.baseline.label and i == n.topline.label for i, n in enumerate(nodes, start=1))

        for node in nodes:
            above = bl_acc.find_baselines_above(node.baseline, 200)
            below = bl_acc.find_baselines_below(node.baseline, 200)

            for i in above:
                node.above.append(nodes[i.label-1])
            for i in below:
                node.below.append(nodes[i.label-1])

        return BaselineGraph(nodes, baseline_acc=bl_acc, topline_acc=tl_acc)

    def is_symmetrical(self):
        for node in self.nodes:
            for above in node.above:
                if node.label not in above.get_below_label_set(): return False
            for below in node.below:
                if node.label not in below.get_above_labels_set(): return False
        return True

    def visualize(self, base_img):
        # make the labeled image rgb
        if base_img is not None:
            rgb = base_img
            if len(rgb.shape) == 2:
                rgb = np.dstack([rgb] * 3)
        else:
            rgb = np.where(self.baseline_acc.dist_image > 0, np.uint8(0), np.uint8(255))
            rgb = np.dstack([rgb]*3)
        from PIL import Image
        im = Image.fromarray(rgb)

        draw = ImageDraw.Draw(im)
        for node in self.nodes:
            for bel in node.below:
                draw.line((node.baseline.points[0][0], node.baseline.points[0][1], bel.baseline.points[0][0], bel.baseline.points[0][1]), fill=(255, 0, 0), width=4)
            for p1,p2 in zip(node.baseline.points, node.baseline.points[1:]):
                draw.line((p1[0],p1[1],p2[0],p2[1]), fill=(0,0,255),width=4)

            if node.above:
                merged_tl = node.get_merged_line_above(node.topline.points)

                for p1, p2 in zip(merged_tl, merged_tl[1:]):
                    draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 255, 255), width=4)
        show_images([np.array(im)], interpolation="bilinear")
        #return rgb


class MarginaliaSeperator:
    def __init__(self):
        pass

    def detect_marginalia(self, content: AnalyzedContent):
        pass




