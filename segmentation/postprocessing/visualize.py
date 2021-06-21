from typing import List, Tuple

import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt

from segmentation.postprocessing.layout_processing import AnalyzedContent


@dataclass
class VisualizerSettings:
    region_labels: bool = False
    textline_labels: bool = True
    baselines: bool = False


@dataclass
class VisRegion:
    poly: List[Tuple]
    color: tuple = (1,0,0)
    tag: str = None
    tag_color: tuple = None

@dataclass
class VisLineSegment:
    line: List[Tuple]
    color: tuple = (0,0,1)

@dataclass
class VisLine:
    segments: List[VisLineSegment]
    tag: str = None
    tag_color: tuple = None


class PageVisualizer:
    def __init__(self, image: np.ndarray):
        self.image :np.ndarray = image
        self.regions : List[VisRegion] = []
        self.lines : List[VisLine] = []

    def add_line(self, line: VisLine):
        self.lines.append(line)

    def add_region(self, region: VisRegion):
        self.regions.append(region)

    def _draw_all(self):
        img = self.image

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        if img.shape[2] == 1:
            # convert to rgb
            img = np.dstack([img, img, img])

        # from matplotlib import pyplot as plt
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.imshow(img)

        region_patches = []
        region_labels = []
        for r in self.regions:
            rarr = np.array(r.poly)
            patch = plt.Polygon(rarr, True, alpha=0.3)
            outline = plt.Polygon(rarr, fill=None, linestyle="--", linewidth="1", edgecolor=(0, 0, 0))
            region_patches.append(patch)
            region_patches.append(outline)

            if r.tag is not None:
                region_label_point = rarr[0].tolist()
                ax.text(region_label_point[0], region_label_point[1], r.tag,
                        bbox=dict(facecolor=r.tag_color, alpha=0.5))

        for l in self.lines:
            min_x = -1
            min_y = -1

            for ls in l.segments:
                larr = np.array(ls.line)
                min_x = min(min_x, float(np.min(larr[:, 0])))
                min_y = min(min_y, float(np.min(larr[:, 1])))

                plt.Line2D(larr[:, 0], larr[:, 1], linewidth=1, linestyle="-", color=ls.color)

            # draw the label
            if l.tag is not None:
                label_point = (min_x, min_y)
                ax.text(label_point, l.tag, bbox=dict(facecolor=r.tag_color, alpha=0.5))
        return ax, fig, plt

    def show(self):
        ax, fig, plt = self._draw_all()
        fig.show()
        plt.show()

    def get_image(self):
        ax, fig, plt = self._draw_all()
        ax.margins(0)
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot



    @staticmethod
    def show_page(img: np.ndarray, pxml: AnalyzedContent, settings = VisualizerSettings()):
        vis = PageVisualizer(img)

        for l in pxml.baselines:
            vis.add_line(VisLine())