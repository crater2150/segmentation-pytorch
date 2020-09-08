from typing import NamedTuple, List, Tuple
import numpy as np


class BaselineResult(NamedTuple):
    baseline: List
    height: int
    font_width: float
    cluster_type: int
    cluster_location: int

    def scale(self, scale_factor):
        baseline = [(x[0] * scale_factor, x[1] * scale_factor) for x in self.baseline]
        return BaselineResult(baseline=baseline,
                              height=self.height * scale_factor,
                              font_width=self.font_width,
                              cluster_location=self.cluster_location,
                              cluster_type=self.cluster_type
                              )

    def get_max_y(self):
        return max([t[1] for t in self.baseline])

    def get_min_y(self):
        return min([t[1] for t in self.baseline])

    def get_avg_y(self):
        return np.mean([t[1] for t in self.baseline])


class BboxCluster():

    def __init__(self, baselines, bbox):
        self.baselines: List[BaselineResult] = baselines
        # (x_min, y_min), (x_max, y_min ), (x_max, y_max), (x_min, y_max)]
        self.bbox: List[Tuple[any, any]] = bbox

    def scale(self, scale_factor):
        return BboxCluster(baselines=[x.scale(scale_factor) for x in self.baselines],
                           bbox=[(x[0] * scale_factor, x[1] * scale_factor) for x in self.bbox])

    def get_average_height(self):
        return np.mean([x.height for x in self.baselines])

    def get_char_cluster_type(self):
        return self.baselines[0].cluster_type

    def get_location_cluster_type(self):
        return self.baselines[0].cluster_location

    def number_of_baselines_in_cluster(self):
        return len(self.baselines)

    def get_bottom_line_of_bbox(self):
        bbox_sorted = sorted(self.bbox, key=lambda k: (k[1], k[0]))
        return bbox_sorted[-2], bbox_sorted[-1]

    def get_top_line_of_bbox(self):
        bbox_sorted = sorted(self.bbox, key=lambda k: (k[1], k[0]))

        return bbox_sorted[0], bbox_sorted[1]

    def get_left_x(self):
        return self.bbox[0][0]

    def get_right_x(self):
        return self.bbox[1][0]

    def get_top_y(self):
        return self.bbox[0][1]

    def get_bottom_y(self):
        return self.bbox[-1][1]

    def set_baselines(self, bl: List[BaselineResult]):
        self.baselines = bl
