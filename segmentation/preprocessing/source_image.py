from PIL import Image

from segmentation.dataset import get_rescale_factor, rescale_pil
from segmentation.preprocessing.ocrupus import binarize

import numpy as np

class SourceImage:
    fail_on_binarize = False

    @staticmethod
    def load(filename):
        img = Image.open(filename)
        return SourceImage(img)

    @staticmethod
    def from_numpy(arr):
        return SourceImage(Image.fromarray(arr))

    def __init__(self, img: Image, scale_factor=1):
        self.pil_image = img
        self.binarized_cache = None
        self.array_cache = None
        self.scale_factor = scale_factor

    def scaled(self, scale_factor) -> 'SourceImage':
        return SourceImage(rescale_pil(self.pil_image, scale_factor), scale_factor=scale_factor)

    def scale_area(self, max_area, additional_scale_factor=None) -> 'SourceImage':

        rescale_factor = get_rescale_factor(self.pil_image, scale_area=max_area)

        if additional_scale_factor is not None:
            rescale_factor = rescale_factor * additional_scale_factor

        return self.scaled(rescale_factor)

    def binarized(self):
        if self.binarized_cache is None:
            self.binarized_cache = binarize(self.array().astype("float64"),
                                            assert_binarized=SourceImage.fail_on_binarize).astype("uint8")
        return self.binarized_cache

    def array(self):
        if self.array_cache is None:
            self.array_cache = np.array(self.pil_image, dtype=np.uint8)
        return self.array_cache

    def is_rescaled(self):
        return self.scale_factor != 1


