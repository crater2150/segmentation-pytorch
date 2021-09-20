from PIL import Image

from segmentation.dataset import get_rescale_factor, rescale_pil

import numpy as np
from doxapy.binarization import BinarizationAlgorithm, binarize, _needs_binarization


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
        return SourceImage(rescale_pil(self.pil_image.copy(), scale_factor), scale_factor=scale_factor)

    def scale_area(self, max_area, additional_scale_factor=None) -> 'SourceImage':

        rescale_factor = get_rescale_factor(self.pil_image, scale_area=max_area)

        if additional_scale_factor is not None:
            rescale_factor = rescale_factor * additional_scale_factor

        return self.scaled(rescale_factor)

    def binarized(self):
        if self.binarized_cache is None:
            if SourceImage.fail_on_binarize:
                if len(self.array().shape) == 3 or _needs_binarization(self.array()):
                    raise AssertionError("Image should already be binarized")
                return self.array()
            else:
                self.binarized_cache = binarize(self.array(), BinarizationAlgorithm.ISauvola)*np.float32(1)
                assert self.binarized_cache.shape[:2] == self.array().shape[:2]
                # Deprecated: Old Ocropus binarization can also be handled by the new binarizer library
                # self.binarized_cache = binarize(self.array().astype("float64"),
                #                            assert_binarized=SourceImage.fail_on_binarize).astype("uint8")

        return self.binarized_cache

    def array(self):
        if self.array_cache is None:
            self.array_cache = np.array(self.pil_image).astype(np.uint8)
        return self.array_cache

    def is_rescaled(self):
        return self.scale_factor != 1

    def get_width(self):
        return int(self.array().shape[1])

    def get_height(self):
        return int(self.array().shape[0])

    # todo: padding_factor will not be stored in the source_image
    def pad(self, factor=0.5) -> 'SourceImage':
        ow, oh = self.get_width(), self.get_height()
        scaled = self.scaled(factor)
        new_im = Image.new("RGB", (ow,oh))  # todo: what if the source image is greyscale
        new_im.paste(scaled.pil_image, (((ow - scaled.get_width()) // 2),
                                       ((oh - scaled.get_height()) // 2)))
        return SourceImage(new_im)



