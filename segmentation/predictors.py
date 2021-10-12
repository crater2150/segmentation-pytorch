import multiprocessing
from typing import List, Tuple, NamedTuple

import numpy as np
import shapely
from shapely.geometry import LineString

from segmentation.network import Network
from segmentation.postprocessing.baseline_extraction import extract_baselines_from_probability_map
from segmentation.postprocessing.baselines_util import simplify_baseline
from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.postprocessing.layout_analysis import get_top_of_baselines
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import PredictorSettings
from segmentation.util import PerformanceCounter, logger


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x, scale_area, additional_scale_factor=None):
        raise DeprecationWarning()
        res = []
        scale_factor = None
        for m in self.models:
            p_map, s_factor = m.predict_single_image_by_path(x, rgb=True, preprocessing=True, scale_area=scale_area,
                                                             additional_scale_factor=additional_scale_factor)
            scale_factor = s_factor
            res.append(p_map)
        if len(res) == 1:
            return res[0], scale_factor
        res = np.stack(res, axis=0)
        return np.mean(res, axis=0), scale_factor

    def predict_image(self, source_image: SourceImage):
        def predict(m):
            # tta_aug=None means default augmentation
            return m.predict_single_image(source_image.array(), rgb=True, preprocessing=True, tta_aug=None)

        if len(self.models) == 1:
            return predict(self.models[0])
        else:
            res = np.zeros(shape=source_image.array().shape, dtype=np.float32)
            for m in self.models:
                res += predict(m)
            return res / len(self.models)  # TODO: check if this is equivalent (it probably is)


class PredictionSettings(NamedTuple):
    model_paths: List[str]
    scale_area: int = None
    min_line_height: int = None
    max_line_height: int = None


class Predictor:
    def __init__(self, settings: PredictionSettings):
        self.settings = settings
        self.networks = [Network(PredictorSettings(MODEL_PATH=x)) for x in self.settings.model_paths]
        self.ensemble = Ensemble(self.networks)

    def predict_image(self, source_image: SourceImage,
                      process_pool: multiprocessing.Pool = None) -> (PredictionResult, SourceImage):
        if process_pool is None:
            process_pool = multiprocessing.pool.ThreadPool(1)
        scale_factor_multiplier = 1
        while True:
            scaled_image = source_image.scale_area(self.settings.scale_area, scale_factor_multiplier)
            with PerformanceCounter("Prediction"):
                p_map = self.ensemble.predict_image(scaled_image)
                baselines = extract_baselines_from_probability_map(p_map, process_pool=process_pool)

            if baselines is not None:
                binary = scaled_image.binarized()
                with PerformanceCounter(function_name="Baseline Height Calculation mp"):
                    out = get_top_of_baselines(baselines, image=1 - binary,
                                               process_pool=None)  # No MP is faster here (avoid image copy)
                heights = [x[2] for x in out]
                med_height = np.median(heights)

                if (self.settings.max_line_height is not None and med_height > self.settings.max_line_height) \
                        or (self.settings.min_line_height is not None and med_height < self.settings.min_line_height) \
                        and scale_factor_multiplier == 1:
                    scale_factor_multiplier = (self.settings.max_line_height - 7) / med_height
                    logger.info("Resizing image Avg:{}, Med:{} \n".format(np.mean(heights), med_height))
                    continue
            break

        # simplify the baselines
        baselines = [simplify_baseline(bl) for bl in baselines or []]
        # now we have the baselines extracted
        return PredictionResult(baselines=baselines,
                                prediction_scale_factor=scaled_image.scale_factor,
                                prediction_shape=list(scaled_image.array().shape)), scaled_image




class BigTextDetector:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    @staticmethod
    def baseline_to_poly_candidates(baselines, slack_radius: float) -> List[shapely.geometry.Polygon]:
        line_strings = [LineString(bl if len(bl) > 1 else bl * 2) for bl in baselines]
        return [ls.buffer(slack_radius) for ls in line_strings]

    def predict_padded(self, img: SourceImage, factor: float = 0.5, process_pool: multiprocessing.Pool = None) -> Tuple[PredictionResult, SourceImage]:
        if process_pool is None:
            process_pool = multiprocessing.pool.ThreadPool(1)
        # pad the image by 50%
        pad_img = img.pad(factor)
        pred_pad, scaled_padded_img = self.predictor.predict_image(pad_img, process_pool)
        # convert the padded_prediction result to original image space
        converted_baselines = []
        offs_x = int(scaled_padded_img.get_width() * (factor / 2))
        offs_y = int(scaled_padded_img.get_height() * (factor / 2))

        for bl in pred_pad.baselines:
            new_bl = [((p[0] - offs_x)*2, (p[1] - offs_y) * 2) for p in bl]
            converted_baselines.append(new_bl)

        pred_pad.baselines = converted_baselines
        return pred_pad, scaled_padded_img


    def predict(self, img: SourceImage, process_pool: multiprocessing.Pool) -> Tuple[PredictionResult, SourceImage]:
        pred_orig, scaled_image = self.predictor.predict_image(img, process_pool)
        pred_pad, scaled_padded_img = self.predict_padded(img, 0.5, process_pool)

        orig_bl_polys = self.__class__.baseline_to_poly_candidates(pred_orig.baselines, 5.0)
        pad_bl_polys = self.__class__.baseline_to_poly_candidates(pred_pad.baselines, 5.0)


        for i, pad_poly in enumerate(pad_bl_polys):
            # see if we intersect with an original poly
            replaced = False
            for orig_i, op in enumerate(orig_bl_polys):
                if pad_poly.intersection(op).area / pad_poly.union(op).area > 0:
                    if len(pred_orig.baselines[orig_i]) < 0.8 * len(pred_pad.baselines[i]):
                        pred_orig.baselines[orig_i] = pred_pad.baselines[i]
                        replaced = True
                    else:
                        break

                    # maybe replace
                    # now we have to decide which baseline we choose
            else:
                # we didn't find this baseline, append it
                if not replaced:
                    pred_orig.baselines.append(pred_pad.baselines[i])
        seen = set()
        new_baselines = [] # deduplicate
        for bl in pred_orig.baselines:
            if id(bl) in seen:
                continue
            else:
                new_baselines.append(bl)
                seen.add(id(bl))
        pred_orig.baselines = new_baselines
        return pred_orig, scaled_image
