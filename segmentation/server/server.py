import argparse
import json
import math
import multiprocessing
import os.path
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
from flask import Flask, request

from segmentation.gui.xml_util import XMLGenerator
from segmentation.postprocessing.baselines_util import make_baseline_continous
from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.postprocessing.layout_settings import LayoutProcessingMethod
from segmentation.predictors import PredictionSettings, Predictor
from segmentation.preprocessing.source_image import SourceImage
from segmentation.server.calamari_config import get_config
from segmentation.scripts.layout import process_layout, LayoutProcessingSettings
from segmentation.util import PerformanceCounter, logger
from PIL import Image
import numpy as np

app = Flask("segmentation server")
try:
    #settings = PredictionSettings(["/opt/segmentation-models/model136.torch"], 1000000, None, None)
    settings = PredictionSettings(["/home/norbert/share/BaselineModEval/mod/model_136.torch"], 1000000, None, None)

    if not torch.cuda.is_available():
        torch.set_num_threads(multiprocessing.cpu_count())
        # TODO: On Ryzen 5600X this does not improve performance
        # To improve CPU prediction performance, maybe prefetch img load and run distance_matrix on multiple cores

    nn_predictor = Predictor(settings)
except:
    print("Cannot load model.")
    nn_predictor = None

SERVER_LAYOUT_METHOD = LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP

@app.route("/schnipschnip", methods=["POST"])
def schnipschnip():
    data = request.get_json()
    image_path = data["image_path"]
    baselines = data["baselines"]
    baselines = [bl["points"] for bl in baselines]
    baselines = [[(round(p["x"]), round(p["y"])) for p in bl] for bl in baselines]
    logger.info(f"{data}\n")

    img = SourceImage.load(image_path)

    prediction = PredictionResult(baselines=baselines, prediction_shape=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, layout_method=SERVER_LAYOUT_METHOD)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, image_path, simplified_xml=False)
    return run_ocr(image_path,xml_gen)
    #return xml_gen.baselines_to_xml_string()

class LineSegment:

    def __init__(self, p1x, p1y, p2x, p2y):
        self.p1x, self.p1y, self.p2x, self.p2y = p1x, p1y, p2x, p2y
        self.len = math.sqrt((self.p2x - self.p1x)**2 + (self.p2y - self.p1y)**2)

    def point_distance(self, px, py):
        return abs((self.p2x - self.p1x) * (self.p1y - py) - (self.p1x - px) * (self.p2y - self.p1y)) / self.len


def marginalia_cut_baseline_by_line(cont_baselines, cut_line_seg):
    cut_baselines = []
    CUT_DIST = 1
    MIN_LENGTH = 5
    for bl_i, bl in enumerate(cont_baselines):
        cut_index = None
        for pi, (px, py) in enumerate(bl[MIN_LENGTH:-MIN_LENGTH], start=MIN_LENGTH):
            if cut_line_seg.point_distance(px, py) <= CUT_DIST:
                cut_index = pi
                break
        else:
            # we didn't cut this baseline
            cut_baselines.append(bl)
            logger.debug(f"Leaving baseline {bl_i}\n")
            continue

        # we have a cut index
        if cut_index >= MIN_LENGTH:
            cut_baselines.append(bl[:cut_index])
            logger.debug(f"Appending first part of {bl_i}\n")
        # else: discard the leading part


        # find where the cut stops
        cut_end = None

        for pi, (px,py) in enumerate(bl[cut_index:], start=cut_index):
            if cut_line_seg.point_distance(px,py) > CUT_DIST:
                cut_end = pi
                break
        if cut_end is not None:
            # we have cut away something
            if len(bl[cut_end:]) > MIN_LENGTH:
                cut_baselines.append(bl[cut_end:])
                logger.debug(f"Appending last part of {bl_i}\n")

    return cut_baselines


@app.route("/marginaliaCut", methods=["POST"])
def marginalia_cut():
    data = request.get_json()
    print(data)
    image_path = data["image_path"]
    baselines = data["baselines"]
    if len(data["cutline"]) != 2:
        return "error", 500

    cut_line_p1x = float(data["cutline"][0]["x"])
    cut_line_p1y = float(data["cutline"][0]["y"])
    cut_line_p2x = float(data["cutline"][1]["x"])
    cut_line_p2y = float(data["cutline"][1]["y"])

    cut_line_seg = LineSegment(cut_line_p1x, cut_line_p1y, cut_line_p2x, cut_line_p2y)
    baselines = [bl["points"] for bl in baselines]
    baselines = [[(round(p["x"]), round(p["y"])) for p in bl] for bl in baselines]
    cont_baselines = [make_baseline_continous(bl) for bl in baselines]
    cut_baselines = marginalia_cut_baseline_by_line(cont_baselines, cut_line_seg)

    img = SourceImage.load(image_path)

    prediction = PredictionResult(baselines=cut_baselines, prediction_shape=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True,
                                               layout_method=SERVER_LAYOUT_METHOD)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, image_path, simplified_xml=False)
    return run_ocr(image_path,xml_gen)
    #return xml_gen.baselines_to_xml_string()


@app.route("/oneclick", methods=["POST"])
def oneclick():
    if not nn_predictor:
        return json.dumps({"error": "Model could not be loaded or pyTorch is not available."}), 500

    data = request.get_json()
    image_path = data["image_path"]
    with PerformanceCounter(f"Oneclick for {image_path}"):
        img = SourceImage.load(image_path)

        prediction, scaled_image = nn_predictor.predict_image(img)
        layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                                   source_scale=True, layout_method=LayoutProcessingMethod.FULL)

        analyzed_content = process_layout(prediction, scaled_image, multiprocessing.Pool(2), layout_settings)
        analyzed_content = analyzed_content.to_pagexml_space(prediction.prediction_scale_factor)
        xml_gen = analyzed_content.export(scaled_image, image_path, simplified_xml=False)
    return run_ocr(image_path, xml_gen.baselines_to_xml_string())
    #return xml_gen.baselines_to_xml_string()

@app.route("/justocr", methods=["POST"])
def justocr():
    data = request.get_json()
    logger.info(f"JustOCR for {data}")
    return run_ocr(data["image_path"], data["pagexml"])


warnings.simplefilter(action='ignore', category=FutureWarning)

import sys

def run_ocr_old(orig_img_path, page_xml_string: str):
    logger.info(page_xml_string)
    config = get_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        if config["ocr_binarize"]:
            logger.info("Binarizing Image for OCR")
            img_file = Path(tmpdir) / Path(orig_img_path).with_suffix(".png").name
            img = SourceImage.load(orig_img_path)
            binarized_arr = (img.binarized() + 0.5).astype(np.uint8) * np.uint8(255)
            Image.fromarray(binarized_arr).save(str(img_file))
        else:
            img_file = (Path(tmpdir) / Path(orig_img_path).name)
            shutil.copy(orig_img_path, img_file)

        xml_file = img_file.with_suffix(".xml")
        with open(xml_file,"w") as f:
            f.write(page_xml_string)

        try:
            config = get_config()
            # run calamari
            ret = os.system(f'''
            . {config['venv']} && export PYTHONPATH="{config["pythonpath"]}" \
            && export CUDA_VISIBLE_DEVICES="" \
            && python -m calamari_ocr.scripts.predict --checkpoint {config["model"]} --dataset PAGEXML --files "{img_file}"
            ''')  # TODO: this is a security hazard
            logger.info(ret)

            with open(xml_file.with_suffix(".pred.xml")) as f:
                ocr_pxml_string = f.read()
                # TODO: replace the image filename with

                return ocr_pxml_string
        except Exception as e:
            logger.error("Error running OCR")
            logger.error(e)
            return page_xml_string

def run_ocr_new(image_filename, xml_gen: XMLGenerator):
    pass

def run_ocr(image_filename, page_xml_string: str):
    config = get_config()
    if config["calamari_version"] == "old":
        return run_ocr_old(image_filename, page_xml_string)
    elif config["calamari_version"] == "new":
        return run_ocr_new(image_filename, page_xml_string)
    else:
        raise KeyError("Invalid Calamari version specified")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, required=True)
    args = parser.parse_args()
    import json
    baselines = json.load(sys.stdin)

    img = SourceImage.load(args.image_file)

    prediction = PredictionResult(baselines=baselines, prediction_shape=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, layout_method=LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, args.image_file, simplified_xml=False)
    print(xml_gen.baselines_to_xml_string())


if __name__ == "__main__":
    app.run(port="17654",threaded=False)  # you cannot use threaded flask or multiple processes, because the model needs to be loaded


