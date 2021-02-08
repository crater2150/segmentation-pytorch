import argparse
import multiprocessing
import warnings

from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.layout import process_layout, LayoutProcessingSettings
from flask import Flask
from flask import request

app = Flask("segmentation server")

@app.route("/schnipschnip", methods=["POST"])
def schnipschnip():
    data = request.get_json()
    image_path = data["image_path"]
    baselines = data["baselines"]
    baselines = [bl["points"] for bl in baselines]
    baselines = [[(round(p["x"]), round(p["y"])) for p in bl] for bl in baselines]


    img = SourceImage.load(image_path)

    prediction = PredictionResult(baselines=baselines, prediction_resolution=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, lines_only=False,
                                               schnip_schnip=True)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, image_path, simplified_xml=False)
    return xml_gen.baselines_to_xml_string()

warnings.simplefilter(action='ignore', category=FutureWarning)

import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, required=True)
    args = parser.parse_args()
    import json
    baselines = json.load(sys.stdin)

    img = SourceImage.load(args.image_file)

    prediction = PredictionResult(baselines=baselines, prediction_resolution=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, lines_only=False,
                                               schnip_schnip=True)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, args.image_file, simplified_xml=False)
    print(xml_gen.baselines_to_xml_string())


if __name__ == "__main__":
    app.run(port="17654")

