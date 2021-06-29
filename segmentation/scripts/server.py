import argparse
import json
import multiprocessing
import warnings

import torch

from segmentation.postprocessing.data_classes import PredictionResult
from segmentation.postprocessing.layout_settings import LayoutProcessingMethod
from segmentation.predictors import PredictionSettings, Predictor
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.layout import process_layout, LayoutProcessingSettings
from flask import Flask, request

from segmentation.util import PerformanceCounter

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

@app.route("/schnipschnip", methods=["POST"])
def schnipschnip():
    data = request.get_json()
    image_path = data["image_path"]
    baselines = data["baselines"]
    baselines = [bl["points"] for bl in baselines]
    baselines = [[(round(p["x"]), round(p["y"])) for p in bl] for bl in baselines]


    img = SourceImage.load(image_path)

    prediction = PredictionResult(baselines=baselines, prediction_shape=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, layout_method=LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, image_path, simplified_xml=False)
    return xml_gen.baselines_to_xml_string()

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

    prediction = PredictionResult(baselines=baselines, prediction_shape=list(img.array().shape))
    layout_settings = LayoutProcessingSettings(marginalia_postprocessing=False,
                                               source_scale=True, layout_method=LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP)

    analyzed_content = process_layout(prediction, img, multiprocessing.Pool(2), layout_settings)
    xml_gen = analyzed_content.export(img, args.image_file, simplified_xml=False)
    print(xml_gen.baselines_to_xml_string())


if __name__ == "__main__":
    app.run(port="17654",threaded=False) # you cannot use threaded flask or multiple processes, because the model needs to be loaded


