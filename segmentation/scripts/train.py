import argparse
import json
from os import path
from typing import List
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from segmentation.settings import PredictorSettings
from segmentation.optimizer import Optimizers


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
    from segmentation.network import TrainSettings, dirs_to_pandaframe, load_image_map_from_file, MaskSetting, MaskType, \
        PCGTSVersion, XMLDataset, Network, compose, MaskGenerator, MaskDataset
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--l-rate", type=float, default=1e-4,
                        help="set learning rate")
    parser.add_argument("-O", "--output", type=str, default="./",
                        help="target directory for model and logs")
    parser.add_argument("--load", type=str, default=None,
                        help="load an existing model and continue training")
    parser.add_argument("-E", "--n_epoch", type=int, default=15,
                        help="number of epochs")
    parser.add_argument("--data-augmentation", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--train_input", type=dir_path, nargs="+", default=[],
                        help="Path to folder(s) containing train images")
    parser.add_argument("--train_mask", type=dir_path, nargs="+", default=[],
                        help="Path to folder(s) containing train xmls")

    parser.add_argument("--test_input", type=dir_path, nargs="*", default=[],
                        help="Path to folder(s) containing test images")
    parser.add_argument("--test_mask", type=dir_path, nargs="+", default=[],
                        help="Path to folder(s) containing test xmls")

    parser.add_argument("--color_map", dest="map", type=str, required=True,
                        help="path to color map to load")
    parser.add_argument('--architecture',
                        default=Architecture.UNET,
                        const=Architecture.UNET,
                        nargs='?',
                        choices=[x.value for x in list(Architecture)],
                        help='Network architecture to use for training')
    parser.add_argument('--encoder',
                        default="efficientnet-b3",
                        const="efficientnet-b3",
                        choices=ENCODERS,
                        nargs='?',
                        help='Network architecture to use for training')
    parser.add_argument('--optimizer', default="adam", const="adam", nargs='?',
                        choices=[x.value for x in list(Optimizers)])
    parser.add_argument('--batch_accumulation', default=1, type=int)
    parser.add_argument('--processes', default=1, type=int)

    parser.add_argument('--eval', action="store_true", help="Starts evaluation on test set after training")

    args = parser.parse_args()
    train = dirs_to_pandaframe(args.train_input, args.train_mask)
    test = dirs_to_pandaframe(args.test_input, args.test_mask)
    test = test if len(test) > 0 else train

    map = load_image_map_from_file(args.map)
    from segmentation.dataset import base_line_transform

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    train_dataset = XMLDataset(train, map, transform=compose([base_line_transform()]),
                               mask_generator=MaskGenerator(settings=settings))
    test_dataset = XMLDataset(test, map, transform=compose([base_line_transform()]),
                              mask_generator=MaskGenerator(settings=settings))

    setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=train_dataset, VAL_DATASET=test_dataset,
                            OUTPUT_PATH=args.output,
                            MODEL_PATH=args.load, EPOCHS=args.n_epoch,
                            OPTIMIZER=Optimizers(args.optimizer), BATCH_ACCUMULATION=args.batch_accumulation,
                            ENCODER=args.encoder,
                            ARCHITECTURE=Architecture(args.architecture), PROCESSES=args.processes)

    trainer = Network(setting, color_map=map)
    trainer.train()
    if args.eval:
        setting = PredictorSettings(PREDICT_DATASET=test_dataset, MODEL_PATH=args.output + ".torch")
        predictor = Network(setting, color_map=map)
        accuracy, loss = predictor.eval()
        print("EXPERIMENT_OUT=" + str(accuracy) + "," + str(loss))


if __name__ == "__main__":
    main()
