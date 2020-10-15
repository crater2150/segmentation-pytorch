import argparse
import json
from os import path
from typing import List
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from segmentation.settings import PredictorSettings
from segmentation.optimizer import Optimizers
from sklearn.model_selection import KFold


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
    parser.add_argument("--test_mask", type=dir_path, nargs="*", default=[],
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
    parser.add_argument('--folds', default=1, type=int)
    parser.add_argument('--eval', action="store_true", help="Starts evaluation on test set after training")
    parser.add_argument("--scale_area", type=int, default=1000000,
                        help="max pixel amount of an image")
    parser.add_argument("--padding_value", type=int, help="padding size of the image", default=32)
    parser.add_argument('--custom_model', action="store_true",
                        help='Use Custom model for training')
    parser.add_argument("--custom_model_kernel_size", type=int, default=3,
                        help="kernel size of the custom model")
    parser.add_argument("--custom_model_padding_size", type=int, default=1, help="padding of the custom model")
    parser.add_argument("--custom_model_stride_size", type=int, default=1, help="stride of the custom model")
    parser.add_argument("--custom_model_encoder_depth", type=int, default=3, help="encoder depth of the custom model")
    parser.add_argument("--custom_model_attention_encoder_depth", type=int, default=3,
                        help="attention_encoder depth of the custom model")
    parser.add_argument("--use_attention", action="store_true", help="use attention for the custom model")
    parser.add_argument("--attention_depth", type=int, default=3, help="attention depth of the custom model")
    parser.add_argument('--encoder_filter', nargs='+', type=int, help="filter of the encoder of the custom model. Number of filters should be equal to enocder depth + 1")
    parser.add_argument('--decoder_filter', nargs='+', type=int, help="filter of the decoder of the custom model. Number of filters should be equal to encoder depth + 1")
    parser.add_argument('--encoder_attention_filter', nargs='+', type=int, help="filter of the attention encoder of the custom model. Number of filters should be equal to attention depth + 1")
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    train = dirs_to_pandaframe(args.train_input, args.train_mask)
    test = dirs_to_pandaframe(args.test_input, args.test_mask)
    test = test if len(test) > 0 else train
    map = load_image_map_from_file(args.map)
    from segmentation.dataset import base_line_transform
    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    model_paths = []
    test_sets = []
    if args.test_input == [] and args.folds > 1:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for ind, x in enumerate(kf.split(train, None)):
            train_fold = train.iloc[x[0]].reset_index(drop=True)
            test_fold = train.iloc[x[1]].reset_index(drop=True)
            train_dataset = XMLDataset(train_fold, map, transform=compose([base_line_transform()]),
                                       mask_generator=MaskGenerator(settings=settings), scale_area=args.scale_area)
            test_dataset = XMLDataset(test_fold, map, transform=compose([base_line_transform()]),
                                      mask_generator=MaskGenerator(settings=settings), scale_area=args.scale_area)
            model_path = args.output + "_fold{}".format(ind)
            custom_model = None
            if args.custom_model:
                from segmentation.settings import CustomModelSettings
                custom_model = CustomModelSettings(
                    ENCODER_FILTER=args.encoder_filter,
                    DECODER_FILTER=args.decoder_filter,
                    ATTENTION_ENCODER_FILTER=args.encoder_attention_filter,
                    ATTENTION=args.use_attention,
                    CLASSES=len(map),
                    ATTENTION_DEPTH=args.attention_depth,
                    ENCODER_DEPTH=args.custom_model_encoder_depth,
                    ATTENTION_ENCODER_DEPTH=args.custom_model_attention_encoder_depth,
                    STRIDE=args.custom_model_stride_size,
                    PADDING=args.custom_model_padding_size,
                    KERNEL_SIZE=args.custom_model_kernel_size,
                )
            setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=train_dataset, VAL_DATASET=test_dataset,
                                    OUTPUT_PATH=model_path,
                                    MODEL_PATH=args.load, EPOCHS=args.n_epoch,
                                    OPTIMIZER=Optimizers(args.optimizer), BATCH_ACCUMULATION=args.batch_accumulation,
                                    ENCODER=args.encoder,
                                    ARCHITECTURE=Architecture(args.architecture), PROCESSES=args.processes,
                                    PADDING_VALUE=args.padding_value,
                                    CUSTOM_MODEL=custom_model)
            trainer = Network(setting, color_map=map)
            trainer.train()
            model_paths.append(model_path)
            test_sets.append(test_dataset)

    else:
        train_dataset = XMLDataset(train, map, transform=compose([base_line_transform()]),
                                   mask_generator=MaskGenerator(settings=settings))
        test_dataset = XMLDataset(test, map, transform=compose([base_line_transform()]),
                                  mask_generator=MaskGenerator(settings=settings))
        model_path = args.output

        custom_model = None
        if args.custom_model:
            from segmentation.settings import CustomModelSettings
            custom_model = CustomModelSettings(
                ENCODER_FILTER=args.encoder_filter,
                DECODER_FILTER=args.decoder_filter,
                ATTENTION_ENCODER_FILTER=args.encoder_attention_filter,
                ATTENTION=args.use_attention,
                CLASSES=len(map),
                ATTENTION_DEPTH=args.attention_depth,
                ENCODER_DEPTH=args.custom_model_encoder_depth,
                ATTENTION_ENCODER_DEPTH=args.custom_model_attention_encoder_depth,
                STRIDE=args.custom_model_stride_size,
                PADDING=args.custom_model_padding_size,
                KERNEL_SIZE=args.custom_model_kernel_size,
            )

        setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=train_dataset, VAL_DATASET=test_dataset,
                                OUTPUT_PATH=args.output,
                                MODEL_PATH=args.load, EPOCHS=args.n_epoch,
                                OPTIMIZER=Optimizers(args.optimizer), BATCH_ACCUMULATION=args.batch_accumulation,
                                ENCODER=args.encoder,
                                ARCHITECTURE=Architecture(args.architecture), PROCESSES=args.processes,
                                CUSTOM_MODEL=custom_model, PADDING_VALUE=args.padding_value)

        trainer = Network(setting, color_map=map)
        trainer.train()
        model_paths.append(model_path)
        test_sets.append(test_dataset)

    if args.eval:
        total_accuracy = 0
        total_loss = 0
        for ind, x in enumerate(model_paths):
            setting = PredictorSettings(PREDICT_DATASET=test_sets[ind], MODEL_PATH=x + ".torch")
            predictor = Network(setting, color_map=map)
            accuracy, loss = predictor.eval()
            total_accuracy += accuracy
            total_loss += loss
        print("EXPERIMENT_OUT=" + str(total_accuracy / len(model_paths)) + "," + str(total_loss / len(model_paths)))


if __name__ == "__main__":
    main()
