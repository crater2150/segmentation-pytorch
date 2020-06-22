from enum import Enum
from segmentation.modules import Architecture
from segmentation.dataset import MaskDataset
from typing import NamedTuple
from segmentation.optimizer import Optimizers
from segmentation.model import CustomModel


class TrainSettings(NamedTuple):
    TRAIN_DATASET: MaskDataset
    VAL_DATASET: MaskDataset
    CLASSES: int
    OUTPUT_PATH: str

    PSEUDO_DATASET: MaskDataset = None
    EPOCHS: int = 15
    OPTIMIZER: Optimizers = Optimizers.ADAM
    LEARNINGRATE_ENCODER: float = 1.e-5
    LEARNINGRATE_DECODER: float = 1.e-4
    LEARNINGRATE_SEGHEAD: float = 1.e-4
    BATCH_ACCUMULATION: int = 8
    TRAIN_BATCH_SIZE: int = 1
    VAL_BATCH_SIZE: int = 1
    ARCHITECTURE: Architecture = Architecture.UNET
    ENCODER: str = 'efficientnet-b3'
    MODEL_PATH: str = None
    CUSTOM_MODEL: CustomModel = None

    PROCESSES: int = 4


class PredictorSettings(NamedTuple):
    PREDICT_DATASET: MaskDataset = None
    MODEL_PATH: str = None
    PROCESSES: int = 4


class BaseLineDetectionSettings(NamedTuple):
    MAXDISTANCE = 100
    ANGLE = 10
