from enum import Enum
from segmentation.modules import Architecture
from segmentation.dataset import MaskDataset
from typing import NamedTuple

class TrainSettings(NamedTuple):
    TRAIN_DATASET: MaskDataset
    VAL_DATASET: MaskDataset
    CLASSES: int

    LEARNINGRATE_ENCODER: float = 1.e-5
    LEARNINGRATE_DECODER: float = 1.e-4
    LEARNINGRATE_SEGHEAD: float = 1.e-4
    BATCH_ACCUMULATION: int = 8
    TRAIN_BATCH_SIZE: int = 1
    VAL_BATCH_SIZE: int = 1
    ARCHITECTURE: Architecture = Architecture.UNET
    ENCODER: str = 'resnet34'

    PROCESSES: int = 4
