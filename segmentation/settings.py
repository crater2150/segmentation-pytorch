from enum import Enum
from segmentation.modules import Architecture
from segmentation.dataset import MaskDataset
from typing import NamedTuple, List, Tuple
from segmentation.optimizer import Optimizers
from segmentation.model import CustomModel
from dataclasses import dataclass, field
from enum import Enum
import json


@dataclass
class TrainSettings:
    TRAIN_DATASET: MaskDataset
    VAL_DATASET: MaskDataset

    CLASSES: int
    OUTPUT_PATH: str

    PSEUDO_DATASET: MaskDataset = None
    CUSTOM_MODEL: CustomModel = None
    EPOCHS: int = 15
    OPTIMIZER: Optimizers = Optimizers.ADAM
    LEARNINGRATE_ENCODER: float = 1.e-5
    LEARNINGRATE_DECODER: float = 1.e-4
    LEARNINGRATE_SEGHEAD: float = 1.e-4
    ENCODER_DEPTH: int = 5
    BATCH_ACCUMULATION: int = 8
    DECODER_CHANNELS: Tuple[int, ...] = field(default_factory=tuple)
    TRAIN_BATCH_SIZE: int = 1
    VAL_BATCH_SIZE: int = 1
    ARCHITECTURE: Architecture = Architecture.UNET
    ENCODER: str = 'efficientnet-b3'
    MODEL_PATH: str = None
    IMAGEMAX_AREA: int = 1000000

    PROCESSES: int = 0

    def __post_init__(self):
        if len(self.DECODER_CHANNELS) == 0:
            self.DECODER_CHANNELS = (256, 128, 64, 32, 16)

    def to_json(self):
        json_dict = {}
        for x in list(self.__dict__.keys()):
            if x == "DECODER_CHANNELS":
                print(x)
            if x in ['PSEUDO_DATASET', 'CUSTOM_MODEL', 'TRAIN_DATASET', 'VAL_DATASET']:
                print(x)
                continue
            else:
                if isinstance(self.__dict__[x], Enum):
                    json_dict[x] = self.__dict__[x].value
                    continue
                json_dict[x] = self.__dict__[x]
        t = json.dumps(json_dict, indent=4)
        return t


    @staticmethod
    def load_from_json(self, json):
        pass


class PredictorSettings(NamedTuple):
    PREDICT_DATASET: MaskDataset = None
    MODEL_PATH: str = None
    PROCESSES: int = 4


class BaseLineDetectionSettings(NamedTuple):
    MAXDISTANCE = 100
    ANGLE = 10
