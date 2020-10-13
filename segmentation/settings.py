from segmentation.modules import Architecture
from segmentation.dataset import MaskDataset
from typing import NamedTuple, Tuple, List
from segmentation.optimizer import Optimizers
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CustomModelSettings:
    CLASSES: int
    ENCODER_FILTER: list
    DECODER_FILTER: list
    ATTENTION_ENCODER_FILTER: list
    TYPE: str = "attentionunet"
    KERNEL_SIZE: int = 3
    PADDING: int = 1
    STRIDE: int = 1
    ENCODER_DEPTH: int = 3
    ATTENTION_DEPTH: int = 3
    ATTENTION_ENCODER_DEPTH: int = 3
    ACTIVATION: bool = False
    CHANNELS_IN: int = 3
    CHANNELS_OUT: int = 16
    ATTENTION: bool = True

    def get_kwargs(self):
        return {
            "in_channels": self.CHANNELS_IN,
            "out_channels": self.CHANNELS_OUT,
            "n_class": self.CLASSES,
            "kernel_size": self.KERNEL_SIZE,
            "padding": self.PADDING,
            "stride": self.STRIDE,
            "attention": self.ATTENTION,
            "encoder_depth": self.ENCODER_DEPTH,
            "attention_depth": self.ATTENTION_DEPTH,
            "encoder_filter": self.ENCODER_FILTER,
            "decoder_filter": self.DECODER_FILTER,
            "attention_encoder_filter": self.ATTENTION_ENCODER_FILTER,
            "attention_encoder_depth": self.ATTENTION_ENCODER_DEPTH,
        }


@dataclass
class TrainSettings:
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
    PADDING_VALUE: int = 32

    CUSTOM_MODEL: CustomModelSettings = None
    DECODER_CHANNELS: Tuple[int, ...] = field(default_factory=tuple)
    ENCODER_DEPTH: int = 5
    ENCODER: str = 'efficientnet-b3'
    BATCH_ACCUMULATION: int = 8
    TRAIN_BATCH_SIZE: int = 1
    VAL_BATCH_SIZE: int = 1
    ARCHITECTURE: Architecture = Architecture.UNET
    MODEL_PATH: str = None
    IMAGEMAX_AREA: int = 1000000

    PROCESSES: int = 0

    def __post_init__(self):
        if len(self.DECODER_CHANNELS) == 0:
            self.DECODER_CHANNELS = (256, 128, 64, 32, 16)

    def to_json(self):
        json_dict = {}
        for x in list(self.__dict__.keys()):

            if x in ['PSEUDO_DATASET', 'TRAIN_DATASET', 'VAL_DATASET']:
                continue
            else:
                if isinstance(self.__dict__[x], Enum):
                    json_dict[x] = self.__dict__[x].value
                    continue
                if isinstance(self.__dict__[x], CustomModelSettings):
                    json_dict[x] = asdict(self.__dict__[x])
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
