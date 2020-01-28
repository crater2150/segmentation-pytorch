from segmentation_models_pytorch import encoders
import segmentation_models_pytorch as smp
from enum import Enum

ENCODERS = smp.encoders.get_encoder_names()


class Architecture(Enum):
    FPN = 'fpn'
    UNET = 'unet'
    PSPNET = 'pspnet'
    LINKNET = 'linknet'
    PAN = 'pan'

    def get_architecture(self):
        return {'fpn': smp.FPN,
                 'unet': smp.Unet,
                 'pspnet': smp.PSPNet,
                 'linknet': smp.Linknet,
                 'pan': smp.PAN}[self.value]

