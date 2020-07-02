from segmentation_models_pytorch import encoders
import segmentation_models_pytorch as smp
from enum import Enum

ENCODERS = smp.encoders.get_encoder_names()


class Architecture(Enum):
    FPN = 'fpn'
    UNET = 'unet'
    PSPNET = 'pspnet'
    LINKNET = 'linknet'

    def get_architecture(self):
        return {'fpn': smp.FPN,
                'unet': smp.Unet,
                'pspnet': smp.PSPNet,
                'linknet': smp.Linknet}[self.value]

    @staticmethod
    def get_all_architectures():
        return [smp.FPN, smp.Unet, smp.PSPNet, smp.Linknet]

    def get_architecture_params(self):
        import inspect
        t = {'fpn': smp.FPN,
             'unet': smp.Unet,
             'pspnet': smp.PSPNet,
             'linknet': smp.Linknet}[self.value]
        signature = inspect.signature(t.__init__)
        a = dict()
        for name, parameter in signature.parameters.items():
            if name == 'self':
                continue
            a[name] = parameter.default
        return a
