from modules.backbones.ConvNext import ConvNeXt
from modules.backbones.wavenet import WaveNet

BACKBONES = {
    'wavenet': WaveNet,
    'ConvNeXt':ConvNeXt
}
