from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
# from .spatial_resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .cspdensenet import CSPDenseNet
from .spatial_cspdensenet import SCDenseNet
# from .spatial_cspdensenet_2 import SCDenseNet


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'DenseNet':
        # return CSPDenseNet(config['densenet'])
        return SCDenseNet(config['densenet'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
