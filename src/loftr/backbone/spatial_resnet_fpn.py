import torch.nn as nn
import torch.nn.functional as F
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x1(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=1, padding=(1, 0))


def conv1x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=1, padding=(0, 1))


class GroupBatchnorm2d(nn.Module):
    """
    Group BatchNorm: 组归一化具体实现(何凯明)
    """

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SpatialBlock(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = True):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigmoid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x  # high informativeness
        x_2 = w2 * x  # low informativeness
        return x_1, x_2


class enBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ver = conv3x1(in_channels, out_channels)
        self.hor = conv1x3(in_channels, out_channels)

    def forward(self, x):
        x1 = self.ver(x)
        x2 = self.hor(x)
        return x1 + x2


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes))

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        self.spatial1 = SpatialBlock(block_dims[0], 16)
        self.spatial2 = SpatialBlock(block_dims[1], 32)
        self.spatial3 = SpatialBlock(block_dims[2], 64)

        self.en1 = enBlock(block_dims[0], block_dims[0])
        self.en2 = enBlock(block_dims[1], block_dims[1])
        self.en3 = enBlock(block_dims[2], block_dims[2])

        self.downsample1 = nn.Sequential(conv1x1(block_dims[0], block_dims[1], stride=2), nn.BatchNorm2d(block_dims[1]))
        self.downsample2 = nn.Sequential(conv1x1(block_dims[1], block_dims[2], stride=2), nn.BatchNorm2d(block_dims[2]))

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=1)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=1)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(dim, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        # self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))

        x11 = self.layer1(x0)  # 1/2
        x11, x12 = self.spatial1(x11)
        x12 = self.en1(x12)
        x1 = x11 + x12
        x1_n, _, x1_h, x1_w = x1.shape

        x1_d = self.downsample1(x1)  # 1/4
        x21, x22 = self.spatial2(x1_d)
        x21 = self.layer2(x21)
        x22 = self.en2(x22)
        x2 = x21 + x22
        x2_n, _, x2_h, x2_w = x2.shape

        x3_d = self.downsample2(x2)  # 1/8
        x31, x32 = self.spatial3(x3_d)
        x31 = self.layer3(x31)
        x32 = self.en3(x32)
        x3 = x31 + x32

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, size=(x2_h, x2_w), mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, size=(x1_h, x1_w), mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)
        # print(x3_out.shape, x1_out.shape)
        return [x3_out, x1_out]


class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        # print(block_dims)

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        return [x4_out, x2_out]


if __name__ == '__main__':
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    _config = lower_config(config)
    loftr_cfg = lower_config(_config['loftr'])

    model = ResNetFPN_8_2(loftr_cfg['resnetfpn'])
    print(model)
    test_x = torch.randn(1, 1, 512, 512)
    ou3, out1 = model(test_x)
    from thop import profile

    flops, params = profile(model, inputs=(test_x,))
    print(f"GFLOPs: {flops / 1e9}, Parameters: {params * 4 / (1024 ** 2)}MB")
