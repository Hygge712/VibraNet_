import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out


class CSPDenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(CSPDenseBlock, self).__init__()
        self.split_channels = in_channels // 2

        self.dense_layers = nn.ModuleList([
            DenseLayer(self.split_channels + i * growth_rate, growth_rate) for i in range(num_layers)
        ])
        self.transition_layer = nn.Conv2d(self.split_channels + num_layers * growth_rate, self.split_channels,
                                          kernel_size=1, bias=False)

    def forward(self, x):
        split_out = x[:, :self.split_channels, :, :]
        out = x[:, self.split_channels:, :, :]
        for layer in self.dense_layers:
            out = layer(out)
        out = self.transition_layer(out)
        out = torch.cat([split_out, out], 1)
        return out


# 定义 Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


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


class SCDenseNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        block_dims = config['block_dims']
        self.growth_rate = config['growth_rate']
        self.num_blocks = config['num_blocks']
        self.num_layers_per_block = config['num_layers_per_block']
        self.num_channels = config['initial_dim']

        self.spatial1 = SpatialBlock(block_dims[0], 16)
        self.spatial2 = SpatialBlock(block_dims[1], 32)
        self.spatial3 = SpatialBlock(block_dims[2], 64)

        self.en1 = enBlock(block_dims[0], block_dims[0])
        self.en2 = enBlock(block_dims[1], block_dims[1])
        self.en3 = enBlock(block_dims[2], block_dims[2])

        self.conv1 = nn.Conv2d(1, self.num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = CSPDenseBlock(self.num_layers_per_block, block_dims[0], self.growth_rate)
        self.mid_layer1 = TransitionLayer(block_dims[0], block_dims[1])

        self.layer2 = CSPDenseBlock(self.num_layers_per_block, block_dims[1], self.growth_rate * 2)
        self.mid_layer2 = TransitionLayer(block_dims[1], block_dims[2])

        self.layer3 = CSPDenseBlock(self.num_layers_per_block, block_dims[2], self.growth_rate * 2)

        # FPN Upsample
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

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        x11, x12 = self.spatial1(x0)
        x11 = self.layer1(x11)
        x12 = self.en1(x12)
        x1 = x11 + x12
        x1_n, _, x1_h, x1_w = x1.shape
        x1_1 = self.mid_layer1(x1)

        x21, x22 = self.spatial2(x1_1)
        x21 = self.layer2(x21)
        x22 = self.en2(x22)
        x2 = x21 + x22
        x2_n, _, x2_h, x2_w = x2.shape
        x2_1 = self.mid_layer2(x2)

        x31, x32 = self.spatial3(x2_1)
        x31 = self.layer3(x31)
        x32 = self.en3(x32)
        x3 = x31 + x32
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, size=(x2_h, x2_w), mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, size=(x1_h, x1_w), mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        # print(x3.shape, x1.shape)

        return [x3_out, x1_out]


if __name__ == '__main__':
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    _config = lower_config(config)
    loftr_cfg = lower_config(_config['loftr'])
    print(type(loftr_cfg))
    net = SCDenseNet(loftr_cfg['densenet'])
    # print(net)
    test_x = torch.randn(1, 1, 512, 512)
    out3, out1 = net(test_x)
    print(out3.shape, out1.shape)

    from thop import profile
    flops, params = profile(net, inputs=(test_x,))
    print(f"GFLOPs: {flops / 1e9}, Parameters: {params * 4 / (1024 ** 2)}MB")
