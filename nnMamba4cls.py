import torch
import torch.nn as nn
from mamba_ssm import Mamba


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.nin2 = conv1x1(dim, dim)
        self.norm2 = nn.BatchNorm3d(dim)  # LayerNorm
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm3d(dim)  # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # act_x = self.relu3(x)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class nnMambaEncoder(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=1):
        super(nnMambaEncoder, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.mamba_layer_stem = MambaLayer(
            dim=channels,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2  # Block expansion factor
        )

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp = nn.Sequential(nn.Linear(channels * 14, channels), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(channels, number_classes))

    def forward(self, x):
        c1 = self.in_conv(x)
        c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1_s)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        pooled_c2_s = self.pooling(c2)
        pooled_c3_s = self.pooling(c3)
        pooled_c4_s = self.pooling(c4)

        h_feature = torch.cat((pooled_c2_s.reshape(c1.shape[0], -1),
                               pooled_c3_s.reshape(c1.shape[0], -1),
                               pooled_c4_s.reshape(c1.shape[0], -1)), dim=1)

        return self.mlp(h_feature)

if __name__ == "__main__":
    model = nnMambaEncoder().cuda()

    input = torch.zeros((8, 1, 128, 128, 128)).cuda()
    output = model(input)
    print(output.shape)
