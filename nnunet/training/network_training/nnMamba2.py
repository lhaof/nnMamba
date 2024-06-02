import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from nnunet.network_architecture.neural_network import SegmentationNetwork

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x)
            out += global_att
        if self.downsample is not None:
            # if self.mamba_layer is not None:
            #     global_att = self.mamba_layer(x)
            #     identity = self.downsample(x+global_att)
            # else:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim) # LayerNorm
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
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print('x_norm.dtype', x_norm.dtype)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
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

class Attentionlayer(nn.Module):
    def __init__(self,dim,r=16,act='relu'):
        super(Attentionlayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim//r))
        self.layer2 = nn.Linear(int(dim//r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp):
        att = self.sigmoid(self.layer2(self.relu(self.layer1(inp))))
        return att.unsqueeze(-1)


class nnMambaSeg(SegmentationNetwork):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaSeg, self).__init__()
        self.do_ds = True
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.att1 = Attentionlayer(channels)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2, mamba_layer=MambaLayer(channels*2))

        self.att2 = Attentionlayer(channels*2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, mamba_layer=MambaLayer(channels*4))
        # self.mamba_layer_2 = MambaLayer(channels*4)

        self.att3 = Attentionlayer(channels*4)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, mamba_layer=MambaLayer(channels*8))
        # self.mamba_layer_3 = MambaLayer(channels*8)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)


        self.ds1_cls_conv = nn.Conv3d(32, 4, kernel_size=1)
        self.ds2_cls_conv = nn.Conv3d(64, 4, kernel_size=1)
        self.ds3_cls_conv = nn.Conv3d(128, 4, kernel_size=1)

    def forward(self, x):
        c1 = self.in_conv(x)
        scale_f1 = self.att1(self.pooling(c1).reshape(c1.shape[0], c1.shape[1])).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(self.pooling(c2).reshape(c2.shape[0], c2.shape[1])).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)

        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(self.pooling(c3).reshape(c3.shape[0], c3.shape[1])).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3*scale_f3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2*scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1*scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)

        logits = []
        logits.append(c8)
        logits.append(self.ds1_cls_conv(c7))
        logits.append(self.ds2_cls_conv(c6))
        logits.append(self.ds3_cls_conv(c5))

        if self.do_ds:
            return logits
        else:
            return logits[0]

        # return c8


class nnMambaSegSL(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaSegSL, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)
        self.pooling =  nn.AdaptiveAvgPool3d((1, 1, 1))

        self.att1 = Attentionlayer(channels)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.att2 = Attentionlayer(channels*2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.att3 = Attentionlayer(channels*4)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        scale_f1 = self.att1(self.pooling(c1).reshape(c1.shape[0], c1.shape[1])).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(self.pooling(c2).reshape(c2.shape[0], c2.shape[1])).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)

        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(self.pooling(c3).reshape(c3.shape[0], c3.shape[1])).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3*scale_f3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2*scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1*scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8


# class nnMambaSeg(nn.Module):
#     def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
#         super(nnMambaSeg, self).__init__()
#         self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
#         # self.mamba_layer_stem = MambaLayer(channels)

#         self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
#         self.mamba_layer_1 = MambaLayer(channels*2)

#         self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
#         self.mamba_layer_2 = MambaLayer(channels*4)

#         self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
#         self.mamba_layer_3 = MambaLayer(channels*8)

#         self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv5 = DoubleConv(channels * 12, channels * 4)
#         self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv6 = DoubleConv(channels * 6, channels * 2)
#         self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv7 = DoubleConv(channels * 3, channels)
#         self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv8 = DoubleConv(channels, number_classes)

#     def forward(self, x):
#         c1 = self.in_conv(x)
#         # c1_s = self.mamba_layer_stem(c1) + c1
#         c2 = self.layer1(c1)
#         c2_s = self.mamba_layer_1(c2) + c2
#         c3 = self.layer2(c2_s)
#         c3_s = self.mamba_layer_2(c3) + c3
#         c4 = self.layer3(c3_s)
#         c4_s = self.mamba_layer_3(c4) + c4

#         up_5 = self.up5(c4_s)
#         merge5 = torch.cat([up_5, c3_s], dim=1)
#         c5 = self.conv5(merge5)
#         up_6 = self.up6(c5)
#         merge6 = torch.cat([up_6, c2_s], dim=1)
#         c6 = self.conv6(merge6)
#         up_7 = self.up7(c6)
#         merge7 = torch.cat([up_7, c1], dim=1)
#         c7 = self.conv7(merge7)
#         up_8 = self.up8(c7)
#         c8 = self.conv8(up_8)
#         return c8


class nnMambaEncoder(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=1):
        super(nnMambaEncoder, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.pooling =  nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp = nn.Sequential(nn.Linear(channels*8, channels), nn.ReLU(), nn.Dropout(0.5), nn.Linear(channels, number_classes))

        # self.sig = nn.Sigmoid()


    def forward(self, x):
        c1 = self.in_conv(x)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        c3 = self.layer2(c2)
        c3_s = self.mamba_layer_2(c3) + c3
        c4 = self.layer3(c3)
        c4_s = self.mamba_layer_3(c4) + c4
        c5 = self.pooling(c4).view(c4.shape[0], -1)
        c5 = self.mlp(c5)
        # c5 = self.sig(c5)
        return c5


if __name__ == "__main__":
    model = nnMambaSeg().cuda()
    # model = nnMambaEncoder().cuda()

    input = torch.zeros((8, 1, 128, 128, 128)).cuda()
    output = model(input)
    print(output.shape)