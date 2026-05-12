import torch.nn as nn
from IPython import embed
import torch.nn.functional as F


class UNetLayer(nn.Module):
    """ U-Net Layer """

    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=False):
        super(UNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        conv1 = conv_op(num_channels_in, num_channels_out, kernel_size=3, padding=1)
        conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        bn1 = batch_nrom_op(num_channels_out)
        bn2 = batch_nrom_op(num_channels_out)
        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)


class ResUNetLayer(nn.Module):
    """ Residual U-Net Layer """

    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=True):
        super(ResUNetLayer, self).__init__()

        # 选择是使用2D卷积还是3D卷积
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_norm_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        # 定义两个卷积操作
        self.conv1 = conv_op(num_channels_in, num_channels_out, kernel_size=3, padding=1)
        self.conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        # 可选的批量归一化
        if batch_norm:
            self.bn1 = batch_norm_op(num_channels_out)
            self.bn2 = batch_norm_op(num_channels_out)
        else:
            self.bn1 = nn.Identity()  # 如果不使用批量归一化，使用恒等映射
            self.bn2 = nn.Identity()

        # 为了匹配输入和输出的通道数，需要一个1x1的卷积
        self.conv1x1 = conv_op(num_channels_in, num_channels_out,
                               kernel_size=1) if num_channels_in != num_channels_out else nn.Identity()

    def forward(self, x):
        # 残差连接
        residual = self.conv1x1(x)

        # 第一个卷积、批量归一化和ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 第二个卷积、批量归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 将残差连接添加到最后的输出
        out += residual
        out = F.relu(out)

        return out
