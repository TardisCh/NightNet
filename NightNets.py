from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import torchvision.ops as ops

from models.nets.base.cbam import CBAMBlock


class conv_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(up_conv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upconv(x)
        return x


class DeformableFusion(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, kernel_size=3):
        super(DeformableFusion, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels_1 + in_channels_2, 2 * kernel_size * kernel_size, kernel_size=3,
                                     padding=1)

        self.deform_conv = ops.DeformConv2d(in_channels_1, out_channels, kernel_size=kernel_size, padding=1)
        self.deform_bn = nn.BatchNorm2d(out_channels)
        self.deform_relu = nn.ReLU(inplace=True)

        self.fusion_conv = nn.Conv2d(out_channels + in_channels_2, out_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # Generate offset
        offset = self.offset_conv(torch.cat([x1, x2], dim=1))

        # Apply deformable convolution
        x1_deform = self.deform_conv(x1, offset)
        x1_deform = self.deform_bn(x1_deform)
        x1_deform = self.deform_relu(x1_deform)

        # Fuse features
        x_fused = self.fusion_conv(torch.cat([x1_deform, x2], dim=1))
        x_fused = self.fusion_bn(x_fused)
        x_fused = self.fusion_relu(x_fused)

        return x_fused


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class NightNets(nn.Module):

    def __init__(self, in_channel=1, out_channel=1):
        super(NightNets, self).__init__()

        n1 = 16  # Base channel width for hidden feature maps; can be adjusted
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Downsampling layers
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature extraction blocks
        self.Conv1 = conv_block(in_channel, filters[0])

        self.conv_corr_lsat_n1 = nn.Sequential(
            nn.Conv2d(6, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_corr_is_n1 = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.conv_ld_n1 = DeformableFusion(in_channels_1=filters[0], in_channels_2=filters[0], out_channels=filters[0] * 2)
        self.conv_merge1 = DeformableFusion(in_channels_1=filters[0], in_channels_2=filters[0] * 2,
                                            out_channels=filters[0])

        self.Conv2 = conv_block(filters[0], filters[1])
        self.conv_corr_lsat_n2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_corr_is_n2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        self.conv_ld_n2 = DeformableFusion(in_channels_1=filters[1], in_channels_2=filters[1],
                                           out_channels=filters[1] * 2)
        self.conv_merge2 = DeformableFusion(in_channels_1=filters[1], in_channels_2=filters[1] * 2,
                                            out_channels=filters[1])

        self.Conv3 = conv_block(filters[1], filters[2])
        self.conv_corr_lsat_n3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_corr_is_n3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )

        self.conv_ld_n3 = DeformableFusion(in_channels_1=filters[2], in_channels_2=filters[2],
                                           out_channels=filters[2] * 2)
        self.conv_merge3 = DeformableFusion(in_channels_1=filters[2], in_channels_2=filters[2] * 2,
                                            out_channels=filters[2])

        self.Conv4 = conv_block(filters[2], filters[3])
        self.conv_corr_lsat_n4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_corr_is_n4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )

        self.conv_ld_n4 = DeformableFusion(in_channels_1=filters[3], in_channels_2=filters[3],
                                           out_channels=filters[3] * 2)
        self.conv_merge4 = DeformableFusion(in_channels_1=filters[3], in_channels_2=filters[3] * 2,
                                            out_channels=filters[3])

        self.Conv5 = conv_block(filters[3], filters[4])
        self.conv_corr_lsat_n5 = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )
        self.conv_corr_is_n5 = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        self.conv_ld_n5 = DeformableFusion(in_channels_1=filters[4], in_channels_2=filters[4],
                                           out_channels=filters[4] * 2)
        self.conv_merge5 = DeformableFusion(in_channels_1=filters[4], in_channels_2=filters[4] * 2,
                                            out_channels=filters[4])

        self.aspp_strengthen = ASPP(filters[4], [1, 6, 12, 18], filters[4])

        # Upsampling layers
        self.up5 = up_conv(filters[4], filters[3])
        self.cbam_up5 = CBAMBlock(channel=filters[4], reduction=32, kernel_size=3)
        self.up_conv5 = conv_block(filters[4], filters[3])

        self.up4 = up_conv(filters[3], filters[2])
        self.cbam_up4 = CBAMBlock(channel=filters[3], reduction=16, kernel_size=3)
        self.up_conv4 = conv_block(filters[3], filters[2])

        self.up3 = up_conv(filters[2], filters[1])
        self.cbam_up3 = CBAMBlock(channel=filters[2], reduction=8, kernel_size=3)
        self.up_conv3 = conv_block(filters[2], filters[1])

        self.up2 = up_conv(filters[1], filters[0])
        self.cbam_up2 = CBAMBlock(channel=filters[1], reduction=4, kernel_size=3)
        self.up_conv2 = conv_block(filters[1], filters[0])

        self.cbam_out = CBAMBlock(channel=filters[0], reduction=16, kernel_size=3)

        self.conv_out = nn.Sequential(
            nn.Conv2d(filters[0], filters[0] // 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(filters[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0] // 2, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        )

        self.conv_edge = nn.Sequential(
            nn.Conv2d(filters[4], filters[4] // 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(filters[4] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4] // 2, 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        )

        self.conv_mask = nn.Sequential(
            nn.Conv2d(filters[4], filters[4] // 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(filters[4] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4] // 2, 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        )

    def forward(self, x_LR, x_LSAT, x_IS):
        x_LR = F.interpolate(x_LR, (256, 256), mode='bilinear', align_corners=True)  # resampling
        # encoder
        e1 = self.Conv1(x_LR)
        lsat1 = self.conv_corr_lsat_n1(x_LSAT)
        is1 = self.conv_corr_is_n1(x_IS)
        ld1 = self.conv_ld_n1(lsat1, is1)
        e1 = self.conv_merge1(e1, ld1)

        e2 = self.Conv2(self.Maxpool1(e1))
        lsat2 = self.conv_corr_lsat_n2(lsat1)
        is2 = self.conv_corr_is_n2(is1)
        ld2 = self.conv_ld_n2(lsat2, is2)
        e2 = self.conv_merge2(e2, ld2)

        e3 = self.Conv3(self.Maxpool2(e2))
        lsat3 = self.conv_corr_lsat_n3(lsat2)
        is3 = self.conv_corr_is_n3(is2)
        ld3 = self.conv_ld_n3(lsat3, is3)
        e3 = self.conv_merge3(e3, ld3)

        e4 = self.Conv4(self.Maxpool3(e3))
        lsat4 = self.conv_corr_lsat_n4(lsat3)
        is4 = self.conv_corr_is_n4(is3)
        ld4 = self.conv_ld_n4(lsat4, is4)
        e4 = self.conv_merge4(e4, ld4)

        e5 = self.Conv5(self.Maxpool4(e4))
        lsat5 = self.conv_corr_lsat_n5(lsat4)
        is5 = self.conv_corr_is_n5(is4)
        ld5 = self.conv_ld_n5(lsat5, is5)
        e5 = self.conv_merge5(e5, ld5)

        mask = self.aspp_strengthen(e5)

        # decoder
        d5 = torch.cat((e4, self.up5(e5)), dim=1)
        d5 = self.cbam_up5(d5)
        d5 = self.up_conv5(d5)
        d4 = torch.cat((e3, self.up4(d5)), dim=1)
        d4 = self.cbam_up4(d4)
        d4 = self.up_conv4(d4)
        d3 = torch.cat((e2, self.up3(d4)), dim=1)
        d3 = self.cbam_up3(d3)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((e1, self.up2(d3)), dim=1)
        d2 = self.cbam_up2(d2)
        d2 = self.up_conv2(d2)

        d_out = self.cbam_out(d2)
        output = self.conv_out(d_out)

        edge_out = self.conv_edge(mask)
        edge_out = F.interpolate(edge_out, (256, 256), mode='bilinear', align_corners=True)  # resampling

        mask_out = self.conv_mask(mask)
        mask_out = F.interpolate(mask_out, (256, 256), mode='bilinear', align_corners=True)  # resampling

        return output, edge_out, mask_out
