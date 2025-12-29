import torch
import torch.nn as nn
import torch.nn.functional as F


# MASAG ---
class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)
        x2 = self.maxpool(x_)
        cat = torch.cat((x, x2), dim=1)
        proj = self.proj(cat)
        return proj


class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        self.reduction = 1 if reduction == None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // self.reduction, kernel_size=1),
            nn.BatchNorm2d(num_features=dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x


class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g):
        x = self.local(x)
        g = self.global_(g)
        fuse = self.bn(x + g)
        return fuse


class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1))

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()

        multi = self.multi(x, g)
        multi = self.selection(multi)

        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)

        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_

        x_att = x_att + x_
        g_att = g_att + g_

        # Bidirectional Interaction
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att

        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))
        weighted = projected * x_
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        return y


# ---  (ResBlock) ---
class BasicResBlock(nn.Module):


    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):


    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        #
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #
        self.res_block = BasicResBlock(out_channels, out_channels)

    def forward(self, x, skip=None):
        #
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        #
        if skip is not None:
            #
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        #
        x = self.fusion_conv(x)
        x = self.res_block(x)
        return x


# ---
class UMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_c=64):
        super().__init__()

        # --- Encoder (D-Path) ---
        # Stem (D1): 7x7 Conv + MaxPool -> 1/4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=7, stride=2, padding=3, bias=False),  # H/2
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4
        )

        # Encoder Stages
        self.d2 = BasicResBlock(base_c, base_c, stride=1)  # H/4, 64
        self.d3 = BasicResBlock(base_c, base_c * 2, stride=2)  # H/8, 128
        self.d4 = BasicResBlock(base_c * 2, base_c * 4, stride=2)  # H/16, 256
        self.d5 = BasicResBlock(base_c * 4, base_c * 8, stride=2)  # H/32, 512

        # --- Bottleneck (MASAG) ---
        #
        self.masag = MultiScaleGatedAttn(dim=base_c * 8)

        # --- Decoder (E-Path) ---
        # E5 -> E4: Up(512) + Cat(D4:256) -> Out(256)
        self.e4 = DecoderBlock(in_channels=base_c * 8, skip_channels=base_c * 4, out_channels=base_c * 4)

        # E4 -> E3: Up(256) + Cat(D3:128) -> Out(128)
        self.e3 = DecoderBlock(in_channels=base_c * 4, skip_channels=base_c * 2, out_channels=base_c * 2)

        # E3 -> E2: Up(128) + Cat(D2:64) -> Out(64)
        self.e2 = DecoderBlock(in_channels=base_c * 2, skip_channels=base_c, out_channels=base_c)

        # --- Final Head (E1) ---
        # E2
        #
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, num_classes, kernel_size=1)  # Final mapping to classes
        )

    def forward(self, x):
        # --- Encoder ---
        x_d1 = self.stem(x)  # H/4
        x_d2 = self.d2(x_d1)  # H/4
        x_d3 = self.d3(x_d2)  # H/8
        x_d4 = self.d4(x_d3)  # H/16
        x_d5 = self.d5(x_d4)  # H/32

        # --- Bottleneck ---
        #
        x_mid = self.masag(x_d5, x_d5)

        # --- Decoder ---
        # E5
        x_e4 = self.e4(x_mid, x_d4)  # H/16
        x_e3 = self.e3(x_e4, x_d3)  # H/8
        x_e2 = self.e2(x_e3, x_d2)  # H/4

        # --- Output ---
        out = self.final_up(x_e2)  # H, 3 Channels

        return out


if __name__ == "__main__":

    model = UMNet(in_channels=3, num_classes=2, base_c=64).cuda()


    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
