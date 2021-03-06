import torch.nn as nn
# from torchvision.models import vgg19
import torch
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(Conv2dBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class Deconv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(Deconv2dBN, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class GeneratorResNet(nn.Module):
    def __init__(self):
        super(GeneratorResNet, self).__init__()
        self.layer1_conv = Conv2dBN(3, 8)
        self.layer2_conv = Conv2dBN(8, 16)
        self.layer3_conv = Conv2dBN(16, 32)
        self.layer4_conv = Conv2dBN(32, 64)
        self.layer5_conv = Conv2dBN(64, 128)
        self.layer6_conv = Conv2dBN(128, 64)
        self.layer7_conv = Conv2dBN(64, 32)
        self.layer8_conv = Conv2dBN(32, 16)
        self.layer9_conv = Conv2dBN(16, 8)
        # self.layer10_conv = nn.Conv2d(8, 6, kernel_size=3,
        #                               stride=1, padding=1, bias=True)
        self.layer10_conv = nn.Conv2d(8, 4, kernel_size=3,
                                      stride=1, padding=1, bias=True)
        # self.color_normalize_layer = nn.Sigmoid()

        self.deconv1 = Deconv2dBN(128, 64)
        self.deconv2 = Deconv2dBN(64, 32)
        self.deconv3 = Deconv2dBN(32, 16)
        self.deconv4 = Deconv2dBN(16, 8)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        # hr = self.color_normalize_layer(outp[:, :3, :, :])
        hr = outp[:, :3, :, :]
        max_ = torch.max(hr)
        min_ = torch.min(hr)
        hr = (hr - min_) / (max_ - min_)
        phy = outp[:, 3:, :, :]
        physical_par = torch.mean(torch.mean(phy, dim=3), dim=2)

        # outp = self.sigmoid(outp)
        return hr, physical_par


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # N x 3 x 200 x 200
        self.cnn1_dis = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),  # inplace=True ?????????????????????????????????
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 4 x 256 x 256

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 8 x 128 x 128

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 16 x 64 x 64

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 32 x 32 x 32
        )

        self.fc1_dis = nn.Sequential(
            nn.Linear(32 * 32 * 32, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn1_dis(x)
        # reshape N x d feature
        output = output.view(output.size()[0], -1)
        output = self.fc1_dis(output)
        return output

    def forward(self, hr):
        # N x 1
        output = self.forward_once(hr)
        return output
