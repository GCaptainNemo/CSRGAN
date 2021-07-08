import torch.nn as nn
# from torchvision.models import vgg19
import torch


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


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()
        # First layer
        self.extention = torch.ones([1, 1, 200, 200]).cuda()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64,
                                             kernel_size=9,
                                             stride=1,
                                             padding=4), nn.PReLU())
        # 16 layer residual layers get output HR img
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks,
                            nn.Conv2d(64, 3,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),
                            nn.PReLU())

        # #############################################################
        # Second conv layers get output physical parameters
        # ############################################################
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),  # inplace=True 直接在原地址上修改变量
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 4 x 100 x 100

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 8 x 50 x 50

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 16 x 25 x 25

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 32 x 12 x 12
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 12 * 12, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 3)
        )

    def forward(self, lr):
        out1 = self.conv1(lr)
        # N x 3 x H x w
        hr = self.res_blocks(out1)
        out2 = self.cnn1(hr)
        output = out2.view(out2.size()[0], -1)
        # N x 4 x 1 x 1
        # blur, scale, noise
        physical_par = self.fc1(output).reshape(-1, 3, 1, 1)
        # 1 x 1 x H x W

        # N x 4 x H x W
        physical_tensor = physical_par * self.extention
        dis_input = torch.cat([hr, physical_tensor, lr], dim=1)
        return dis_input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # N x 9 x 200 x 200
        self.cnn1_dis = nn.Sequential(
            nn.Conv2d(9, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),  # inplace=True 直接在原地址上修改变量
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 4 x 100 x 100

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 8 x 50 x 50

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 16 x 25 x 25

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 32 x 12 x 12
        )

        self.fc1_dis = nn.Sequential(
            nn.Linear(32 * 12 * 12, 500),
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

    def forward(self, hr_phy_lr):
        # N x 1
        output = self.forward_once(hr_phy_lr)
        return output
