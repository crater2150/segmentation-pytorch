import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum


class CustomModel(Enum):
    UNET = 'unet'
    AttentionNet = 'attentionunet'

    def __call__(self):
        return {'unet': UNet,
                'attentionunet': AttentionUnet,
                }[self.value]


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                 kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UpConv_woskip(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, stride):
        super(UpConv_woskip, self).__init__()

        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=stride, padding=0, stride=stride)  # calculate padding ?

    def forward(self, x):
        x = self.conv_trans(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size,
                 padding, stride, activation=None):
        super(UNet, self).__init__()
        self.activation = None
        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size,
                              padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels,
                          kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels,
                          kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
                          kernel_size, padding, stride)
        if activation is not None:
            self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        if self.activation is not None:
            x_out = F.log_softmax(self.out(x_up), 1)
        else:
            x_out = x_up
        return x_out


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding, stride)
        self.up1 = UpConv_woskip(8 * out_channels, out_channels, kernel_size, padding, stride=(8, 8))

    def forward(self, x):
        # print('attention')
        x = self.init_conv(x)
        # print(x.shape)
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        up1 = self.up1(x3)
        # print(up1.shape)

        return up1


class AttentionUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride, attention=True):
        super().__init__()
        self.attention = attention

        if attention:
            self.m1 = UNet(in_channels, out_channels, n_class, kernel_size, padding, stride)
            self.m2 = UNet(in_channels, out_channels, n_class, kernel_size, padding, stride)
            self.m3 = UNet(in_channels, out_channels, n_class, kernel_size, padding, stride)
            self.a1 = Attention(in_channels, out_channels, kernel_size, padding, stride)
            self.a2 = Attention(in_channels, out_channels, kernel_size, padding, stride)
            self.a3 = Attention(in_channels, out_channels, kernel_size, padding, stride)

            self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

            self.dpool1 = nn.AvgPool2d((2, 2))
            self.dpool2 = nn.AvgPool2d((2, 2))
            self.dpool3 = nn.AvgPool2d((2, 2))

            self.upscale = nn.UpsamplingNearest2d(())

        else:
            self.m1 = UNet(in_channels, out_channels, n_class, kernel_size, padding, stride, activation='softmax')

    def forward(self, x):
        if self.attention:
            x_d1 = self.dpool1(x)
            x_d2 = self.dpool2(x_d1)
            x_d3 = self.dpool3(x_d2)
            # Encoder
            # print(x.shape)
            x_m = self.m1(x)
            # print(x_m.shape)
            x_a = self.a1(x)
            # print(x_a.shape)

            x1 = x_m * x_a
            x2 = self.m2(x_d1) * self.a2(x_d1)
            x3 = self.m3(x_d2) * self.a3(x_d2)

            x4 = F.upsample_nearest(x1, x.shape[2:]) + F.upsample_nearest(x2, x.shape[2:]) + F.upsample_nearest(x3,
                                                                                                                x.shape[
                                                                                                                2:])
            x_out = F.log_softmax(self.out(x4), 1)
        else:
            x_out = self.m1(x)
        return x_out


def test():
    # Create 10-class segmentation dummy image and target
    nb_classes = 10
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, nb_classes, (1, 512, 512))

    model = UNet(in_channels=3,
                 out_channels=16,
                 n_class=10,
                 kernel_size=3,
                 padding=1,
                 stride=1)

    if torch.cuda.is_available():
        model = model.to('cuda')
        x = x.to('cuda')
        y = y.to('cuda')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss.item()))


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)
