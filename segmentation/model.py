import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
from segmentation.util import logger


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


def pad(tensor, factor=32):
    shape = list(tensor.shape)[2:]
    h_dif = factor - (shape[0] % factor)
    x_dif = factor - (shape[1] % factor)
    x_dif = x_dif if factor != x_dif else 0
    h_dif = h_dif if factor != h_dif else 0
    augmented_image = tensor
    if h_dif != 0 or x_dif != 0:
        augmented_image = torch.nn.functional.pad(input=tensor, pad=[0, x_dif, 0, h_dif])
    return augmented_image


def unpad(tensor, o_shape):
    output = tensor[:, :, :o_shape[0], :o_shape[1]]
    return output


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
    def __init__(self, in_channels=3, out_channels=16, n_class=10, kernel_size=3, padding=1, stride=1, activation=None,
                 depth=3, encoder_filter=None, decoder_filter=None):
        super(UNet, self).__init__()
        if encoder_filter is None:
            encoder_filter = [16, 32, 64, 128]
        if decoder_filter is None:
            decoder_filter = [16, 32, 64, 128]
        assert ((depth + 1) == len(encoder_filter)), "Length of encoder filter must match encoder depth + 1"
        assert ((depth + 1) == len(decoder_filter)), "Length of decoder filter must match encoder depth + 1"
        self.activation = None
        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        for i in range(1, depth + 1):
            self.down.append(DownConv(encoder_filter[i - 1], encoder_filter[i], kernel_size, padding, stride))
            self.up.append(UpConv(encoder_filter[i], encoder_filter[i - 1], encoder_filter[i - 1],
                                  kernel_size, padding, stride))
        # for i in range(1, depth + 1):
        #    self.down.append(DownConv(2 ** (i - 1) * out_channels, 2 ** i * out_channels, kernel_size,
        #                              padding, stride))
        #    self.up.append(UpConv(2 ** i * out_channels, 2 ** (i - 1) * out_channels, 2 ** (i - 1) * out_channels,
        #                          kernel_size, padding, stride))

        if activation is not None:
            self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)

        res_d = [x]
        for layer in self.down:
            res_d.append(layer(res_d[-1]))

        res_u = [res_d[-1]]
        s = 2
        for layer in reversed(self.up):
            res_u.append(layer(res_u[-1], res_d[-s]))
            s += 1
        if self.activation is not None:
            x_out = F.log_softmax(self.out(res_u[-1]), 1)
        else:
            x_out = res_u[-1]
        return x_out


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, depth=3, encoder_filter=None):
        super().__init__()
        if encoder_filter is None:
            encoder_filter = [16, 32, 64, 128]
        assert ((depth + 1) == len(encoder_filter)), "Length of encoder filter must match encoder depth + 1"
        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)
        self.down = nn.ModuleList([])
        for i in range(1, depth + 1):
            self.down.append(DownConv(encoder_filter[i - 1], encoder_filter[i], kernel_size, padding, stride))

        # for i in range(1, depth + 1):
        #    self.down.append(DownConv(2 ** (i - 1) * out_channels, 2 ** i * out_channels, kernel_size, padding, stride))
        self.up1 = UpConv_woskip(8 * out_channels, out_channels, kernel_size, padding, stride=(2 ** depth, 2 ** depth))

    def forward(self, x):
        x = self.init_conv(x)
        for layer in self.down:
            x = layer(x)
        up1 = self.up1(x)

        return up1


class AttentionUnet(nn.Module):

    def __init__(self, in_channels=3, out_channels=16, n_class=10, kernel_size=3, padding=1, stride=1, attention=True,
                 encoder_depth=3, attention_depth=3, attention_encoder_depth=3, encoder_filter=None,
                 decoder_filter=None,
                 attention_encoder_filter=None):
        super().__init__()
        self.attention = attention
        self.unets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.attention_depth = attention_depth
        if attention:
            for i in range(attention_depth):
                self.unets.append(UNet(in_channels=in_channels, out_channels=out_channels, n_class=n_class,
                                       kernel_size=kernel_size, padding=padding, stride=stride, depth=encoder_depth,
                                       encoder_filter=encoder_filter, decoder_filter=decoder_filter))
                self.attentions.append(
                    Attention(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, depth=attention_encoder_depth,
                              encoder_filter=attention_encoder_filter))
            self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)
            self.dpool = nn.AvgPool2d((2, 2))
        else:
            self.m1 = UNet(in_channels=in_channels, out_channels=out_channels, n_class=n_class,
                           kernel_size=kernel_size, padding=padding, stride=stride, depth=encoder_depth,
                           encoder_filter=encoder_filter, decoder_filter=decoder_filter)

    def forward(self, x):
        if self.attention:
            resized_images = [x]
            for t in range(self.attention_depth - 1):
                resized_images.append(self.dpool(resized_images[-1]))
            results = []
            for ind, t in enumerate(resized_images):
                results.append(self.attentions[ind](t) * self.unets[ind](t))
            end_res = F.upsample_nearest(results[0], x.shape[2:])
            for t in results[1:]:
                end_res += F.upsample_nearest(t, x.shape[2:])
            x_out = end_res

        else:
            x_out = self.m1(x)
        return x_out


def test():
    # Create 10-class segmentation dummy image and target
    nb_classes = 10
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, nb_classes, (1, 512, 512))

    model = CustomModel("attentionunet")()()
    if torch.cuda.is_available():
        model = model.to('cuda')
        x = x.to('cuda')
        y = y.to('cuda')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        model = model.to('cuda')
        output = model(x)
        output.to('cuda')
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss.item()))
