"""  
unet:
    input: x=[B, 7, 100, 100] c通道数
    encoder: z=[B, 3, 1600, 1600] 每个块的输出
        transConv2d
    decoder: y=[B,3,1000,1000] c1通道数
        Conv2d
   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AED(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(AED, self).__init__()
        hidden_channels = [input_shape[0], 6, 5, 4, output_shape[0]]
        self.encoder = Encoder(hidden_channels)
        self.decoder = Decoder(output_shape)

    def forward(self, x):
        # [b, c, h, w]
        z = self.encoder(x)
        y = self.decoder(z)
        return y

class Encoder(nn.Module):
    def __init__(self, hidden_channels):
        super(Encoder, self).__init__()
        self.block1 = BlockConvTrans(hidden_channels[0], hidden_channels[1])
        self.block2 = BlockConvTrans(hidden_channels[1], hidden_channels[2])
        self.block3 = BlockConvTrans(hidden_channels[2], hidden_channels[3])
        self.block4 = BlockConvTrans(hidden_channels[3], hidden_channels[4])

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x4

class Decoder(nn.Module):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.block1 = BlockConv(output_shape[0], output_shape[0], 3, 1, 2)
        self.block2 = BlockConv(2*output_shape[0], output_shape[0], 3, 1, 1)

    def forward(self, x):
        z1 = self.block1(x)
        h1 = F.interpolate(z1, self.output_shape[1:], mode='bilinear', align_corners=True)
        h2 = F.interpolate(x, self.output_shape[1:], mode='bilinear', align_corners=True)
        y = self.block2(torch.cat((h1, h2), 1))
        return y
                
class BlockConvTrans(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channel, out_channel, 3, padding=1, stride=2)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel+out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        bi_xshape = (b, c, 2*h, 2*w)
        z1 = self.bn(self.convTrans(x, output_size=bi_xshape))
        z2 = F.interpolate(x, bi_xshape[2:], mode='bilinear', align_corners=True)
        y = self.conv(torch.cat((z1, z2), 1))
        return y
   
class BlockConv(nn.Module):

    def __init__(self, in_channel, out_channel,kernel_size, padding, stride):
        super().__init__()
        self.patch_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        return self.patch_conv(x)
    