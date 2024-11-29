# This code is adapted and modified from the StegaStamp_pytorch repository by Jisong Xie on GitHub.
# Source: https://github.com/JisongXie/StegaStamp_pytorch
# License: Please refer to the license in the GitHub repository for more details.

import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

BCH_POLYNOMIAL = 137
BCH_BITS = 5

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1, padding=0):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)




class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(320000, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image

class Decoder(nn.Module):
    def __init__(self, secret_size=100):
        super(Decoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 48, 1, activation='relu'),
            Conv2D(48, 48, 3, activation='relu'),
            # nn.Dropout(p=0.5),
            Conv2D(48, 48, 3, activation='relu'),
            Conv2D(48, 48, 1, activation='relu'),
            Conv2D(48, 48, 3, activation='relu'),
            Conv2D(48, 48, 3, activation='relu'),
            Conv2D(48, 24, 1, activation='relu'),
            # nn.Dropout(p=0.5),
            Conv2D(24, 24, 3, activation='relu'),
            Conv2D(24, 24, 3, activation='relu'),
            Conv2D(24, 1, 1, activation='relu'),
            #nn.AdaptiveAvgPool2d(10),  
            Flatten(),
            # nn.Dropout(p=0.5),
            Dense(160000, secret_size, activation=None)
            )

    def forward(self, image):
        image = image - .5
        transformed_image = self.stn(image)
        return torch.sigmoid(self.decoder(transformed_image))
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    decoder = Decoder()
    decoder.load_state_dict(torch.load(args.model))
    decoder.eval()
    if args.cuda:
        decoder = decoder.cuda()

    bch = bchlib.BCH( BCH_POLYNOMIAL, BCH_BITS)

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, size)
            image = to_tensor(image).unsqueeze(0)
            if args.cuda:
                image = image.cuda()

            secret = decoder(image)
            # secret = secret + 0.5
            if args.cuda:
                secret = secret.cpu()
            secret = np.array(secret[0])
            secret = np.round(secret)

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            bitflips = bch.decode_inplace(data, ecc)

            if bitflips != -1:
                try:
                    code = data.decode("utf-8", 'ignore')
                    print(filename, code)
                    continue
                except Exception as e:
                    print(filename, f"Error decoding data: {e}")
                    continue
            else:
                try:
                    code = data.decode("utf-8", 'ignore')
                    print(filename, code)
                    continue
                except Exception as e:
                    print(filename, f"Error decoding data: {e}")
                    continue
            print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
