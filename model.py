# This code is adapted and modified from the StegaStamp_pytorch repository by Jisong Xie on GitHub.
# Source: https://github.com/JisongXie/StegaStamp_pytorch
# License: Please refer to the license in the GitHub repository for more details.

import sys
sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import cv2
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms
import bchlib
from Encoder import Encoder
import random

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
        return input.reshape(input.size(0), -1)



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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation=None))

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def generate_random_polygon(image_shape, num_vertices_range=(5, 12), target_area_ratio=0.1, random_seed=None):
    """
    Generate the vertex coordinates of a random irregular polygon, ensuring its area approximately occupies a specified ratio of the image area.

    Parameters:
        image_shape (tuple): Shape of the image (height, width).
        num_vertices_range (tuple): Range for the number of vertices (minimum, maximum).
        target_area_ratio (float): Ratio of the polygon area to the image area.
        random_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Vertex coordinates of the polygon with shape (number of vertices, 2).
    """
    np.random.seed(random_seed)
    height, width = image_shape
    total_area = height * width
    target_area = total_area * target_area_ratio  # Target covered area


    min_vertices, max_vertices = num_vertices_range
    num_vertices = np.random.randint(min_vertices, max_vertices + 1)

    # Randomly generate the center point
    center_x = np.random.randint(0, width)
    center_y = np.random.randint(0, height)


    max_radius = np.sqrt(target_area / np.pi)
    radius_variation = 0.3
    radii = np.random.uniform(0.7 * max_radius, max_radius, num_vertices)


    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))

    # Calculate vertex coordinates based on angles and radii
    vertices_x = (center_x + radii * np.cos(angles)).clip(0, width - 1).astype(int)
    vertices_y = (center_y + radii * np.sin(angles)).clip(0, height - 1).astype(int)

    return np.stack((vertices_x, vertices_y), axis=1)

def random_irregular_mask(encoded_image, image_shape, mask_value=0, random_seed=None, use_cuda=True, target_area_ratio=0.1):
    """
    Apply a random irregular-shaped mask to the input image, covering an area of the specified ratio.

    Parameters:
        encoded_image (torch.Tensor): Encoded image tensor with shape [batch_size, channels, height, width].
        image_shape (tuple): Shape of the image (height, width).
        mask_value (int): Pixel value for the masked region.
        random_seed (int): Random seed for reproducibility.
        use_cuda (bool): Whether to use CUDA for tensor operations.
        target_area_ratio (float): Ratio of the masked area to the image area.

    Returns:
        torch.Tensor: Image tensor after masking.
    """
    batch_size, channels, height, width = encoded_image.size()


    mask = np.ones((height, width), dtype=np.uint8)


    vertices = generate_random_polygon(image_shape, target_area_ratio=target_area_ratio, random_seed=random_seed)


    cv2.fillPoly(mask, [vertices.astype(np.int32)], 0)


    mask = torch.tensor(mask, dtype=torch.float32)
    if use_cuda:
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width).cuda()
    else:
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)


    encoded_image = (encoded_image + 1) / 2


    masked_image = encoded_image * mask + mask_value * (1 - mask)
    return masked_image


def transform_net(encoded_image, args, global_step, epoch):
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    N_blur = 7
    f = utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                 wmin_line=3)
    if args.cuda:
        f = f.cuda()
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # noise
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
    if args.cuda:
        sat_weight = sat_weight.cuda()
    encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, 400, 400])
    encoded_image = encoded_image.cuda()

    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)

    
    image_shape = (400, 400)
    if epoch > 10:
        if random.random() < 0.2:
            encoded_image = random_irregular_mask(encoded_image, image_shape)
            print('mask add')
    
    
    return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).detach().cpu().numpy()
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255 / np.sqrt(mse))






def string_to_bits_tensor(input_str):   
    byte_data = input_str.encode('utf-8') 
    bits_list = [int(bit) for byte in byte_data for bit in format(byte, '08b')]
    bits_tensor = torch.tensor(bits_list).view(1, -1)
    return bits_tensor





def build_model(encoder, decoder, discriminator, lpips_fn, secret_input, image_input, l2_edge_gain,
                borders, secret_size, M, loss_scales, yuv_scales, args, global_step, epoch, writer):
    test_transform = transform_net(image_input, args, global_step, epoch)

    input_warped = torchgeometry.warp_perspective(image_input, M[:, 1, :, :], dsize=(400, 400), flags='bilinear')
    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400),
                                                 flags='bilinear')
    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(residual_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')

    if borders == 'no_edge':
        encoded_image = image_input + residual


    if borders == 'no_edge':
        D_output_real, _ = discriminator(image_input)

        D_output_fake, D_heatmap = discriminator(encoded_image)

    else:
        D_output_real, _ = discriminator(input_warped)
        D_output_fake, D_heatmap = discriminator(encoded_warped)

    transformed_image = transform_net(encoded_image, args, global_step, epoch)
    decoded_secret = decoder(transformed_image)
    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)
    print('epoch:',epoch,'bit_acc:',bit_acc)

    normalized_input = image_input * 2 - 1
    normalized_encoded = encoded_image * 2 - 1
    lpips_loss = torch.mean(lpips_fn(normalized_input, normalized_encoded))

    cross_entropy = nn.BCELoss()
    if args.cuda:
        cross_entropy = cross_entropy.cuda()
    secret_loss = cross_entropy(decoded_secret, secret_input)


    size = (int(image_input.shape[2]), int(image_input.shape[3]))
    gain = 10
    falloff_speed = 4
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
        falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
        falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
    for j in range(int(falloff_im.shape[1] / falloff_speed)):
        falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
    falloff_im = 1 - falloff_im
    falloff_im = torch.from_numpy(falloff_im).float()
    if args.cuda:
        falloff_im = falloff_im.cuda()
    falloff_im *= l2_edge_gain

    encoded_image_yuv = color.rgb_to_yuv(encoded_image)
    image_input_yuv = color.rgb_to_yuv(image_input)
    im_diff = encoded_image_yuv - image_input_yuv
    im_diff += im_diff * falloff_im.unsqueeze_(0)
    yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
    yuv_scales = torch.Tensor(yuv_scales)
    if args.cuda:
        yuv_scales = yuv_scales.cuda()
    image_loss = torch.dot(yuv_loss, yuv_scales)

    Wasserstein_D = D_output_real - D_output_fake
    G_loss = -D_output_fake
    loss = loss_scales[0] * image_loss + loss_scales[1] * lpips_loss + loss_scales[
        2] * secret_loss  # _mean + loss_scales[3] * secret_loss_var
    if not args.no_gan:
        loss += loss_scales[3] * G_loss

    PSNR = psnr(image_input, encoded_image)


    writer.add_scalar('loss/Wasserstein_D', Wasserstein_D, global_step)
    writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
    writer.add_scalar('loss/secret_loss', secret_loss, global_step)
    writer.add_scalar('loss/secret_loss', secret_loss, global_step)

    writer.add_scalar('loss/PSNR', PSNR, global_step)
    writer.add_scalar('loss/G_loss', G_loss, global_step)
    writer.add_scalar('loss/loss', loss, global_step)

    writer.add_scalar('metric/bit_acc', bit_acc, global_step)
    writer.add_scalar('metric/str_acc', str_acc, global_step)
    if global_step % 20 == 0:
        writer.add_image('input/image_input', image_input[0], global_step)
        writer.add_image('input/image_warped', input_warped[0], global_step)
        writer.add_image('encoded/encoded_warped', encoded_warped[0], global_step)
        writer.add_image('encoded/residual_warped', residual_warped[0] + 0.5, global_step)
        writer.add_image('encoded/encoded_image', encoded_image[0], global_step)
        writer.add_image('transformed/transformed_image', transformed_image[0], global_step)
        writer.add_image('transformed/test', test_transform[0], global_step)

    return loss, secret_loss, Wasserstein_D, bit_acc, str_acc