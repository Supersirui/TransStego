# This code is adapted and modified from the StegaStamp_pytorch repository by Jisong Xie on GitHub.
# Source: https://github.com/JisongXie/StegaStamp_pytorch
# License: Please refer to the license in the GitHub repository for more details.

import os
import glob
import bchlib
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image, ImageOps
import torchvision
import torch
from torchvision import transforms, datasets
from vit import ViT
from einops import rearrange
import Encoder as transmodel
import cv2
import string
import random
import lpips


lpips_model = lpips.LPIPS(net='vgg').to('cuda')

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def calc_lpips_from_saved_images(host_img, container_img):

    host_img = cv2.cvtColor(host_img, cv2.COLOR_BGR2RGB)
    container_img = cv2.cvtColor(container_img, cv2.COLOR_BGR2RGB)


    transform = torchvision.transforms.ToTensor()
    host_img = transform(host_img).unsqueeze(0).to('cuda')
    container_img = transform(container_img).unsqueeze(0).to('cuda')


    host_img = (host_img * 2) - 1
    container_img = (container_img * 2) - 1


    lpips_value = lpips_model(host_img, container_img)
    return lpips_value.item()




def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255 / np.sqrt(mse))
        
def ms_ssim(img1, img2):
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

    
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  
    levels = weights.size

    mssim = np.zeros(levels)
    mcs = np.zeros(levels)

    for i in range(levels):
        ssim_map, cs_map = ssim(img1, img2)
        mssim[i] = np.mean(ssim_map)
        mcs[i] = np.mean(cs_map)

        
        img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    
    overall_mssim = np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])

    return overall_mssim


def ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=1.0):
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2

    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return ssim_map, cs_map



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=r'./Output')
    parser.add_argument('--secret', type=str, default='pengsen')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(os.path.join(args.images_dir, '*.*'), recursive=True)

    else:
        print('Missing input image')
        return
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    psnr_values = []
    lpips_list = []

    encoder = transmodel.TransUNet(img_dim=400, in_channels=3, out_channels=192, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=3)
    encoder.load_state_dict(torch.load(args.model))
    encoder.eval()
    if args.cuda:
        encoder = encoder.cuda()

    bch = bchlib.BCH( BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return



    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir+ '/resdiual/'):
            os.makedirs(args.save_dir+ '/resdiual/')
        if not os.path.exists(args.save_dir+ '/stego/'):
            os.makedirs(args.save_dir+ '/stego/')

        with torch.no_grad():
            for filename in files_list:
                data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
                ecc = bch.encode(data)
                packet = data + ecc

                packet_binary = ''.join(format(x, '08b') for x in packet)
                secret = [int(x) for x in packet_binary]
                secret.extend([0, 0, 0, 0])
                secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
                if args.cuda:
                    secret = secret.cuda()
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, size)
                image = to_tensor(image).unsqueeze(0)
                if args.cuda:
                    image = image.cuda()

                residual = encoder((secret, image))
                encoded = image + residual

                lpips = lpips_model(image, encoded)
                lpips = lpips.item()
                lpips_list.append(lpips)
                
                if args.cuda:
                    residual = residual.cpu()
                    encoded = encoded.cpu()

                encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

                residual = residual[0] + .5
                residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

                save_name = os.path.basename(filename).split('.')[0]

                im = Image.fromarray(encoded)
                im.save(args.save_dir + '/stego/' + save_name + '_hidden.jpg')

                im = Image.fromarray(residual)
                im.save(args.save_dir + '/resdiual/' + save_name + '_residual.jpg')
                img1 = np.array(Image.open(filename).resize((400, 400)).convert('YCbCr')).astype(np.float64)
                img2 = np.array(Image.open(args.save_dir + '/stego/' + save_name + '_hidden.jpg').convert('YCbCr')).astype(np.float64)
                PSNR_score = psnr(img1, img2)
                psnr_values.append(PSNR_score)
                img1 = cv2.imread(filename)
                img1 = cv2.resize(img1, (400, 400))
                img2 = cv2.imread(args.save_dir + '/stego/' + save_name + '_hidden.jpg')
                ms_ssim_score = ms_ssim(img1, img2)


                print(filename, 'PSNR', PSNR_score, "MS-SSIM score:", ms_ssim_score, 'lpips', lpips)

    print("Average PSNR:", np.mean(psnr_values))
    print("Average SSIM:", np.mean(ms_ssim_score))
    print("Average lpips:", np.mean(lpips_list))


if __name__ == "__main__":
    main()
