import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch
import bchlib
import random
import string

BCH_POLYNOMIAL = 137
BCH_BITS = 5

class bchcode:
    def bch_encode(self, secret_ori):
        bch = bchlib.BCH( BCH_POLYNOMIAL, BCH_BITS)

        if len(secret_ori) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return

        data = bytearray(secret_ori, 'utf-8')
        data += b' ' * (7 - len(data))
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0, 0, 0, 0])
        secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
        return secret
class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(224, 224), num_images=30000):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.num_images = num_images
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        # 使用ImageNet数据集
        self.imagenet_dataset = datasets.ImageFolder(root=self.data_path, transform=self.transform)
        self.imagenet_size = len(self.imagenet_dataset)

        self.to_tensor = transforms.ToTensor()



    def __getitem__(self, idx):
        img_cover_path = self.imagenet_dataset.imgs[idx][0]
        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()


        return img_cover, secret


    '''def __init__(self, data_path, secret_size=100, size=(224, 224), num_images=30000):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.num_images = num_images
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        # List all images in the folder
        self.image_paths = glob(os.path.join(self.data_path, '*'))
        self.num_images = min(num_images, len(self.image_paths))  # Limit to num_images if necessary

        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, idx):
        img_cover_path = self.image_paths[idx]
        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)

        # Generate random secret
        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return img_cover, secret'''

    def __len__(self):
        #return len(self.imagenet_dataset)

        return self.num_images




if __name__ == '__main__':
     #dataset = StegaData(data_path='F:\DeepLearning\watermarking\StegaStamp_pytorch-master\data\CN_2.5W')
     #print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)

    dataset = StegaData(data_path=r'./data/unlabeled2017/', secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
