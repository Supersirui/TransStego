# This code is adapted and modified from the StegaStamp_pytorch repository by Jisong Xie on GitHub.
# Source: https://github.com/JisongXie/StegaStamp_pytorch
# License: Please refer to the license in the GitHub repository for more details.

import os
import yaml
import random
import model as model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import bchlib
import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import lpips
import Encoder as transmodel

BCH_POLYNOMIAL = 137
BCH_BITS = 5


with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)





def main():
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


    encoder = transmodel.TransUNet(img_dim=400, in_channels=3, out_channels=192, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=3)
    decoder = model.Decoder(secret_size=args.secret_size)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex = lpips_alex.cuda()
    if args.pretrained is not None:
        print("Pretrained path is not None:", args.pretrained)
        encoder_path = os.path.join(args.checkpoints_path, 'mask', 'en_name.pth')
        decoder_path = os.path.join(args.checkpoints_path, 'mask', 'de_name.pth')
        discriminator_path = os.path.join(args.checkpoints_path, 'mask', 'discriminator.pth')
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
      
    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = 400
    width = 400


    global_step = 0


    total_steps_per_epoch = len(dataset) // args.batch_size
    total_epochs = args.num_epochs
    for epoch in range(total_epochs):
        for step, (image_input, secret_ori) in enumerate(dataloader):

            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)

            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, Wasserstein_D, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator,
                                                                                   lpips_alex,
                                                                                   secret_input, image_input,
                                                                                   args.l2_edge_gain, args.borders,
                                                                                   args.secret_size, Ms, loss_scales,
                                                                                   yuv_scales, args, global_step, epoch,
                                                                                   writer)


            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()


            print('Loss = {:.4f}'.format(loss))
            if global_step % 5000 == 0:
                save_path_checkpoint_encoder = os.path.join(args.checkpoints_path, args.exp_name, 'en_name.pth')
                os.makedirs(os.path.dirname(save_path_checkpoint_encoder), exist_ok=True)
                save_path_checkpoint_decoder = os.path.join(args.checkpoints_path, args.exp_name, 'de_name.pth')
                os.makedirs(os.path.dirname(save_path_checkpoint_decoder), exist_ok=True)
                save_path_checkpoint_discriminator = os.path.join(args.checkpoints_path, args.exp_name, 'discriminator.pth')
                os.makedirs(os.path.dirname(save_path_checkpoint_discriminator), exist_ok=True)
                torch.save(encoder.state_dict(), os.path.join(save_path_checkpoint_encoder))
                torch.save(decoder.state_dict(), os.path.join(save_path_checkpoint_decoder))
                torch.save(discriminator.state_dict(), os.path.join(save_path_checkpoint_discriminator))
            

    writer.close()
    torch.save(encoder.state_dict(), os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(args.saved_models, "decoder.pth"))


if __name__ == '__main__':
    main()
