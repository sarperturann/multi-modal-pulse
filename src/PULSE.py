from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import PIL
from PIL import Image
from loss import LossBuilder
from functools import partial
from drive import open_url
import dnnlib
import torch_utils
import pickle
import gc
import random


class PULSE(torch.nn.Module):
    def __init__(self, cache_dir, verbose=True):

        super(PULSE, self).__init__()

        gc.collect()

        torch.cuda.empty_cache()
        self.G = None
        with open('stylegan2-celebahq-256x256.pkl', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()
        c = None
        self.verbose = verbose

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        if self.verbose:
            print("\tLoading Mapping Network")

        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("The device is" + str(device))
        print(
            "############################# Assigned Device ##############################")

        torch.cuda.empty_cache()

        # Describing the model architecture to initialise with data from checkpoint
        self.resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
        self.resnext50_32x4d.fc = nn.Linear(2048, 40)
        ct = 0
        for child in self.resnext50_32x4d.children():
            ct += 1
            if ct < 6:
                for param in child.parameters():
                    param.requires_grad = False

        self.resnext50_32x4d.to(device)
        path_toLoad = "/content/gdrive/MyDrive/485/model_1_epoch.pt"
        checkpoint = torch.load(path_toLoad)

        # Initializing the model with the model parameters of the checkpoint.
        self.resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
        # Setting the model to be in evaluation mode. This will set the batch normalization parameters.
        self.resnext50_32x4d.eval()

        # Mapping network
        with torch.no_grad():
            torch.manual_seed(0)
            latent = torch.randn(
                (100000, 512), dtype=torch.float32, device="cuda")
            latent_out = torch.nn.LeakyReLU(5)(self.G.mapping(
                latent, c).cuda())
            self.gaussian_fit = {"mean": latent_out.mean(
                0), "std": latent_out.std(0)}

    def forward(self, ref_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate,
                latent_num,
                **kwargs):

        batch_size = ref_im.shape[0]
        images = []
        latents = []
        scores_smile = []
        scores_bang = []
        scores_blonde = []
        scores_brown = []
        scores_black = []

        for i in range(latent_num):
            seeds = [random.randint(0, 100000) for _ in range(latent_num)]

            for i, seed in enumerate(seeds):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

            if tile_latent:
                latent = torch.randn(
                    (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
            else:
                latent = torch.randn(
                    (batch_size, 14, 512), dtype=torch.float, requires_grad=True, device='cuda')

            # Generate list of noise tensors
            noise = []  # stores all of the noise tensors

            for i in range(14):
                # dimension of the ith noise tensor
                res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False

                noise.append(new_noise)

            var_list = [latent]+noise

            opt_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'sgdm': partial(torch.optim.SGD, momentum=0.9),
                'adamax': torch.optim.Adamax
            }
            opt_func = opt_dict[opt_name]
            opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

            schedule_dict = {
                'fixed': lambda x: 1,
                'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
                'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
            }
            schedule_func = schedule_dict[lr_schedule]
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt.opt, schedule_func)

            loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()

            min_loss = np.inf
            min_l2 = np.inf
            best_summary = ""
            start_t = time.time()
            gen_im = None

            if self.verbose:
                print("Optimizing")
            for j in range(steps):
                opt.opt.zero_grad()
                # Duplicate latent in case tile_latent = True
                if (tile_latent):
                    latent_in = latent.expand(-1, 14, -1)
                else:
                    latent_in = latent

                # Apply learned linear mapping to match latent distribution to that of the mapping network
                latent_in = self.lrelu(
                    latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])
                # Normalize image to [0,1] instead of [-1,1]
                gen_im = (self.G.synthesis(
                    latent_in, noise_mode='random', force_fp32=True) + 1) / 2

                # Calculate Losses
                loss, loss_dict = loss_builder(latent_in, gen_im)
                loss_dict['TOTAL'] = loss
                best_latent = None
                # Save best summary for log
                if (loss < min_loss):
                    min_loss = loss
                    best_summary = f'BEST ({j+1}) | '+' | '.join(
                        [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                    best_im = gen_im.clone()
                    best_latent = latent_in

                loss_l2 = loss_dict['L2']

                if (loss_l2 < min_l2):
                    min_l2 = loss_l2

                # Save intermediate HR and LR images
                if (save_intermediate):
                    yield (best_im.cpu().detach().clamp(0, 1), loss_builder.D(best_im).cpu().detach().clamp(0, 1))

                loss.backward()
                opt.step()
                scheduler.step()

            total_t = time.time()-start_t
            current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
            if self.verbose:
                print(best_summary+current_info)
            images.append((gen_im.clone().cpu().detach().clamp(0, 1)))

            device = torch.device('cuda')
            batch_imageTensor = torch.cuda.FloatTensor(1, 3, 256, 256)
            print(gen_im.shape)
            batch_imageTensor[0] = transforms.ToTensor()(
                transforms.ToPILImage()(gen_im[0]))

            batch_imageTensor.to(device)
            # Doing prediction on test data
            scores = self.resnext50_32x4d(batch_imageTensor)
            scores = scores.to('cpu')
            scores_smile.append(scores[0][31])
            scores_bang.append(scores[0][5])
            scores_blonde.append(scores[0][9])
            scores_black.append(scores[0][8])
            scores_brown.append(scores[0][11])
            latents.append(latent_in)

        yield (images, latents, scores_smile, scores_bang, scores_blonde, scores_brown, scores_black)
