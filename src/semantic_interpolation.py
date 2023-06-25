from manipulator import linear_interpolate
from tqdm import tqdm
import torch
import pickle
import torchvision
from PIL import Image
import numpy as np
from pathlib import Path

toPIL = torchvision.transforms.ToPILImage()
# Select a sample ID to perform interpolation


G = None
with open('stylegan2-celebahq-256x256.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latents = np.load('/content/pulse/latent_vectors/w.npy')
latents = latents.reshape((100, 14, 512))
latent_codes = latents[1]

out_path = Path("edit_smile")
out_path.mkdir(parents=True, exist_ok=True)
boundary = np.load('/content/pulse/our_boundaries/boundary_smile.npy')
i = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -40, 40, 5) for boun in boundary.reshape((14, 512))]

    j = 0
    for pol in interpolations:
        k = 0
        for p in pol:
            gen_im = (G.synthesis(
                torch.from_numpy(p.reshape((1, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"{i}_{j}_{k}.png")
            k += 1
        j += 1
    i += 1

out_path = Path("edit_bangs")
out_path.mkdir(parents=True, exist_ok=True)
boundary = np.load('/content/pulse/our_boundaries/boundary_bangs.npy')
latent_codes = latents[3]
i = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -40, 40, 5) for boun in boundary.reshape((14, 512))]

    j = 0
    for pol in interpolations:
        k = 0
        for p in pol:
            gen_im = (G.synthesis(
                torch.from_numpy(p.reshape((1, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"{i}_{j}_{k}.png")
            k += 1
        j += 1
    i += 1

out_path = Path("edit_blond")
out_path.mkdir(parents=True, exist_ok=True)
boundary = np.load('/content/pulse/our_boundaries/boundary_blondhair.npy')
latent_codes = latents[5]
i = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -40, 40, 5) for boun in boundary.reshape((14, 512))]

    j = 0
    for pol in interpolations:
        k = 0
        for p in pol:
            gen_im = (G.synthesis(
                torch.from_numpy(p.reshape((1, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"{i}_{j}_{k}.png")
            k += 1
        j += 1
    i += 1

out_path = Path("edit_black")
out_path.mkdir(parents=True, exist_ok=True)
boundary = np.load('/content/pulse/our_boundaries/boundary_blackhair.npy')
latent_codes = latents[7]
i = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -40, 40, 5) for boun in boundary.reshape((14, 512))]

    j = 0
    for pol in interpolations:
        k = 0
        for p in pol:
            gen_im = (G.synthesis(
                torch.from_numpy(p.reshape((1, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"{i}_{j}_{k}.png")
            k += 1
        j += 1
    i += 1

out_path = Path("edit_brown")
out_path.mkdir(parents=True, exist_ok=True)
boundary = np.load('/content/pulse/our_boundaries/boundary_brownhair.npy')
i = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -40, 40, 5) for boun in boundary.reshape((14, 512))]

    j = 0
    for pol in interpolations:
        k = 0
        for p in pol:
            gen_im = (G.synthesis(
                torch.from_numpy(p.reshape((1, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"{i}_{j}_{k}.png")
            k += 1
        j += 1
    i += 1
