import os
from config import *
import torch
from vae import VAE, VAETest
from shapenetcore import get_random_models
from tqdm import tqdm

def save_latent_representation():
    device = 'cpu'
    vae  = VAETest().to(device)
    vae_weights_dir = f"{vae_checkpoint_dir}/best_model.pth"
    # vae_weights_dir = f'{config.vae_weights_dir}/{weights_dir}_aeroplanes_1.pth'
    checkpoint = torch.load(vae_weights_dir)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    os.makedirs(latent_dir, exist_ok=True)
    with torch.no_grad():
        for i, triplanes in tqdm(get_random_models(), desc=f"Progress"):
            triplanes = triplanes.to(device)
            triplanes = triplanes.squeeze()
            recon_x = vae.encode(triplanes)
            z_decoded = vae.decode(recon_x)
            z_decoded = z_decoded.cpu().permute(0, 2, 3, 1).contiguous().numpy()