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
    checkpoint = torch.load(vae_weights_dir)
    vae.load_state_dict(checkpoint['vae'])
    vae.eval()
    os.makedirs(latent_dir, exist_ok=True)
    with torch.no_grad():
        for path in tqdm(get_random_models(), desc=f"Progress"):
            if f"{"_".join(path.split("/"))}.pt" not in os.listdir(latent_dir):
                voxel = torch.load(f"{voxel_dir}/{"_".join(path.split("/"))}.pt", weights_only=False)
                h = vae.encoder_conv(voxel)
                mu, logvar = vae.conv_mu(h), vae.conv_logvar(h)
                z = vae.reparameterize(mu, logvar)
                torch.save(z, f"{latent_dir}/{"_".join(path.split("/"))}.pt")

def main():
    save_latent_representation()

if __name__ == "__main__":
    main()