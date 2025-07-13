import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import voxel_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import config
from vae import *

def analyze_voxel_sparsity(dataloader):
    total_vox = 0
    total_nonzero = 0
    for vox in tqdm(dataloader, desc="Analyzing voxel sparsity"):
        total_vox += vox.numel()
        total_nonzero += (vox > 0).sum().item()
    ratio = total_nonzero / total_vox
    print(f"🔥 Overall Nonzero Voxel Ratio: {ratio:.6f}")

def train_test_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = voxel_dataloader()
    vae = VAE(latent_dim = 512*2*2*2).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    os.makedirs("./checkpoints/vae", exist_ok=True)
    num_epochs = 100
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_test_loss = float("inf")
    # Train VAE
    analyze_voxel_sparsity(train_loader)
    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for voxel_gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = voxel_gt.to(device)
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"🔥 Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f}")
        # Test VAE
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for voxel_gt in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x = voxel_gt.to(device)
                recon_x, mu, logvar = vae(x)
                loss = vae_loss(recon_x, x, mu, logvar)
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            scheduler.step(avg_test_loss)
            print(f"🧪 Test Loss = {avg_test_loss:.6f}")
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    "epoch": epoch,
                    "vae": vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "test_loss": best_test_loss
                }, f"./checkpoints/vae/best_model.pth")
                print(f"Saved best model at {epoch+1}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > early_stopping_patience:
                    print("Early stopping triggered")
                    break
        # torch.cuda.empty_cache()

def main():
    train_test_vae()

if __name__ == "__main__":
    main()