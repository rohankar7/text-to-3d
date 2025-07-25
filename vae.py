import torch
from torch import nn
import torch.nn.functional as F
from config import *


class VAE(nn.Module): # 64³ voxel VAE using GroupNorm (no residual blocks)
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1, bias=False), # (B, 1, 32, 32, 32) -> (B, 32, 16, 16, 16)
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # (B, 32, 16, 16, 16) -> (B, 64, 8, 8, 8)
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # (B, 64, 8, 8, 8) -> (B, 128, 4, 4, 4)
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
        )
        enc_out_dim = 128*4*4*4
        # enc_out_dim = 256*4*4*4
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4, 4))
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, enc_out_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(), # Constrain output to [0, 1]
        )
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        x = self.flatten(self.encoder_conv(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decoder
        z = self.fc_z(z)
        recon_x = self.decoder_conv(self.unflatten(z))
        return recon_x, mu, logvar

class VAETest(nn.Module):
    def __init__(self, latent_channels=vae_latent_channels):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, vae_hidden_dim, 4, 2, 1, bias=False),
            nn.GroupNorm(vae_batch_size, vae_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Conv3d(vae_hidden_dim, latent_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(vae_batch_size, latent_channels), nn.ReLU(inplace=True), nn.Dropout(0.2),
        )
        # 1×1×1 convs give channel‑wise μ and log σ², shape = (B, latent_channels, 4, 4, 4)
        self.conv_mu = nn.Conv3d(latent_channels, latent_channels, kernel_size=1, stride=1, padding=0)
        self.conv_logvar = nn.Conv3d(latent_channels, latent_channels, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, vae_hidden_dim, 4, 2, 1),
            nn.GroupNorm(vae_batch_size, vae_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.ConvTranspose3d(vae_hidden_dim, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_conv(x) # (B, 128, 4, 4, 4)
        mu, logvar = self.conv_mu(h), self.conv_logvar(h)
        z = self.reparameterize(mu, logvar) # Latent (B, 128, 4, 4, 4)
        recon = self.decoder_conv(z)
        return recon, mu, logvar

def total_variance_loss(x):
    B, C, D, H, W = x.size()
    tvl_d = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2).sum() # TV-L2
    tvl_h = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2).sum() # TV-L2
    tvl_w = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2).sum() # TV-L2
    # tvl_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).sum() # TV-L1
    norm = (B * C * (D - 1) * H * W) + (B * C * D * (H - 1) * W) + (B * C * D * H * (W - 1))
    return (tvl_d + tvl_h + tvl_w) / norm

def get_annealed_beta(epoch, warmup_epochs=20, max_beta=1e-2):
    return max_beta * min(1, epoch / warmup_epochs)

def vae_loss(recon_x, x, mu, logvar, beta_kld):
    bce = F.binary_cross_entropy(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0) 
    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() , dim=[1, 2, 3, 4]).mean() # shape: (B, C, 4, 4, 4)
    tvl = total_variance_loss(recon_x)
    lambda_tvl = vae_lambda_tvl
    # lambda_tvl = 0
    return bce + kld*beta_kld + tvl*lambda_tvl, bce, kld