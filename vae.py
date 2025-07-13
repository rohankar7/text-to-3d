import torch
from torch import nn
import torch.nn.functional as F
import config


class VAE(nn.Module): # 64³ voxel VAE using GroupNorm (no residual blocks)
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.GroupNorm(num_groups=8, num_channels=1024),
            # nn.ReLU(inplace=True),
        )
        enc_out_dim = 4096
        # enc_out_dim = 256*4*4*4
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 2, 2, 2))
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, enc_out_dim)
        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1),
            # nn.GroupNorm(num_groups=8, num_channels=512),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
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

def total_variance_loss(x):
    B, C, D, H, W = x.size()
    tvl_d = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2).sum()
    tvl_h = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2).sum()
    tvl_w = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2).sum()
    return (tvl_d + tvl_h + tvl_w) / (B * C * D * H * W)


def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tvl = total_variance_loss(recon_x)
    beta_kld = 1e-3
    lambda_tvl = 1e-3
    return bce + kld*beta_kld + tvl*lambda_tvl