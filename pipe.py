from voxel_to_latent import VoxelEncoder
from latent_to_triplane import TriplaneDecoder
from obj_to_voxel import save_voxels
import torch
import torch.nn.functional as F
from triplane_to_voxel import sample_feature_from_planes, FeatureDecoderMLP
import torch.optim as optim
from viz import viz_voxel, viz_mesh
from tqdm import tqdm
from data_loader import voxel_dataloader
import torch.optim as optim
import os

def train_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = voxel_dataloader()
    encoder = VoxelEncoder(latent_dim=1024).to(device)
    triplane_decoder = TriplaneDecoder(latent_dim=1024, out_channels=32).to(device)
    decoder_mlp = FeatureDecoderMLP(in_dim=32).to(device)
    # Assume model includes: encoder, triplane_generator, mlp_decoder
    params = list(encoder.parameters()) + list(triplane_decoder.parameters()) + list(decoder_mlp.parameters())
    optimizer = optim.Adam(params, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    os.makedirs("./checkpoints", exist_ok=True) # Checkpoint directory
    num_epochs = 100
    early_stopping_patience = num_epochs
    early_stopping_counter = 0
    # Train
    for epoch in range(num_epochs):
        encoder.train(); triplane_decoder.train(); decoder_mlp.train()
        total_loss = 0
        for voxel_gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            voxel_gt = voxel_gt.float().clamp(0.0, 1.0).to(device)
            # viz_voxel(voxel_gt.view(64, 64, 64)) # Visualize the voxel data
            latent = encoder(voxel_gt) # Encode to latent [B, latent_dim=1024]
            triplanes = triplane_decoder(latent) # Decode to triplanes [B, 3, 32, 128, 128]
            B = voxel_gt.shape[0]
            N = 16 ** 3 # Sample 3D coordinates
            coords = torch.rand(B, N, 3, device=device) * 2 - 1  # in [-1, 1]
            gt_values = F.grid_sample(voxel_gt, coords.view(B, 1, N, 1, 3), align_corners=True).squeeze(-1) # Sample GT voxel values at coords
            gt_values = gt_values.clamp(0.0, 1.0)
            # Sample features from triplanes
            features = sample_feature_from_planes(triplanes, coords).permute(0, 2, 1) # [B, N, C]
            pred = decoder_mlp(features).squeeze(-1) # Decode occupancy values [B, N]
            # print(pred.min(), pred.max())
            loss = F.binary_cross_entropy(pred, gt_values.squeeze()) # Loss
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"🔥 Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f}")
        # Test
        best_test_loss = float("inf")
        encoder.eval(); triplane_decoder.eval(); decoder_mlp.eval()
        with torch.no_grad():
            test_loss = 0
            for voxel_gt in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                voxel_gt = voxel_gt.float().clamp(0.0, 1.0).to(device)
                # viz_voxel(voxel_gt.view(64, 64, 64)) # Visualize the voxel data
                latent = encoder(voxel_gt) # Encode to latent [B, latent_dim=1024]
                triplanes = triplane_decoder(latent) # Decode to triplanes [B, 3, 32, 128, 128]
                B = voxel_gt.shape[0]
                N = 16 ** 3 # Sample 3D coordinates
                coords = torch.rand(B, N, 3, device=device) * 2 - 1  # in [-1, 1]
                gt_values = F.grid_sample(voxel_gt, coords.view(B, 1, N, 1, 3), align_corners=True).squeeze(-1) # Sample GT voxel values at coords
                gt_values = gt_values.clamp(0.0, 1.0)
                features = sample_feature_from_planes(triplanes, coords).permute(0, 2, 1) # [B, N, C]
                pred = decoder_mlp(features).squeeze(-1) # Decode occupancy values [B, N]      
                loss = F.binary_cross_entropy(pred, gt_values.squeeze()) # Loss
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            scheduler.step(avg_test_loss)
            print(f"🧪 Test Loss = {avg_test_loss:.6f}")
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({ # Save best model
                'epoch': epoch,
                "encoder": encoder.state_dict(),
                "triplane_decoder": triplane_decoder.state_dict(),
                "decoder_mlp": decoder_mlp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_loss": best_test_loss
                }, f"checkpoints/best_model.pth")
                print(f"Saved best model at {epoch+1}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > early_stopping_patience:
                    print("Early stopping triggered")
                    break
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))
        torch.cuda.empty_cache()

def main():
    print('Main function: VAE')
    # save_voxels()
    train_test()

if __name__ == '__main__':
    main()