directory = "C:/ShapeNetCore"
suffix_dir = "models/model_normalized.obj"
random_seed = 36
sample_size = 50

# Best VAE hyperparameters
voxel_dir = "./voxels_128"
voxel_res = 128
vae_batch_size = 4
vae_num_epochs = 1000
vae_stopping_patience = 50
vae_latent_channels = 256
vae_hidden_dim = 32
vae_beta_kld = 1e-2
vae_optim_lr = 1e-3
vae_lambda_tvl = 1e-2
vae_checkpoint_dir = "./checkpoints/vae"
latent_dir = "./latents"
# 🔥 Epoch 455: Avg Train Loss = 0.000748 | Recon: 0.000530 | KLD: 0.019817
# Epoch 455/1000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.99it/s] 
# 🧪 Test Loss = 0.000594
# Saved best model at 455
# 🔥 Epoch 433: Avg Train Loss = 0.000685 | Recon: 0.000474 | KLD: 0.018973
# Epoch 433/1000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.12it/s] 
# 🧪 Test Loss = 0.000465
# Saved best model at 433
# Latent Res = (B, 256, 32, 32, 32)
embedding_dir = "./embeddings"