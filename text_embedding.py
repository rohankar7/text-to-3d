import torch
import open_clip
import pandas as pd
import os
from config import *
from tqdm import tqdm
from shapenetcore import get_random_models

def save_embeddings():
    captions_df = pd.read_csv("./data/ShapeNetCore_Captions.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name  = "ViT-L-14"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="laion2b_s32b_b82k", # "openai"
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    os.makedirs(embedding_dir, exist_ok=True)
    for path in tqdm(sorted(get_random_models()), desc=f"Progress"):
        c, s = path.split("/")
        if f"{c}_{s}.pt" not in os.listdir(embedding_dir):
            caption = captions_df[(captions_df["Class"]==int(c)) & (captions_df["Subclass"]==s)]
            tokens  = tokenizer(caption, context_length=77).to(device)  # shape: (1, 77)
            with torch.no_grad():
                embedding = model.encode_text(tokens)           # (1, 768)
                per_token = model.token_embedding(tokens)      # (1, 77, 768)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            # print("Embedding :", embedding.shape)        # (1, 768)
            # print("Per-token :", per_token.shape)        # (1, 77, 768)
            torch.save(embedding, f"{embedding_dir}/{c}_{s}.pt") # Normalizing the embedding

def main():
    save_embeddings()

if __name__ == "__main__":
    main()