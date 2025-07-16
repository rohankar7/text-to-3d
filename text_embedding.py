import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load a frozen text encoder (ViT‑L/14 = SD default)
model_name  = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained="openai", device=device
)
tokenizer = open_clip.get_tokenizer(model_name)

# 2) Caption → 77‑token tensor
caption = "a white airplane with blue and yellow accents and two engines on each wing"
tokens  = tokenizer(caption, context_length=77).to(device)  # shape: (1, 77)

# 3) Get embedding (no grad)
with torch.no_grad():
    text_emb = model.encode_text(tokens)           # (1, 768) global
    per_token = model.token_embedding(tokens)      # (1, 77, 768) if you want full ctx

# 4) Normalise the CLS embedding (optional)
text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

print("global CLS vector :", text_emb.shape)        # (1, 768)
print("per‑token ctx    :", per_token.shape)        # (1, 77, 768)