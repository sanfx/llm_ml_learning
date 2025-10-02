import torch
import torch.nn.functional as F
import torch.nn as nn

# Suppose we have a vocab of size 5 and token index = 3
token_idx = torch.tensor([3])

# 1️⃣ One-hot encoding
one_hot = F.one_hot(token_idx, num_classes=5).float()
print("One-hot:", one_hot)

# 2️⃣ Dense embedding
embedding = nn.Embedding(num_embeddings=5, embedding_dim=4)
dense_vec = embedding(token_idx)
print("Dense embedding:", dense_vec)
