import spacy
import torch

# 1. Load spacy model
nlp = spacy.load("en_core_web_sm")

# 2. Tokenize text at word level
text = "Apple is looking at buying the U.K. startup for $1 billion.."
doc = nlp(text)
tokens = [token.text for token in doc]   # word-level tokens
print("Tokens:", tokens)

# 3. Build vocabulary
token2idx = {token: idx for idx, token in enumerate(set(tokens))}
idx2token = {idx: token for token, idx in token2idx.items()}
vocab_size = len(token2idx)
print("Vocab size:", vocab_size)

# 4. Convert to tensor (sequence of indices)
data = torch.tensor([token2idx[token] for token in tokens])
inputs = data[:-1].unsqueeze(0)   # batch=1
targets = data[1:].unsqueeze(0)
print("Inputs:", inputs)
print("Targets:", targets)