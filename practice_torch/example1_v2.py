import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# Sample dataset
text = "hello world"
# chars = list(set(text))
chars = list(OrderedDict.fromkeys(text))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

# Encode text into integers
data = [char2idx[ch] for ch in text]

# Model: small RNN (Recurrent Neural Network)
class TinyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=16):
        super().__init__()
        self.set_blueprint(vocab_size, hidden_size)

    def set_blueprint(self, vocab_size, hidden_size):
        # Convert tokens (like word indices or character IDs) → dense vectors.
        self.embed = nn.Embedding(vocab_size, hidden_size) # preprocessing step (map IDs to vectors).
        # recurrent core where the feedback loop happens internally in the forward method.
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # maps hidden state back to logits for prediction.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """PyTorch models, the forward method ends with raw logits."""
        # 1. Embedding: convert tokens → dense vectors
        x = self.embed(x)
        # 2. RNN: process embeddings, maintain hidden state
        out, hidden = self.rnn(x, hidden)    # 🔄 feedback loop inside here, [batch, hidden_size]
        # 3. Linear: map hidden states to logits for vocab prediction.
        logits = self.fc(out)   # [batch, vocab_size]
        # 4. Return BOTH logits (for prediction) + hidden (for feedback loop)
        return logits, hidden

# Training setup
model = TinyLM(len(chars))
torch.save(model.state_dict(), "tiny_lm.pth")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
inputs = torch.tensor([data[:-1]])
targets = torch.tensor([data[1:]])

for epoch in range(200):
    optimizer.zero_grad()
    logits, hidden = model(inputs)
    # loss needs only logits (shape: [batch, seq_len, vocab_size])
    loss = criterion(logits.view(-1, len(chars)), targets.view(-1))
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    model.eval()
    # Test prediction
    test_input = torch.tensor([[char2idx["h"], char2idx["e"]]])
    logits, hidden = model(test_input)
    pred = logits.argmax(dim=2)
    print("Predicted next char:", idx2char[pred[0, -1].item()])
    
