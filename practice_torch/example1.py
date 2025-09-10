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

# Model: small RNN (Reoccuring Neural Network)
class TinyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)

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
    output = model(inputs)
    loss = criterion(output.view(-1, len(chars)), targets.view(-1))
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # Test prediction
    test_input = torch.tensor([[char2idx["l"], char2idx["l"]]])
    pred = model(test_input).argmax(dim=2)
    # print("Predicted next char:", idx2char[pred.item()])
    print("Predicted next char:", idx2char[pred[0, -1].item()])
