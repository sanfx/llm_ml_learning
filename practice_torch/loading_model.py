import torch
import torch.nn as nn

from example1 import chars, char2idx, idx2char

# Base model class
class TinyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=16, rnn_type="LSTM"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)


# Recreate the model with the same architecture
# model = TinyLM(len(chars), rnn_type="LSTM")  # or "GRU" # if the model was not saved in LSTM it will error
# Load the saved weights
# model.load_state_dict(torch.load("tiny_lm.pth"))
# model.eval()

# Load the vanilla RNN and transfer embeds.
rnn_state = torch.load("tiny_lm.pth")
new_model = TinyLM(len(chars), rnn_type="LSTM")
new_model.embed.load_state_dict({'weight': rnn_state['embed.weight']})

# Set to evaluation mode (important for dropout/batchnorm layers)
new_model.eval()

# Example: Predict next character after 'l'
test_input = torch.tensor([[char2idx["l"], char2idx["l"]]])  # Shape: (1, 1)


if __name__ == "__main__":
    # Forward pass
    with torch.no_grad():
        output = new_model(test_input)  # Shape: (1, 1, vocab_size)
        pred = new_model(test_input).argmax(dim=2)

    # Decode prediction
    print("Predicted next char:", idx2char[pred[0, -1].item()])

    # the answer is o after ll