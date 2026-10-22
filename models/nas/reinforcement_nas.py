"""RL-based architecture search — policy gradient (Equation 3 adapted for NAS)."""
import torch, torch.nn as nn

class NASController(nn.Module):
    """LSTM controller generating architecture tokens."""
    def __init__(self, num_choices=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTMCell(num_choices, hidden)
        self.decoder = nn.Linear(hidden, num_choices)

    def forward(self, x, h=None, c=None):
        h, c = self.lstm(x, (h, c) if h is not None else None)
        return torch.softmax(self.decoder(h), dim=-1), h, c
