# Compute the internal gates
import numpy as np
import torch.nn


class LSTMCell():
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.cell = torch.nn.LSTMCell(1, reservoir_size)
        self.h = torch.zeros((1, self.reservoir_size))
        self.c = torch.zeros((1, self.reservoir_size))

    def forward(self, input):
        i = torch.tensor(np.reshape(input, (-1, 1)), dtype=torch.float32)
        self.h, self.c = self.cell.forward(i, (self.h, self.c))
        return self.h.T.detach().numpy()

    def reset(self):
        self.h = torch.zeros((1, self.reservoir_size))
        self.c = torch.zeros((1, self.reservoir_size))
