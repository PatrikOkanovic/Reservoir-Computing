import numpy as np
import torch.nn


class GRUCell():
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.cell = torch.nn.GRUCell(1, reservoir_size)
        self.h = torch.zeros((1, self.reservoir_size))

    def forward(self, input):
        i = torch.tensor(np.reshape(input, (-1, 1)), dtype=torch.float32)
        self.h = self.cell.forward(i, self.h)
        return self.h.T.detach().numpy()

    def reset(self):
        self.h = torch.zeros((1, self.reservoir_size))
