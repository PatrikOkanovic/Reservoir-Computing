import numpy as np
import torch
from echotorch.nn.reservoir import ESNCell as _ESNCell
from echotorch.utils import NormalMatrixGenerator, UniformMatrixGenerator


class ESNCell():
    def __init__(self, reservoir_size, radius, sparsity, sigma_input):
        self.reservoir_size = reservoir_size
        generator_h = NormalMatrixGenerator(connectivity=sparsity, spectral_radius=radius,
                                            scale=sigma_input)
        generator_in = NormalMatrixGenerator()
        self.cell = _ESNCell(1, reservoir_size, generator_h.generate((reservoir_size, reservoir_size)),
                             generator_in.generate((1, reservoir_size)), generator_in.generate((1, reservoir_size)),
                             dtype=torch.float64)
        self.cell._input_layer = lambda x: self.cell.w_in * x

    def forward(self, input):
        i = torch.tensor(np.reshape(input, (-1, 1)), dtype=torch.float64)
        self.h = self.cell.forward(i, reset_state=False)
        return self.h.T.squeeze((-1)).detach().numpy()

    def reset(self):
        self.cell.reset_hidden()
