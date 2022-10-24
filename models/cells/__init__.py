from models.cells.esn_cell import ESNCell as ESNCell_numpy
from models.cells.esn_cell_torch import ESNCell as ESNCell_torch
from models.cells.gru_cell import GRUCell
from models.cells.lstm_cell import LSTMCell
from models.cells.rnn_cell import RNNCell


def get_cell(type, reservoir_size, radius, sparsity, sigma_input, W_scaling=1, flip_sign=False, resample=True):
    if type == 'GRU':
        return GRUCell(reservoir_size)
    elif type == 'LSTM':
        return LSTMCell(reservoir_size)
    elif type == 'RNN':
        return RNNCell(reservoir_size)
    elif type == 'ESN':
        return ESNCell_numpy(reservoir_size, radius, sparsity, sigma_input, W_scaling, flip_sign, resample)
    elif type == 'ESN_torch':
        return ESNCell_torch(reservoir_size, radius, sparsity, sigma_input)
    else:
        raise RuntimeError('Unknown cell type.')
