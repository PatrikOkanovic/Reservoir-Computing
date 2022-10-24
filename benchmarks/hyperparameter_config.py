import copy

from models.utils import get_hyperparam_parser

args = get_hyperparam_parser()

esn_torch_hyperparam = {
    "reservoir_size": [100, 1000, 2000, 5000],
    "sparsity": [0.01, 0.1, 0.25],  # 0.001 does not work/converge
    "radius": [0.5, 0.25, 0.1],
    "sigma_input": [1],
    "dynamics_fit_ratio": [2 / 7, 3 / 7, 4 / 7, 5 / 7],
    "regularization": [0.0],  # , 0.2, 1.0],
    "scaler_tt": ['Standard'],

    "cell_type": ['ESN_torch'],
    "solver": ['pinv'],
    "seed": [1, 2, 3]
}

# sparsity, radius not needed for RNN, LSTM, GRU
esn_lstm_hyperparam = copy.deepcopy(esn_torch_hyperparam)
esn_lstm_hyperparam['cell_type'] = ['LSTM']
esn_lstm_hyperparam.pop('sparsity', None)
esn_lstm_hyperparam.pop('radius', None)

esn_rnn_hyperparam = copy.deepcopy(esn_lstm_hyperparam)
esn_rnn_hyperparam['cell_type'] = ['RNN']

esn_gru_hyperparam = copy.deepcopy(esn_lstm_hyperparam)
esn_gru_hyperparam['cell_type'] = ['GRU']

esn_numpy_hyperparam = {
    "reservoir_size": [100, 1000, 2000, 5000],
    "sparsity": [0.01, 0.1, 0.25],  # 0.001 does not work/converge
    "radius": [0.5, 0.25, 0.1],
    "sigma_input": [1],
    "dynamics_fit_ratio": [2 / 7, 3 / 7, 4 / 7],
    "regularization": [0.0],  # , 0.2, 1.0],
    "scaler_tt": ['Standard'],

    "cell_type": ['ESN'],
    "solver": ['pinv'],
    "W_scaling": [1, 5, 10],
    "flip_sign": [True, False],
    # "seed": [1, 2, 3],
    "resample": [True, False]
}

network_outputs = [1, 4]
network_outputs = [1]
pts_per_period = args.pts_per_period
network_inputs = [5, 10, int(0.5 * pts_per_period), pts_per_period]  # can't have kernel less than 5

# LSTM already evaluated
rnn_rnn_hyperparam = {
    "input_chunk_length": network_inputs,
    "output_chunk_length": network_outputs,
    "model": ["RNN"],
    "n_rnn_layers": [2],
    "n_epochs": [200]
}
rnn_gru_hyperparam = copy.deepcopy(rnn_rnn_hyperparam)
rnn_gru_hyperparam["model"] = ["GRU"]


def get_single_config(hyperparam_dict):
    for k, v in hyperparam_dict.items():
        hyperparam_dict[k] = [v[0]]

    return hyperparam_dict


# pinv (rank 6), ESNTorch (rank 5), reservoir_size 100 can also be good,
# have to be indexed by the model class (does not work if there's an underline in there)
hyperparameter_configs = {
    'esn_ESN_torch': esn_torch_hyperparam,
    'esn_ESN': esn_numpy_hyperparam,
    'esn_LSTM': esn_lstm_hyperparam,
    'esn_GRU': esn_gru_hyperparam,
    'esn_RNN': esn_rnn_hyperparam,

    'RNNModel_RNN': rnn_rnn_hyperparam,
    'RNNModel_GRU': rnn_gru_hyperparam
}
