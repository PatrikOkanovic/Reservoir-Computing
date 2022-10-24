import argparse
import collections
import os
import random
import traceback
import warnings

import darts
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries

from benchmarks.results.read_results import ResultsObject
from dysts.datasets import load_file

NUM_RANDOM_RESTARTS = 15


def set_seed(seed):
    seed %= 4294967294
    print(f'Using seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # multi-gpu mode
        torch.cuda.manual_seed_all(seed)
    # does not set tensorflow
    # import tensorflow as tf
    # tf.set_random_seed(seed)


def eval_simple(model):
    train_data = np.arange(1200)
    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    print('-----Evaluating on simple sequence 0 1 2 3 ...', y_train_ts.values().shape)

    try:
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
    except Exception as e:
        raise e
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

    metric_func = getattr(darts.metrics.metrics, 'mse')
    score = metric_func(true_y, pred_y)
    print('MSE on simple sequence 0 ... 1000 ', score)
    if score > 10:
        warnings.warn(f'Predicting very simple sequence, check if training/predicting is correct. '
                      f'MSE is {score}, anything above 100 is a likely error for the sequence 1000 ... 1200')


def eval_single_dyn_syst(model, dataset):
    cwd = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]
    equation_name = load_file(input_path).dataset[dataset]
    model_name = model.model_name
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)

    train_data = np.copy(np.array(equation_name["values"]))

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    try:
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
    except Exception as e:
        warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
        return np.inf
        failed_combinations[model_name].append(equation_name)
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

    print('-----', dataset, y_train_ts.values().shape)
    value = None
    for metric_name in metric_list:
        metric_func = getattr(darts.metrics.metrics, metric_name)
        score = metric_func(true_y, pred_y)
        print(metric_name, score)
        if metric_name == METRIC:
            value = score
            rank = results.update_results(dataset, model_name, score)
    return value, rank  # , model._cell.W_h


def eval_all_dyn_syst(model):
    cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = os.getcwd()
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        # 'mase', # requires scaling with train partition; difficult to report accurately
        'mse',
        # 'ope', # runs into issues with zero handling
        'r2_score',
        'rmse',
        # 'rmsle', # requires positive only time series
        'smape'
    ]
    equation_data = load_file(input_path)
    model_name = model.model_name
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)
    for equation_name in equation_data.dataset:

        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

        try:

            if model.model_name == 'RC-CHAOS-ESN':

                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))

            else:
                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))

        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            traceback.print_exc()
            continue

        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

        print('-----', equation_name, y_train_ts.values().shape)
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            print(metric_name, score)
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

        # TODO: print ranking relative to others for that dynamical system
    print('Failed combinations', failed_combinations)
    results.get_average_rank(model_name, print_out=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hyperparam_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparam_config", help="See hyperparameter_config.py", type=str)
    parser.add_argument("--test_single_config", help="", type=str2bool, default=False)
    parser.add_argument("--pts_per_period", help="", type=int, default=100)
    args = parser.parse_args()
    return args


class scaler(object):
    def __init__(self, tt):
        self.tt = tt
        self.data_min = 0
        self.data_max = 0
        self.data_mean = 0
        self.data_std = 0

    def scaleData(self, input_sequence, reuse=None):
        # data_mean = np.mean(train_input_sequence,0)
        # data_std = np.std(train_input_sequence,0)
        # train_input_sequence = (train_input_sequence-data_mean)/data_std
        if reuse == None:
            self.data_mean = np.mean(input_sequence, 0)
            self.data_std = np.std(input_sequence, 0)
            self.data_min = np.min(input_sequence, 0)
            self.data_max = np.max(input_sequence, 0)
        if self.tt == "MinMaxZeroOne":
            input_sequence = np.array((input_sequence - self.data_min) / (self.data_max - self.data_min))
        elif self.tt == "Standard" or self.tt == "standard":
            input_sequence = np.array((input_sequence - self.data_mean) / self.data_std)
        elif self.tt != "no":
            raise ValueError("Scaler not implemented.")
        return input_sequence

    def descaleData(self, input_sequence):
        if self.tt == "MinMaxZeroOne":
            input_sequence = np.array(input_sequence * (self.data_max - self.data_min) + self.data_min)
        elif self.tt == "Standard" or self.tt == "standard":
            input_sequence = np.array(input_sequence * self.data_std.T + self.data_mean)
        elif self.tt != "no":
            raise ValueError("Scaler not implemented.")
        return input_sequence


def getNewESNParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", help="cell for embedding", type=str, default='ESN_torch')
    parser.add_argument("--reservoir_size", help="reservoir_size", type=int, default=1000)
    parser.add_argument("--sparsity", help="sparsity", type=float, default=0.01)
    parser.add_argument("--radius", help="radius", type=float, default=0.5)
    parser.add_argument("--sigma_input", help="sigma_input", type=float, default=1)
    parser.add_argument("--dynamics_fit_ratio", help="dynamics_fit_ratio", type=float, default=2 / 7)
    parser.add_argument("--regularization", help="regularization", type=float, default=0.0)
    parser.add_argument("--scaler_tt", help="scaler_tt", type=str, default='Standard')
    parser.add_argument("--solver", help="solver used to learn mapping H -> Y, it can be [pinv, saga, gd]", type=str,
                        default="pinv")
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--resample", type=str2bool, default=False)

    return parser


def new_args_dict():
    parser = getNewESNParser()
    args = parser.parse_args()

    args_dict = args.__dict__
    return args_dict
