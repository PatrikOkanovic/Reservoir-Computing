#!/usr/bin/python

import sys
import os
import json

import dysts
from benchmarks.find_hyperparameters import get_model
from dysts.datasets import *

import pandas as pd
import numpy as np

import darts
from darts.models import *
from darts import TimeSeries
import darts.models


# to change
pts_per_period = 15  # 100
hyp_file_ending = '' # '_DEBUG_DEBUG_DEBUG'
results_path_ending = '' # can also be '_DEBUG'

cwd = os.path.dirname(os.path.realpath(__file__))
input_path = os.path.dirname(cwd)  + f"/dysts/data/test_univariate__pts_per_period_{pts_per_period}__periods_12.json"

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/results_" + dataname + ".json"
dataname = dataname.replace("test", "train" )
hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + f"{hyp_file_ending}.json"

metric_list = [
    'coefficient_of_variation',
    'mae',
    'mape',
    'marre',
    #'mase', # requires scaling with train partition; difficult to report accurately
    'mse',
    #'ope', # runs into issues with zero handling
    'r2_score',
    'rmse',
    #'rmsle', # requires positive only time series
    'smape'
]

equation_data = load_file(input_path)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    

for equation_name in equation_data.dataset:
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    
    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()
    
    for model_name in all_hyperparameters[equation_name].keys():
        if model_name in all_results[equation_name].keys():
            continue
        all_results[equation_name][model_name] = dict()
        
        print(equation_name + " " + model_name, flush=True)
        
        # look up season object from string
        for hyperparameter_name in all_hyperparameters[equation_name][model_name]:
            if "season" in hyperparameter_name:
                old_val = all_hyperparameters[equation_name][model_name][hyperparameter_name]
                all_hyperparameters[equation_name][model_name][hyperparameter_name] = getattr(darts.utils.utils.SeasonalityMode, old_val)

        model = get_model(model_name)

        if model_name == 'Prophet':
            df = pd.DataFrame(np.squeeze(y_train_ts.values()))
            df.index = pd.DatetimeIndex(y_train_ts.time_index)
            y_train_ts = TimeSeries.from_dataframe(df)

        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
        
        all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()
        
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            all_results[equation_name][model_name][metric_name] = metric_func(true_y, pred_y)
        
        with open(output_path[:-5] + f'{results_path_ending}.json', 'w') as f:
            json.dump(all_results, f, indent=4)   
        



