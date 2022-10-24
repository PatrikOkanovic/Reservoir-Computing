import json
import os

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries

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

METRIC = 'smape'

def eval_test(model, model_name, equation_data, equation_name, all_results, output_path, results_path_ending):
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))


    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()

    all_results[equation_name][model_name] = dict()

    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()

    model.fit(y_train_ts)
    y_val_pred = model.predict(len(y_val))
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
    all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()
    for metric_name in metric_list:
        metric_func = getattr(darts.metrics.metrics, metric_name)
        score = metric_func(true_y, pred_y)
        all_results[equation_name][model_name][metric_name] = score
        if metric_name == METRIC:
            print(metric_name, score)

    with open(output_path[:-5] + f'{results_path_ending}.json', 'w') as f:
        json.dump(all_results, f, indent=4)

