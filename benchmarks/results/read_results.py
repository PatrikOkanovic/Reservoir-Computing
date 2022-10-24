import collections
import json
import traceback
import warnings
from collections import defaultdict

import numpy as np


# skip_dyn_systems: Skips the selected dynamical systems
# incomplete_model: Skips all the dynamical systems that <incomplete_model> was not evaluated on
class ResultsObject():
    def __init__(self, path='results_test_univariate__pts_per_period_100__periods_12.json', skip_dyn_systems=None,
                 incomplete_model=None):
        if skip_dyn_systems is None:
            skip_dyn_systems = ["AtmosphericRegime"]  # not evaluated with the original models

        f = open(path)
        self.results = json.load(f)
        for dyn_syst in skip_dyn_systems:
            self.results.pop(dyn_syst, None)
        if incomplete_model is not None:
            warnings.warn(f"Only calculating results for which {incomplete_model} was able to evaluate")
            for dyn_syst in list(self.results.keys()):
                if incomplete_model not in self.results[dyn_syst]:
                    self.results.pop(dyn_syst)
        self.n_dyn_systems = len(self.results)
        self.n_models = len(next(iter(self.results.values())))
        self.new_model_ranks = defaultdict(list)
        self.scores = defaultdict(list)

    def average_all_methods(self):
        models = set()
        for dyn_syst in self.scores:
            for rank, (model_name, score) in enumerate(self.scores[dyn_syst]):
                self.new_model_ranks[model_name].append((dyn_syst, rank, score))
                models.add(model_name)

        models_ranked = []
        for model in models:
            models_ranked.append((model, *self.get_average_rank(model, print_out=True)))

        models_ranked.sort(key=lambda x: x[1])
        for model, rank, mean, median in models_ranked:
            print(f'Rank {rank:2.3f} Mean {mean:3.3f} Median {median:3.3f} {model}')

        print(f"Evaluated {self.n_models} models on {self.n_dyn_systems} dynamical systems")

    def sort_results(self, metric='smape', print_out=False):
        results = self.results
        for dyn_syst in results:
            scores = sorted(
                [(model, results[dyn_syst][model][metric]) for model in results[dyn_syst] if model != 'values'],
                key=lambda x: x[1])
            self.scores[dyn_syst] = scores
            best = scores[0]
            worst = scores[len(scores) - 1]
            if print_out: print(f'SMAPE {dyn_syst:25} {best[0]:25} {best[1]:10.3f} - {worst[0]:25} {worst[1]:10.03f}')

    def update_results(self, equation_name, model_name, score):
        dyn_sys = self.scores[equation_name]
        i = 0
        while i < len(dyn_sys) and dyn_sys[i][1] < score:
            i += 1
        dyn_sys.insert(i, (model_name, score))

        print('-----', equation_name)
        if i > 0:
            best = dyn_sys[0]
            print(f'\t  0 {best[0]:25} {best[1]:10.3f}')
        if i > 1:
            print(f'\t {i - 1:2} {dyn_sys[i - 1][0]:25} {dyn_sys[i - 1][1]:10.3f}')

        print(f'\t {i:2} {model_name:25} {score:10.3f} \t <----')

        if i < len(dyn_sys) - 2:
            print(f'\t {i + 1:2} {dyn_sys[i + 1][0]:25} {dyn_sys[i + 1][1]:10.3f}')

        if i < len(dyn_sys) - 1:
            worst = dyn_sys[len(dyn_sys) - 1]
            print(f'\t {len(dyn_sys) - 1:2} {worst[0]:25} {worst[1]:10.3f}')

        self.new_model_ranks[model_name].append((equation_name, i, score))
        self.scores[equation_name] = dyn_sys
        return i

    def get_average_rank(self, model_name, print_out=True):
        n = len(self.new_model_ranks[model_name])
        rank_sum = sum([x[1] for x in self.new_model_ranks[model_name]])
        avg_rank = rank_sum / n
        scores = [x[2] for x in self.new_model_ranks[model_name]]
        avg_score = np.mean(scores)
        median_score = np.median(scores)
        if print_out:
            print(
                f'{model_name} average rank {avg_rank} out of {self.n_models} | avg {avg_score} median {median_score}')
        return avg_rank, avg_score, median_score


def combine_results_files(files=None, prefix='esn_'):
    if files is None:
        return None
    all_results = None
    failed_combinations = collections.defaultdict(list)
    for path in files:
        model_name = path.split(f"periods_12_")[1][:-5]
        print(f"Combining {model_name} {path}")
        f = open(path)
        results = json.load(f)
        if all_results is None:
            all_results = results
            continue

        for dyn_syst in results:
            try:
                all_results[dyn_syst][model_name] = results[dyn_syst][model_name]
            except KeyError as e:
                failed_combinations[dyn_syst].append(model_name)
                # traceback.print_exc()   # Not an error, we continue

    # save new file
    start_index = files[0].index(f"{prefix}") + len(prefix)
    combined_path = files[0][:start_index] + "ALL.json"
    with open(combined_path, "w") as file:
        json.dump(all_results, file)

    print(f"Stored combined results in {combined_path}")
    print('Failed combinations, skip the following dyn_syst when calculating avg rank', failed_combinations)
    return combined_path


result_files_100 = ["results_test_univariate__pts_per_period_100__periods_12_esn_ESN_torch.json",
                    "results_test_univariate__pts_per_period_100__periods_12_esn_RNN.json",
                    "results_test_univariate__pts_per_period_100__periods_12_esn_LSTM.json",
                    "results_test_univariate__pts_per_period_100__periods_12_esn_GRU.json"]

result_files_15 = ["results_test_univariate__pts_per_period_15__periods_12_esn_ESN_torch.json",
                   "results_test_univariate__pts_per_period_15__periods_12_esn_RNN.json",
                   "results_test_univariate__pts_per_period_15__periods_12_esn_LSTM.json",
                   "results_test_univariate__pts_per_period_15__periods_12_esn_GRU.json"]

result_files_100_rnn = ["results_test_univariate__pts_per_period_100__periods_12_RNNModel_GRU.json",
                        "results_test_univariate__pts_per_period_100__periods_12_RNNModel_RNN.json",
                        "results_test_univariate__pts_per_period_100__periods_12_esn_ESN_torch.json"]

result_files_15_rnn = ["results_test_univariate__pts_per_period_15__periods_12_RNNModel_GRU.json",
                       "results_test_univariate__pts_per_period_15__periods_12_RNNModel_RNN.json",
                       "results_test_univariate__pts_per_period_15__periods_12_esn_ESN_torch.json"]


def cell_comparison():
    # 15 coarse
    ## Cell comparison
    combine_results_files(result_files_15)
    results = ResultsObject(path='results_test_univariate__pts_per_period_15__periods_12_esn_ALL.json',
                            skip_dyn_systems=["PiecewiseCircuit", "AtmosphericRegime"])
    results.sort_results(print_out=True)
    results.average_all_methods()
    # ESN_torch
    results = ResultsObject(path='results_test_univariate__pts_per_period_15__periods_12_esn_ESN_torch.json')
    results.sort_results(print_out=False)
    results.average_all_methods()
    # 100 fine
    ## Cell comparison
    combine_results_files(result_files_100)
    results = ResultsObject(path='results_test_univariate__pts_per_period_100__periods_12_esn_ALL.json')
    results.sort_results(print_out=True)
    results.average_all_methods()
    # ESN_torch
    results = ResultsObject(path='results_test_univariate__pts_per_period_100__periods_12_esn_ESN_torch.json')
    results.sort_results(print_out=False)
    results.average_all_methods()


def rnn_comparison():
    # 15 coarse
    path = combine_results_files(result_files_15_rnn, prefix="RNNModel_")
    results = ResultsObject(path=path,
                            skip_dyn_systems=["PiecewiseCircuit", "AtmosphericRegime"])
    results.sort_results(print_out=True)
    results.average_all_methods()
    # 100 fine
    path = combine_results_files(result_files_100_rnn, prefix="RNNModel_")
    results = ResultsObject(path=path)
    results.sort_results(print_out=True)
    results.average_all_methods()


if __name__ == "__main__":
    rnn_comparison()
    cell_comparison()

    # Incomplete results of numpy ESN
    results = ResultsObject(path='results_test_univariate__pts_per_period_100__periods_12_esn_ESN.json',
                            incomplete_model='esn_ESN')
    results.sort_results(print_out=False)
    results.average_all_methods()
    print('Finished')

    # only best model, use this for reporting its ranks
    results = ResultsObject(path='results_test_univariate__pts_per_period_100__periods_12_esn_ESN_torch.json')
    results.sort_results(print_out=False)
    results.average_all_methods()
    print('Finished')
