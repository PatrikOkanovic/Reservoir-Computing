import numpy as np
import pandas as pd
from darts import TimeSeries
from sklearn.ensemble import VotingRegressor
import os
import sys

module_paths = [
    os.path.abspath(os.getcwd()),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)


from models.utils import eval_simple, eval_all_dyn_syst, new_args_dict
from models.RC_ESN import esn


class ESNEnsemble(VotingRegressor):
    def __init__(self, estimators, model_name='RC-CHAOS-ESN-Ensemble'):
        super().__init__(estimators)
        self.model_name = model_name

    def fit(self, X, y=None, sample_weight=None):
        self.training_series = X
        super().fit(X, [], sample_weight)

    def predict(self, X):
        prediction = super().predict(X)
        df = pd.DataFrame(np.squeeze(prediction))
        n = X   # X is actually the number of steps we predict
        df.index = range(len(self.training_series), len(self.training_series) + n)
        return TimeSeries.from_dataframe(df)


def main():
    model_name = 'RC-CHAOS-ESN-Ensemble'
    kwargs = new_args_dict()
    models = []
    n_models = 5
    for seed in range(n_models):
        kwargs['seed'] = seed
        kwargs['model_name'] = f'RC-CHAOS-ESN-{seed}'
        kwargs['ensemble_base_model'] = True
        # TODO try different fit_dynamics_ratio
        models.append((f'esn{seed}', esn(**kwargs)))

    ensemble = ESNEnsemble(models, model_name)

    eval_simple(ensemble)
    # eval_simple(esn(**new_args_dict()))
    eval_all_dyn_syst(ensemble)
    # eval_all_dyn_syst(esn(**new_args_dict()))


if __name__ == '__main__':
    main()
