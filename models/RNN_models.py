# !/usr/bin/env python
import argparse

import darts.models
import os
import sys

module_paths = [
    os.path.abspath(os.getcwd()),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

from models.utils import  eval_all_dyn_syst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="cell for embedding, can be: RNN, LSTM or GRU", type=str, default='RNN',)
    args = parser.parse_args()

    model = darts.models.RNNModel()
    model.model_name = args.model_name
    eval_all_dyn_syst(model)


if __name__ == '__main__':
    main()
