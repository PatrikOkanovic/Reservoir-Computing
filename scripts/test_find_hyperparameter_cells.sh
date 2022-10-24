#!/bin/bash
python benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --test_single_config 1

python benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --test_single_config 1

exit 0;
# can likely also be run for shorter
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --pts_per_period 15

bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --pts_per_period 100


bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --pts_per_period 15

bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --pts_per_period 100