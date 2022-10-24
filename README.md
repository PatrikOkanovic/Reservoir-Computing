# Reservoir Computing for Chaotic Dynamical Systems

This project was done as part of the mandatory group assignment for the [Deep Learning Course at ETH Zurich](http://da.inf.ethz.ch/teaching/2021/DeepLearning/) in Fall '21. It has received a grade of 5.775/6.0, whereas a grade of 6.0 by ETH Zurich standards implies "Good enough for submission to an international conference".

For more information, please check our [report](report.pdf).

__Abstract:__ Chaotic dynamical systems continue to puzzle and amaze practitioners due to their inherent unpredictability, despite their finite and concise representations. In spite of its simplicity, Reservoir Computing ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) has been demonstrated to be well-equipped at the task of predicting the trajectories of chaotic systems where more intricate and computationally intensive Deep Learning methods have failed, but it has so far only been evaluated on a small and selected set of chaotic systems ([P.R. Vlachas et al.](https://doi.org/10.1016/j.neunet.2020.02.016)). We build and evaluate the performance of a Reservoir Computing model known as the Echo State Network ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) on a large collection of chaotic systems recently published by [W. Gilpin](https://arxiv.org/abs/2110.05266) and show that ESN does in fact beat all but the top approach out of the 16 forecasting baselines reported by the author.



## Main installation
    git clone https://github.com/PatrikOkanovic/Reservoir-Computing.git
    cd dysts
    git checkout reservoir_computing

    virtualenv rc --python=python3.7
    source rc/bin/activate
    pip install -r req.txt
    pip install -i https://test.pypi.org/simple/ EchoTorch

# Code
We have adapted the code from the https://github.com/williamgilpin/dysts repository which provides the initial implementations of the dynamical chaos systems as well as the original benchmark. 
For reproducibility reasons we have tried to keep the existing code and experiments as similar as possible and have e.g. *not* refactored existing redundant code as it could potentially  change results and introduce errors.
This additionally will allow for easier pull requests when we push our changes to the original repository. For that reason we ask for leniency regarding code quality.

# Rerunning Experiments

We provide bash scripts to reproduce our experiments and results.
After some manual hyperparameter evaluation we have identified multiple Reservoir Computing and RNN models over which we run an extensive hyperparameter search.

    bash scripts/test_find_hyperparameter_cells.sh

Note that this will only run a single hyperparameter configuration (indicated by _--test_single_config 1_) as rerunning all experiments takes significant computational resources.
Commands to execute the full experiments on Euler have been provided in the latter part of the file (indicated by _bsub_).

Other experiments (including some which were not followed further due to bad performance) can be ran by calling:

    bash scripts/default_tests.sh

These were included for completeness reasons.

To get some more info about the results, call the following:
    
    cd benchmarks/results
    python3 read_results.py

This will print out average ranks and scores for the different models over all dynamical systems and also provide an overview about which models performed best and worst.
