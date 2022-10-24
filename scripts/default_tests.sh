#!/bin/bash

# Default RC_ESN with resampling
python models/RC_ESN.py --cell_type ESN # --resample True

# Ensemble
python models/RC_ESN_ensemble.py

# Backprop ESN
python models/RC_ESN_torch.py
