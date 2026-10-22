#!/bin/sh
# Reproduce every experiment (CPU, ~15 min). Figures/JSON land in results/.
set -e
python tests/test_kan.py
python experiments/exp1_grid_extension.py
python experiments/exp2_scaling_vs_mlp.py
python experiments/exp3_symbolic_regression.py
python experiments/exp4_continual_learning.py
python experiments/exp5_pde_poisson.py
