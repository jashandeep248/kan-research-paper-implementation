# KAN: Kolmogorov–Arnold Networks — from-scratch PyTorch re-implementation

Re-implementation of Liu et al., *KAN: Kolmogorov–Arnold Networks* (ICLR 2025, arXiv:2404.19756).

```
kan/layer.py      KANLayer: B-spline basis (Cox–de Boor), w_b*silu + w_s*spline, grid update, grid extension (least squares)
kan/model.py      KAN (shape [n0..nL]), L1+entropy regularisation, pruning, fix/suggest/auto symbolic, symbolic_formula; MLP baseline
kan/symbolic.py   symbolic library and affine fit y ≈ c·f(ax+b)+d (grid search + linear regression)
kan/train.py      LBFGS/Adam trainer, data helper
experiments/      exp1 grid extension · exp2 scaling vs MLP · exp3 symbolic regression · exp4 continual learning · exp5 Poisson PINN
tests/            unit tests (partition of unity, parameter count, grid-extension fidelity, backprop)
```
Run `pip install -r requirements.txt && ./run_all.sh`. Results/figures are written to `results/`.

Not reproduced (data/simulators not available here): DeepMind knot-theory dataset, Anderson-localization models,
the full Feynman/special-function sweeps, LAN/SIREN image fitting.
