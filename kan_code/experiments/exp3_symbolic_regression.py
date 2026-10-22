"""Sec 2.5.2 / Fig 2.4: Alice's workflow: train with sparsity -> prune -> symbolify -> affine fine-tune -> formula."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kan import KAN
from kan.train import fit, make_data, rmse
torch.set_default_dtype(torch.float64)
f = lambda X: torch.exp(torch.sin(np.pi*X[:,0]) + X[:,1]**2)
d = make_data(f, 2, 2000, 1000)
m = KAN([2, 5, 1], G=5, seed=2)
fit(m, d, steps=60, lamb=3e-3)                       # step 1: sparsification
print("step1 test", rmse(m, d["Xte"], d["yte"]))
print("step2 active nodes:", m.prune(d["Xtr"], 1e-2), "edges:", m.active_edges())   # step 2: prune
log = {"after_prune_nodes": [int(k.sum()) for k in m.active]}
fit(m, d, steps=40, update_grid=False)               # retrain pruned net
print("pruned test", rmse(m, d["Xte"], d["yte"]))
for l in range(2):                                   # step 3: suggest symbolic
    for i in range(m.layers[l].n_in):
        for j in range(m.layers[l].n_out):
            if m.layers[l].mask[i, j] > 0:
                print(" edge", (l, i, j), m.suggest_symbolic(l, i, j, d["Xtr"]))
# Step 3 (human in the loop): auto suggestion for the outer edge is ambiguous (tanh~exp on a narrow range),
# so "Alice" overrides it with her prior knowledge, as in the paper (fix_symbolic(1,0,0,'exp')).
print("fix sin  :", m.fix_symbolic(0, 0, 1, "sin", d["Xtr"]))
print("fix x^2  :", m.fix_symbolic(0, 1, 1, "x^2", d["Xtr"]))
print("fix exp  :", m.fix_symbolic(1, 1, 0, "exp", d["Xtr"]))
log = {"auto_suggestions_outer_edge": [(n, round(r, 6)) for n, r in m.suggest_symbolic(1, 1, 0, d["Xtr"], 3)]}
rm = rmse(m, d["Xte"], d["yte"]); print("test RMSE after symbolification:", rm); log["rmse_symbolic"] = rm
# Step 4 (affine fine-tuning) is already done inside fix_symbolic: each edge's (a,b,c,d) is fitted
# by grid search + linear regression on the observed pre/post-activations.
import sympy as sp
f_sym = m.symbolic_formula(["x", "y"])[0]
f_sym = f_sym.xreplace({n: round(float(n), 3) for n in f_sym.atoms(sp.Number)})
print("formula:", f_sym); log["formula"] = str(f_sym)
Xs = d["Xte"]; fn = sp.lambdify(sp.symbols("x y"), f_sym, "numpy")
pred = fn(Xs[:, 0].numpy(), Xs[:, 1].numpy()); rm2 = float(np.sqrt(np.mean((pred - d["yte"].squeeze().numpy()) ** 2)))
print("test RMSE of extracted closed-form:", rm2); log["rmse_closed_form"] = rm2
json.dump(log, open("results/exp3.json", "w"), indent=1)
