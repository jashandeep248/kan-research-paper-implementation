"""Sec 3.1 / Fig 3.1: KAN vs MLP scaling (test RMSE vs #params) on toy functions with known KA form."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kan import KAN, MLP
from kan.train import fit, make_data
torch.set_default_dtype(torch.float64)
TASKS = {
 "exp(sin(pi x)+y^2)": (lambda X: torch.exp(torch.sin(np.pi*X[:,0])+X[:,1]**2), [2,1,1]),
 "x*y":                (lambda X: X[:,0]*X[:,1],                               [2,2,1]),
}
grids = [3, 5, 10, 20]; steps = 120
res = {}
fig, axs = plt.subplots(1, len(TASKS), figsize=(5.5*len(TASKS), 4))
for ax, (name, (f, shape)) in zip(axs, TASKS.items()):
    d = make_data(f, 2, 1000, 1000); res[name] = {}
    m = KAN(shape, G=grids[0], seed=0); P, E = [], []
    for G in grids:
        if G != grids[0]: m.extend_grid(G, d["Xtr"])
        tr, te = fit(m, d, steps=steps, update_grid=(G == grids[0])); P.append(m.n_params()); E.append(te[-1])
    res[name]["KAN"] = (P, E); ax.loglog(P, E, "o-", c="tab:blue", label=f"KAN {shape}")
    for depth, c in [(2, "tab:orange"), (3, "tab:green")]:
        P, E = [], []
        for w in [5, 10, 20, 40]:
            mlp = MLP([2] + [w]*(depth-1) + [1], seed=0); tr, te = fit(mlp, d, steps=300)
            P.append(mlp.n_params()); E.append(te[-1])
        res[name][f"MLP d{depth}"] = (P, E); ax.loglog(P, E, "s--", c=c, label=f"MLP depth {depth}")
    ax.set_title(name); ax.set_xlabel("# parameters"); ax.set_ylabel("test RMSE"); ax.legend()
    print(name, {k: [f"{e:.1e}" for e in v[1]] for k, v in res[name].items()})
plt.tight_layout(); plt.savefig("results/fig_scaling_vs_mlp.png", dpi=130)
json.dump(res, open("results/exp2.json", "w"), indent=1)
