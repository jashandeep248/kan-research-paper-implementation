"""Sec 2.4 / Fig 2.3: staircase loss curves from grid extension on f=exp(sin(pi x)+y^2)."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kan import KAN
from kan.train import fit, make_data, rmse
torch.set_default_dtype(torch.float64)
f = lambda X: torch.exp(torch.sin(np.pi * X[:, 0]) + X[:, 1] ** 2)
data = make_data(f, 2, 1000, 1000)
grids = [3, 5, 10, 20, 50]; steps = 120
out = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for shape, col in [([2, 1, 1], "tab:blue"), ([2, 5, 1], "tab:red")]:
    m = KAN(shape, G=grids[0], seed=1); tr_all, te_all, te_G = [], [], []
    for G in grids:
        if G != grids[0]: m.extend_grid(G, data["Xtr"])
        tr, te = fit(m, data, steps=steps, update_grid=(G == grids[0]))
        tr_all += tr; te_all += te; te_G.append(te[-1])
        print(shape, "G=", G, "train %.2e test %.2e params %d" % (tr[-1], te[-1], m.n_params()))
    out[str(shape)] = dict(grids=grids, test=te_G, train_final=tr_all[-1])
    ax[0 if shape[1] == 5 else 1].semilogy(tr_all, label="train"); ax[0 if shape[1] == 5 else 1].semilogy(te_all, label="test")
    ax[0 if shape[1] == 5 else 1].set_title(f"KAN {shape}"); ax[0 if shape[1] == 5 else 1].set_xlabel("LBFGS step")
    ax[0 if shape[1] == 5 else 1].set_ylabel("RMSE"); ax[0 if shape[1] == 5 else 1].legend()
    for i, G in enumerate(grids): ax[0 if shape[1] == 5 else 1].axvline(i * steps, color="gray", ls=":", lw=.5)
    ax[2].loglog(grids, te_G, "o-", label=f"{shape} test RMSE", color=col)
G = np.array(grids, float); ax[2].loglog(G, 0.3 * G ** -4.0 * (G[0] ** 4) * out["[2, 1, 1]"]["test"][0] / 0.3, "k--", label="$G^{-4}$")
ax[2].set_xlabel("grid size G"); ax[2].set_ylabel("test RMSE"); ax[2].legend(); ax[2].set_title("scaling with grid size")
plt.tight_layout(); plt.savefig("results/fig_grid_extension.png", dpi=130)
json.dump(out, open("results/exp1.json", "w"), indent=1)
