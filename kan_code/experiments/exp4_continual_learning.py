"""Sec 3.5 / Fig 3.4: continual learning. 5 Gaussian peaks presented sequentially to KAN and MLP."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kan import KAN, MLP
torch.set_default_dtype(torch.float64)
centers = np.linspace(-0.8, 0.8, 5)
target = lambda x: sum(torch.exp(-((x - c) ** 2) / (2 * 0.06 ** 2)) for c in centers)
xs = torch.linspace(-1, 1, 400).unsqueeze(1)
def phase_data(p):
    x = torch.rand(300, 1) * 0.32 + (centers[p] - 0.16)
    return x, target(x)
def train(model, x, y, steps=800):
    o = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        o.zero_grad(); loss = ((model(x) - y) ** 2).mean(); loss.backward(); o.step()
res = {}
fig, ax = plt.subplots(2, 5, figsize=(16, 5), sharey=True)
def kan_local():
    m = KAN([1, 1], G=50, k=3, seed=0)
    m.layers[0].w_b.data.zero_(); m.layers[0].w_b.requires_grad_(False)   # remove global silu residual
    m.layers[0].w_s.requires_grad_(False)   # w_s is a *global* scale shared by all coefs -> freeze (keep only local B-spline coefs trainable)
    m.layers[0].update_grid = False
    return m
for r, (name, mk) in enumerate([("KAN", kan_local), ("MLP", lambda: MLP([1, 64, 64, 1], seed=0))]):
    m = mk(); errs = []
    for p in range(5):
        x, y = phase_data(p); train(m, x, y)
        with torch.no_grad(): pred = m(xs).squeeze()
        # error on peaks learned so far
        errs_p = []
        for q in range(p + 1):
            mask = (xs.squeeze() > centers[q] - 0.16) & (xs.squeeze() < centers[q] + 0.16)
            errs_p.append(((pred[mask] - target(xs)[:, 0][mask]) ** 2).mean().sqrt().item())
        errs.append(errs_p)
        ax[r, p].plot(xs, target(xs), c="lightgray"); ax[r, p].plot(xs, pred.numpy(), c="k")
        ax[r, p].set_title(f"{name}, phase {p+1}"); ax[r, p].set_ylim(-0.5, 1.5)
    res[name] = errs
    print(name, "RMSE on old peaks after phase 5:", [f"{e:.3f}" for e in errs[-1]])
plt.tight_layout(); plt.savefig("results/fig_continual.png", dpi=120)
json.dump(res, open("results/exp4.json", "w"), indent=1)
