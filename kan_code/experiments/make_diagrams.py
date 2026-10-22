"""Schematic figures for the report/slides: MLP vs KAN, and B-spline basis."""
import sys; sys.path.insert(0, ".")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from kan.layer import bspline_basis, extend_grid
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
def draw(a, title, edge_fn):
    xs = [0, 1, 2]; a.set_xlim(-.4, 2.4); a.set_ylim(-.6, 1.4); a.axis("off"); a.set_title(title, fontsize=13)
    layers = [[0.0, 1.0], [-.2, .35, .9, 1.2], [0.5]]
    for l in range(2):
        for i, yi in enumerate(layers[l]):
            for j, yj in enumerate(layers[l + 1]):
                a.plot([l, l + 1], [yi, yj], c="#bbbbbb", lw=1, zorder=1); edge_fn(a, l, yi, l + 1, yj, i + j)
    for l, ys in enumerate(layers): a.scatter([l] * len(ys), ys, s=180, c="#1f4e79", zorder=3)
t = np.linspace(-1, 1, 30)
def mlp_edge(a, x0, y0, x1, y1, s): a.text((x0 + x1) / 2, (y0 + y1) / 2, "w", fontsize=8, ha="center", va="center", bbox=dict(fc="w", ec="none", pad=.5), zorder=2)
def kan_edge(a, x0, y0, x1, y1, s):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; ia = a.inset_axes([cx - .07, cy - .07, .14 * .6, .14 * .6], transform=a.transData)
    ia.plot(t, np.sin(t * (2 + s % 3)) if s % 2 else t ** 2 * (1 + s % 2), c="#c0392b", lw=1); ia.set_xticks([]); ia.set_yticks([])
draw(ax[0], "MLP: fixed σ on nodes, linear weights w on edges", mlp_edge)
draw(ax[1], "KAN: learnable φ(x) on edges, nodes only sum", kan_edge)
g = extend_grid(torch.linspace(-1, 1, 6).unsqueeze(0), 3); x = torch.linspace(-.999, .999, 400).unsqueeze(1)
B = bspline_basis(x, g, 3)[:, 0].numpy(); rng = np.random.RandomState(0); c = rng.randn(B.shape[1])
for k in range(B.shape[1]): ax[2].plot(x, B[:, k], lw=1, alpha=.6)
ax[2].plot(x, B @ c, "k", lw=2.5, label="φ(x)=Σ cᵢBᵢ(x)"); ax[2].legend(); ax[2].set_title("Cubic B-spline basis (G=5, k=3) → one KAN edge", fontsize=13)
plt.tight_layout(); plt.savefig("results/fig_architecture.png", dpi=130)
