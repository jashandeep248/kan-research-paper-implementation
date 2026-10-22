import torch, torch.nn as nn
from .layer import KANLayer
from .symbolic import fit_symbolic, suggest_symbolic, SYMBOLIC_LIB


class KAN(nn.Module):
    """KAN with shape [n0, n1, ..., nL] (Eq. 2.3). KAN(x) = Phi_{L-1} o ... o Phi_0 (x)."""
    def __init__(self, width, G=5, k=3, grid_range=(-1, 1), seed=0, noise=0.1):
        super().__init__()
        torch.manual_seed(seed)
        self.width = width
        self.layers = nn.ModuleList(
            [KANLayer(width[i], width[i + 1], G, k, noise, grid_range) for i in range(len(width) - 1)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    @torch.no_grad()
    def update_grids(self, x):
        for l in self.layers:
            if l.update_grid:
                l.update_grid_from_samples(x)
            x = l(x)

    @torch.no_grad()
    def extend_grid(self, G_new, x):
        for l in self.layers:
            l.extend_grid_to(G_new, x)
            x = l(x)

    # ---- regularisation (Eq. 2.17-2.20) ----
    def forward_reg(self, x, lamb=1e-3, mu1=1.0, mu2=1.0):
        """Forward pass returning (output, regularisation term with gradients)."""
        reg = 0.0
        for l in self.layers:
            y = l.post_acts(x)
            l.pre, l.post = x.detach(), y.detach()
            L1 = y.abs().mean(0)                       # (n_in, n_out)
            tot = L1.sum()
            p = L1 / (tot + 1e-12)
            ent = -(p * torch.log(p + 1e-12)).sum()
            reg = reg + mu1 * tot + mu2 * ent
            x = y.sum(1)
        return x, lamb * reg

    # ---- pruning (Eq. 2.21) ----
    @torch.no_grad()
    def prune(self, x, theta=1e-2):
        self(x)
        keep = [torch.ones(self.width[0], dtype=torch.bool)]
        for li in range(len(self.layers) - 1):
            inc = self.layers[li].l1().max(0).values            # I_{l,i}
            out = self.layers[li + 1].l1().max(1).values        # O_{l,i}
            keep.append((inc > theta) & (out > theta))
        keep.append(torch.ones(self.width[-1], dtype=torch.bool))
        for li, l in enumerate(self.layers):
            l.mask *= (keep[li][:, None] & keep[li + 1][None, :]).float()
        self.active = keep
        return [int(k.sum()) for k in keep]

    def active_edges(self):
        return [int((l.mask > 0).sum()) for l in self.layers]

    # ---- symbolic regression (Sec 2.5.1 step 4) ----
    @torch.no_grad()
    def fix_symbolic(self, l, i, j, name, x):
        self(x)
        layer = self.layers[l]
        pre, post = layer.pre[:, i], layer.post[:, i, j]
        params, r2 = fit_symbolic(pre, post, SYMBOLIC_LIB[name][0])
        layer.sym[(i, j)] = (name, SYMBOLIC_LIB[name][0], params)
        return params, r2

    @torch.no_grad()
    def suggest_symbolic(self, l, i, j, x, top=3):
        self(x)
        layer = self.layers[l]
        return suggest_symbolic(layer.pre[:, i], layer.post[:, i, j], top)

    @torch.no_grad()
    def auto_symbolic(self, x, r2_min=0.9):
        """Replace every active activation by its best-fitting library function."""
        self(x)
        found = {}
        for li, l in enumerate(self.layers):
            for i in range(l.n_in):
                for j in range(l.n_out):
                    if l.mask[i, j] == 0:
                        continue
                    name, r2 = self.suggest_symbolic(li, i, j, x, 1)[0]
                    if r2 > r2_min:
                        self.fix_symbolic(li, i, j, name, x)
                        found[(li, i, j)] = (name, r2)
            self(x)
        return found

    def symbolic_formula(self, var_names=None):
        import sympy as sp
        syms = [sp.Symbol(v) for v in (var_names or [f"x{i+1}" for i in range(self.width[0])])]
        cur = syms
        for l in self.layers:
            nxt = []
            for j in range(l.n_out):
                s = 0
                for i in range(l.n_in):
                    if l.mask[i, j] == 0:
                        continue
                    if (i, j) not in l.sym:
                        raise ValueError(f"edge ({i},{j}) is not symbolic yet")
                    name, _, (a, b, c, d) = l.sym[(i, j)]
                    s += float(c) * SYMBOLIC_LIB[name][1](float(a) * cur[i] + float(b)) + float(d)
                nxt.append(s)
            cur = nxt
        return [sp.nsimplify(sp.expand(e), rational=False).evalf(4) for e in cur]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    def __init__(self, width, act=nn.SiLU, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
