"""Symbolic library and affine fit  y ~ c*f(a x + b) + d  (paper Sec 2.5.1, step 4):
iterative grid search over (a, b) with linear regression for (c, d)."""
import torch, sympy as sp

SYMBOLIC_LIB = {
    "x":    (lambda x: x,                 lambda x: x),
    "x^2":  (lambda x: x ** 2,            lambda x: x ** 2),
    "x^3":  (lambda x: x ** 3,            lambda x: x ** 3),
    "sin":  (torch.sin,                   sp.sin),
    "cos":  (torch.cos,                   sp.cos),
    "exp":  (torch.exp,                   sp.exp),
    "log":  (lambda x: torch.log(x.clamp_min(1e-6)), sp.log),
    "tanh": (torch.tanh,                  sp.tanh),
    "sqrt": (lambda x: torch.sqrt(x.clamp_min(0)), sp.sqrt),
    "abs":  (torch.abs,                   sp.Abs),
}


def _lin_fit(fx, y):
    A = torch.stack([fx, torch.ones_like(fx)], 1)
    sol = torch.linalg.lstsq(A, y[:, None]).solution.squeeze(1)
    pred = A @ sol
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return sol[0], sol[1], 1 - ss_res / ss_tot


def fit_symbolic(x, y, f, a_range=(-10, 10), b_range=(-10, 10), n=101, iters=3):
    """Returns ((a,b,c,d), R^2)."""
    best = (-1e9, None)
    a_lo, a_hi, b_lo, b_hi = *a_range, *b_range
    for _ in range(iters):
        a_grid = torch.linspace(a_lo, a_hi, n)
        b_grid = torch.linspace(b_lo, b_hi, n)
        for a in a_grid:
            for b in b_grid:
                fx = f(a * x + b)
                if not torch.isfinite(fx).all() or fx.std() < 1e-8:
                    continue
                c, d, r2 = _lin_fit(fx, y)
                if r2 > best[0]:
                    best = (r2.item(), (a.item(), b.item(), c.item(), d.item()))
        if best[1] is None:
            return (1.0, 0.0, 0.0, y.mean().item()), 0.0
        a0, b0 = best[1][:2]
        da, db = (a_hi - a_lo) / (n - 1) * 2, (b_hi - b_lo) / (n - 1) * 2
        a_lo, a_hi, b_lo, b_hi = a0 - da, a0 + da, b0 - db, b0 + db
        n = 21
    return best[1], best[0]


def suggest_symbolic(x, y, top=3, coarse_n=41):
    res = []
    for name, (f, _) in SYMBOLIC_LIB.items():
        _, r2 = fit_symbolic(x, y, f, n=coarse_n, iters=1)
        res.append((name, r2))
    return sorted(res, key=lambda t: -t[1])[:top]
