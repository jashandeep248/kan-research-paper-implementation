"""KAN layer: a matrix of learnable 1-D functions phi_{j,i}(x) = w_b*silu(x) + w_s*spline(x)
(paper Eq. 2.5, 2.10-2.12). Splines are B-splines of order k on a grid of G intervals."""
import torch, torch.nn as nn


def extend_grid(grid, k):
    """Augment (n, G+1) grid with k extra knots on each side (uniform spacing)."""
    h = (grid[:, -1:] - grid[:, :1]) / (grid.shape[1] - 1)
    for _ in range(k):
        grid = torch.cat([grid[:, :1] - h, grid, grid[:, -1:] + h], dim=1)
    return grid


def bspline_basis(x, grid, k):
    """Cox-de Boor recursion. x: (B, n), grid: (n, G+2k+1) -> (B, n, G+k)."""
    x = x.unsqueeze(-1)
    b = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for p in range(1, k + 1):
        b = (x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)]) * b[..., :-1] \
            + (grid[:, p + 1:] - x) / (grid[:, p + 1:] - grid[:, 1:-p]) * b[..., 1:]
    return b


def curve2coef(x, y, grid, k, eps=1e-8):
    """Least-squares B-spline coefficients so that spline(x) ~ y.
    x:(B,n) y:(B,n,m) -> coef (n,m,G+k). Used for grid extension (Eq. 2.16)."""
    B = bspline_basis(x, grid, k).permute(1, 0, 2)          # (n, B, K)
    y = y.permute(1, 0, 2)                                   # (n, B, m)
    A = B.transpose(1, 2) @ B + eps * torch.eye(B.shape[2], device=x.device)
    rhs = B.transpose(1, 2) @ y
    return torch.linalg.solve(A, rhs).permute(0, 2, 1)       # (n, m, K)


class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, G=5, k=3, noise=0.1, grid_range=(-1, 1),
                 base_fn=torch.nn.SiLU(), update_grid=True):
        super().__init__()
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, G, k
        self.base_fn, self.update_grid = base_fn, update_grid
        g = torch.linspace(grid_range[0], grid_range[1], G + 1).repeat(n_in, 1)
        self.register_buffer("grid", extend_grid(g, k))
        # spline(x) ~ 0 at init: small noise coefficients (paper footnote 2)
        self.coef = nn.Parameter(torch.randn(n_in, n_out, G + k) * noise / G)
        # w_b Xavier init, w_s = 1
        self.w_b = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.w_b)
        self.w_s = nn.Parameter(torch.ones(n_in, n_out))
        self.register_buffer("mask", torch.ones(n_in, n_out))   # for pruning / symbolic
        self.sym = {}                                             # (i,j) -> (name, fn, (a,b,c,d))

    def splines(self, x):
        B = bspline_basis(x, self.grid, self.k)                  # (B, n_in, K)
        return torch.einsum("bik,iok->bio", B, self.coef)        # (B, n_in, n_out)

    def post_acts(self, x):
        """phi_{o,i}(x_i) for every edge: (B, n_in, n_out)."""
        base = self.base_fn(x).unsqueeze(-1) * self.w_b
        out = base + self.w_s * self.splines(x)
        for (i, j), (_, f, (a, b, c, d)) in self.sym.items():  # symbolic edges
            out = out.clone()
            out[:, i, j] = c * f(a * x[:, i] + b) + d
        return out * self.mask

    def forward(self, x):
        self.pre, self.post = x.detach(), None
        y = self.post_acts(x)
        self.post = y.detach()
        return y.sum(1)                                           # nodes just sum

    @torch.no_grad()
    def update_grid_from_samples(self, x, margin=0.01):
        """Adapt the grid to the input range (paper impl. detail 3) and refit coefs."""
        xs, _ = torch.sort(x, dim=0)
        y = self.splines(x)
        n = x.shape[0]
        idx = torch.linspace(0, n - 1, self.G + 1).long()
        adapt = xs[idx].T                                         # quantile grid (n_in, G+1)
        lo, hi = xs[0] - margin, xs[-1] + margin
        uni = lo[:, None] + (hi - lo)[:, None] * torch.linspace(0, 1, self.G + 1)[None]
        g = 0.5 * adapt + 0.5 * uni                               # blend adaptive & uniform
        g[:, 0], g[:, -1] = lo, hi
        self.grid.copy_(extend_grid(g, self.k))
        self.coef.data = curve2coef(x, y, self.grid, self.k)

    @torch.no_grad()
    def extend_grid_to(self, G_new, x):
        """Grid extension (Sec 2.4): fit finer spline to the old coarse one by least squares."""
        y = self.splines(x)
        lo, hi = self.grid[:, self.k], self.grid[:, -self.k - 1]
        g = lo[:, None] + (hi - lo)[:, None] * torch.linspace(0, 1, G_new + 1)[None]
        self.G = G_new
        self.grid = extend_grid(g, self.k)
        self.coef = nn.Parameter(curve2coef(x, y, self.grid, self.k))

    def l1(self):
        """Mean |phi| per edge over the batch (Eq. 2.17): (n_in, n_out)."""
        return self.post.abs().mean(0)
