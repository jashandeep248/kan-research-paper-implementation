"""Sec 3.4 / Fig 3.3: Poisson u_xx+u_yy=f on [-1,1]^2, u=sin(pi x)sin(pi y^2), PINN loss with KAN vs MLP."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
from kan import KAN, MLP
torch.set_default_dtype(torch.float64)
pi = np.pi
u_true = lambda z: torch.sin(pi*z[:,0])*torch.sin(pi*z[:,1]**2)
f_rhs = lambda z: -pi**2*(1+4*z[:,1]**2)*torch.sin(pi*z[:,0])*torch.sin(pi*z[:,1]**2) + 2*pi*torch.sin(pi*z[:,0])*torch.cos(pi*z[:,1]**2)
n = 30; g = torch.linspace(-1, 1, n); Zi = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
t = torch.linspace(-1, 1, 50)
Zb = torch.cat([torch.stack([t, -torch.ones(50)], 1), torch.stack([t, torch.ones(50)], 1),
                torch.stack([-torch.ones(50), t], 1), torch.stack([torch.ones(50), t], 1)])
Zt = torch.rand(4000, 2)*2 - 1
def run(model, steps, alpha=0.01):
    if hasattr(model, "layers"):
        model.update_grids(Zi)
    opt = torch.optim.LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    def lap(z):
        z = z.clone().requires_grad_(True); u = model(z).squeeze(-1)
        gr = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        uxx = torch.autograd.grad(gr[:, 0].sum(), z, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(gr[:, 1].sum(), z, create_graph=True)[0][:, 1]
        return uxx + uyy
    for s in range(steps):
        def closure():
            opt.zero_grad()
            li = ((lap(Zi) - f_rhs(Zi))**2).mean(); lb = (model(Zb)**2).mean()
            loss = alpha*li + lb; loss.backward(); return loss
        opt.step(closure)
    with torch.no_grad(): return ((model(Zt).squeeze(-1) - u_true(Zt))**2).mean().item()
res = {}
for name, mk in [("KAN [2,5,1] G=5", lambda: KAN([2,5,1], G=5, seed=0)), ("KAN [2,10,1] G=5", lambda: KAN([2,10,1], G=5, seed=0)),
                 ("MLP [2,10,1]", lambda: MLP([2,10,1], act=torch.nn.Tanh, seed=0)), ("MLP [2,100,100,100,1]", lambda: MLP([2,100,100,100,1], act=torch.nn.Tanh, seed=0))]:
    m = mk(); mse = run(m, 40 if "KAN" in name else 40)
    res[name] = dict(params=m.n_params(), L2_mse=mse); print(name, res[name])
json.dump(res, open("results/exp5.json", "w"), indent=1)
