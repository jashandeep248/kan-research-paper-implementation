import torch, numpy as np


def rmse(model, X, y):
    with torch.no_grad():
        return torch.sqrt(((model(X) - y) ** 2).mean()).item()


def fit(model, data, steps=100, opt="lbfgs", lr=1.0, lamb=0.0, update_grid=True,
        mu1=1.0, mu2=1.0, log_every=0):
    """Train on data = dict(Xtr,ytr,Xte,yte). LBFGS (as in the paper) or Adam."""
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]
    is_kan = hasattr(model, "layers")
    if is_kan and update_grid:
        model.update_grids(Xtr)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt == "lbfgs":
        o = torch.optim.LBFGS(params, lr=lr, history_size=10, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        o = torch.optim.Adam(params, lr=lr)
    tr, te = [], []
    import copy
    best = (rmse(model, Xtr, ytr), copy.deepcopy(model.state_dict()))
    for s in range(steps):
        def closure():
            o.zero_grad()
            if is_kan and lamb > 0:
                out, reg = model.forward_reg(Xtr, lamb, mu1, mu2)
                loss = ((out - ytr) ** 2).mean() + reg
            else:
                loss = ((model(Xtr) - ytr) ** 2).mean()
            loss.backward()
            return loss
        o.step(closure)
        tr.append(rmse(model, Xtr, ytr))
        if np.isfinite(tr[-1]) and tr[-1] < best[0]:
            best = (tr[-1], copy.deepcopy(model.state_dict()))
        te.append(rmse(model, Xte, yte))
        if log_every and s % log_every == 0:
            print(f"  step {s:4d} train {tr[-1]:.3e} test {te[-1]:.3e}")
    # LBFGS can occasionally jump out of a good basin: keep the best-train-loss iterate
    if opt == "lbfgs" and lamb == 0:
        model.load_state_dict(best[1])
        tr[-1], te[-1] = rmse(model, Xtr, ytr), rmse(model, Xte, yte)
    return tr, te


def make_data(f, n_in, n_train=1000, n_test=1000, lo=-1, hi=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n_train + n_test, n_in, generator=g) * (hi - lo) + lo
    y = f(X).unsqueeze(-1)
    return dict(Xtr=X[:n_train], ytr=y[:n_train], Xte=X[n_train:], yte=y[n_train:])
