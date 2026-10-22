import sys, torch; sys.path.insert(0, ".")
from kan import KAN, KANLayer
from kan.layer import bspline_basis, extend_grid
torch.set_default_dtype(torch.float64)

def test_partition_of_unity():
    g = extend_grid(torch.linspace(-1, 1, 6).repeat(2, 1), 3)
    x = torch.rand(200, 2) * 1.98 - 0.99
    assert torch.allclose(bspline_basis(x, g, 3).sum(-1), torch.ones(200, 2))

def test_param_count():            # O(N^2 L (G+k)) + w_b, w_s  (paper "Parameter count")
    m = KAN([3, 4, 1], G=5, k=3)
    assert m.n_params() == 3*4*(5+3) + 4*1*(5+3) + 2*(3*4 + 4*1)

def test_grid_extension_preserves_function():
    l = KANLayer(1, 1, G=5); l.coef.data = torch.randn(1, 1, 8)
    x = torch.rand(2000, 1) * 2 - 1; y0 = l.splines(x); l.extend_grid_to(10, x)   # nested grid
    assert (y0 - l.splines(x)).abs().max() < 1e-6

def test_shapes_and_backprop():
    m = KAN([2, 3, 1]); x = torch.rand(16, 2) * 2 - 1
    m(x).sum().backward(); assert all(p.grad is not None for p in m.parameters())

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
