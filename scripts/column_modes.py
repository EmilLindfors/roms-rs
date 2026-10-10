"""Flat-column linear model of the 3D scheme at rest (TODO P1.3 research).

Run: `uv run --with numpy python scripts/column_modes.py [depth_m]`. Result
(2026-10-10, notes/geometry-mesh.md): neutral (all K eigenvalues real and
positive) for every c in [-1, 1.5] at 23-31 m; the vertical scheme alone never
destabilises the rest state.

Hydrostatic Boussinesq, rigid lid, one horizontal wavenumber k (it scales out:
lambda^2 = -k^2 eig(K)). Layers on the fixture's sigma grid (20 Song-Haidvogel
5/0.4 levels, pure sigma) in a column of depth H, the fixture's summer
pycnocline, linear EOS. The PGF is the code's Hermite pressure integral with
harmonic-mean slopes, linearised by finite differences; the vertical advection
acts through the background's surface values only (Omega_bar = 0), with
Akima's curvature correction at weight c (0 centred, 1/2 Hermite mean, 1 Akima).

    du_l/dt = -i k (g/rho0) p'_l  + rigid-lid projection
    h_l drho'_l/dt = -W_{l+1/2} (rho*_{l+1/2} - rho_l) - W_{l-1/2} (rho_l - rho*_{l-1/2})
    W_{l+1/2} = -i k sum_{m<=l} h_m u_m

so d2u/dt2 = -k^2 K u with K = (g/rho0) Pi P R L. Neutral iff eig(K) real >= 0.
"""
import sys
import numpy as np

G, RHO0 = 9.81, 1025.0
T0, S0, ALPHA, BETA = 10.0, 35.0, 1.7e-4, 7.6e-4


def sigma_grid(n=20, ts=5.0, tb=0.4):
    s = -1.0 + np.arange(n + 1) / n
    cs = (1 - np.cosh(ts * s)) / (np.cosh(ts) - 1)
    cb = np.tanh(tb * (s + 1)) / np.tanh(tb) - 1
    w = ts / (ts + tb) * cs + tb / (ts + tb) * cb
    return 0.5 * (w[:-1] + w[1:]), np.diff(w)


def profile(z, deep=0.002, centre=-15.0, scale=4.0):
    step = 0.5 * (1 + np.tanh((z - centre) / scale))
    t = T0 + 4 * step + deep * np.minimum(z, 0)
    s = S0 - 1.5 * step
    return t, s


def rho_of(t, s):
    return RHO0 * (1 - ALPHA * (t - T0) + BETA * (s - S0))


def hm_slopes(f, x):
    """Harmonic-mean slopes of f at x (one-sided at the ends)."""
    sec = np.diff(f) / np.diff(x)
    d = np.empty_like(f)
    d[0], d[-1] = sec[0], sec[-1]
    a, b = sec[:-1], sec[1:]
    prod = a * b
    with np.errstate(invalid="ignore", divide="ignore"):
        d[1:-1] = np.where(prod > 0, 2 * prod / (a + b), 0.0)
    return d


def pressure(rho, z, eta=0.0):
    """The code's Hermite pressure integral at the levels (per g)."""
    n = len(rho)
    d = hm_slopes(rho, z)
    p = np.empty(n)
    dz = eta - z[-1]
    p[-1] = rho[-1] * dz + 0.5 * d[-1] * dz * dz
    for l in range(n - 2, -1, -1):
        h = z[l + 1] - z[l]
        p[l] = p[l + 1] + h * (0.5 * (rho[l] + rho[l + 1]) + h * (d[l] - d[l + 1]) / 12)
    return p


def surface_values(f, z, c):
    """Background values at the interior surfaces: mean + c * Akima's correction."""
    d = hm_slopes(f, z)  # Akima's slopes (one-sided ends: the code repeats the end gradient)
    h = np.diff(z)
    return 0.5 * (f[:-1] + f[1:]) + c * h * (d[:-1] - d[1:]) / 6


def operator(c, depth=25.0, n=20, deep=0.002, separate=True, cs=None):
    sr, ds = sigma_grid(n)
    z = sr * depth
    hl = ds * depth
    t, s = profile(z, deep)
    rho = rho_of(t, s)
    # Background surface values of rho (T and S reconstructed separately, as the code does)
    if separate:
        rs = rho_of(surface_values(t, z, c), surface_values(s, z, c))
    else:
        rs = surface_values(rho, z, c)
    # P: dp'/drho' by central differences
    P = np.empty((n, n))
    for m in range(n):
        e = np.zeros(n)
        e[m] = 1e-6
        P[:, m] = (pressure(rho + e, z) - pressure(rho - e, z)) / 2e-6
    # R: drho'/dt = R W, W at the n-1 interior surfaces
    R = np.zeros((n, n - 1))
    for f in range(n - 1):  # surface between layers f and f+1
        R[f, f] -= (rs[f] - rho[f]) / hl[f]
        R[f + 1, f] -= (rho[f + 1] - rs[f]) / hl[f + 1]
    # L: W = -i k L u (the -i k is in the overall factor)
    L = np.tril(np.ones((n - 1, n))) * hl[None, :]
    Pi = np.eye(n) - np.outer(np.ones(n), hl) / depth
    # du/dt = -ik g/rho0 Pi P rho', drho'/dt = R (-ik L u) => d2u = -k^2 (g/rho0) Pi P R L u
    K = G / RHO0 * Pi @ P @ R @ L
    return K, z, rho, rs


def modes(c, **kw):
    K = operator(c, **kw)[0]
    ev = np.linalg.eigvals(K)
    ev = ev[np.argsort(-ev.real)]
    return ev


if __name__ == "__main__":
    depth = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
    for c in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        ev = modes(c, depth=depth)
        # internal modes: drop the rigid-lid null mode (eigenvalue ~0)
        im = np.abs(ev.imag).max()
        neg = ev.real.min()
        speeds = np.sqrt(np.abs(ev.real[:4]))
        print(f"c {c:+.2f}: max |Im eig K| {im:.2e}, min Re {neg:+.2e}, "
              f"leading speeds {np.array2string(speeds, precision=4)} m/s")
