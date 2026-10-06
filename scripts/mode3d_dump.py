"""Inspect a growing 3D mode from `froya_real_data levels=N debug_3d=...,dump=PREFIX`.

The example writes PREFIX.geom.bin (node x, y, bed, mass weights; the
layer centres' sigma and the layers' d_sigma) and PREFIX.{init,prev,end}.bin
(eta, ubar, vbar, then u, v, T, S, rho per level) as little-endian f64 after
a header of (elements, nodes, levels) as u64 (TODO P1.3).

    uv run --with numpy python scripts/mode3d_dump.py output/mode36/base

prints how the perturbation (state minus rest) grew between the `prev` and
`end` dumps per field (a clean eigenmode has one ratio and a correlation near
1), the fastest elements, and the vertical structure of the fastest one.
"""

import sys

import numpy as np


def geometry(prefix):
    ne, nn, nl = map(int, np.fromfile(prefix + ".geom.bin", dtype="<u8", count=3))
    data = np.fromfile(prefix + ".geom.bin", dtype="<f8", offset=24)
    n = ne * nn
    fields = {}
    offset = 0
    for name, size in [("x", n), ("y", n), ("bed", n), ("mass", n), ("sigma", nl), ("d_sigma", nl)]:
        fields[name] = data[offset : offset + size]
        offset += size
    return ne, nn, nl, fields


def dump(path):
    ne, nn, nl = map(int, np.fromfile(path, dtype="<u8", count=3))
    data = np.fromfile(path, dtype="<f8", offset=24)
    n, n3 = ne * nn, ne * nn * nl
    fields = {}
    offset = 0
    for name in ["eta", "ubar", "vbar"]:
        fields[name] = data[offset : offset + n]
        offset += n
    for name in ["u", "v", "T", "S", "rho"]:
        fields[name] = data[offset : offset + n3].reshape(n, nl)
        offset += n3
    return fields


def main(prefix):
    ne, nn, nl, geom = geometry(prefix)
    rest, prev, end = (dump(f"{prefix}.{which}.bin") for which in ["init", "prev", "end"])
    pert = {
        name: (prev[name] - rest[name], end[name] - rest[name])
        for name in ["u", "v", "T", "S", "rho", "eta", "ubar"]
    }
    print("growth prev -> end per field (ratio of maxima, projection, correlation):")
    for name, (a, b) in pert.items():
        projection = (a * b).sum() / (a * a).sum()
        correlation = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        print(
            f"  {name:5s} {np.abs(b).max():.3e} / {np.abs(a).max():.3e} = "
            f"{np.abs(b).max() / np.abs(a).max():.3f}; {projection:.3f}; {correlation:.4f}"
        )
    speed = np.hypot(*(pert[name][1] for name in ["u", "v"]))
    per_element = speed.reshape(ne, nn * nl).max(axis=1)
    depth = rest["eta"] - geom["bed"]
    print("fastest elements:")
    for k in np.argsort(-per_element)[:10]:
        nodes = slice(k * nn, (k + 1) * nn)
        node, level = divmod(int(np.argmax(speed[nodes])), nl)
        print(
            f"  {k:5d}: {per_element[k]:.2e} m/s at node {node} level {level}; "
            f"depths {np.round(depth[nodes], 1).tolist()}"
        )
    k = int(np.argmax(per_element))
    z = rest["eta"][:, None] + geom["sigma"][None, :] * depth[:, None]
    drho = pert["rho"][1]
    np.set_printoptions(linewidth=200, precision=2)
    print(f"element {k}, per node: z of the levels, |u| and δρ (each over its largest)")
    for i in range(nn):
        idx = k * nn + i
        print(f"  node {i}: depth {depth[idx]:.1f} m")
        print("    z  ", z[idx])
        print("    |u|", speed[idx] / speed.max())
        print("    δρ ", drho[idx] / np.abs(drho).max())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/mode36/base")
