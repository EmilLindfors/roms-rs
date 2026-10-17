#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
A 3D run against NorKyst-800 along a section, from the per-point files of
`examples/norkyst_profiles_3d.rs section=name:lon0,lat0:lon1,lat1:n`
(`profiles_<name>_00.txt` …, both models at NorKyst's z-levels).

Over the frames after `spinup_hours`, per point and depth: the mean T and S,
the mean velocity normal to the section (positive to the left of the
direction from the first end to the second), and the M2 amplitude of that
velocity (least squares on a mean, M2 and M4; needs ≥ 12.4 h). Draws model,
NorKyst and model − NorKyst against distance along the section and depth,
and prints the section's bias and RMSE of each field (depths both columns
reach).

Usage:
    uv run scripts/norkyst_section_3d.py <profiles_dir> <name> [spinup_hours=3] [out=<png>]
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M2 = 2 * math.pi / (12.4206012 * 3600)
M4 = 2 * M2


def read_point(path):
    header = path.read_text(encoding="utf-8").splitlines()[0]
    lon, lat = map(float, re.search(r"\(([\d.]+) E, ([\d.]+) N\)", header).groups())
    data = np.loadtxt(path, comments="#", ndmin=2)
    return lon, lat, data


def fit(t, x):
    """Mean and M2 amplitude of x(t) (least squares on 1, M2, M4)."""
    a = np.column_stack([np.ones_like(t), np.cos(M2 * t), np.sin(M2 * t),
                         np.cos(M4 * t), np.sin(M4 * t)])
    c, *_ = np.linalg.lstsq(a, x, rcond=None)
    return c[0], math.hypot(c[1], c[2])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    folder, name = Path(args[0]), args[1]
    spinup = float(opts.get("spinup_hours", 3))
    files = sorted(folder.glob(f"profiles_{name}_[0-9][0-9].txt"))
    if len(files) < 2:
        sys.exit(f"fewer than two points of section {name} in {folder}")
    # Columns too shallow to compare have no rows
    points = [p for p in map(read_point, files) if p[2].shape[0] > 0]
    if len(points) < 2:
        sys.exit(f"fewer than two points of section {name} with data")
    lon0, lat0 = points[0][:2]
    lon1, lat1 = points[-1][:2]
    kx = 111.32e3 * math.cos(math.radians(0.5 * (lat0 + lat1)))
    ky = 110.57e3
    dx, dy = (lon1 - lon0) * kx, (lat1 - lat0) * ky
    length = math.hypot(dx, dy)
    # Unit normal to the left of the section's direction (east, north)
    nx, ny = -dy / length, dx / length
    depths = np.unique(np.concatenate([p[2][:, 4] for p in points]))
    dist = np.array([math.hypot((p[0] - lon0) * kx, (p[1] - lat0) * ky) / 1e3 for p in points])
    # fields[f][model|norkyst][depth, point]
    names = ["T (°C)", "S", "mean u⊥ (m/s)", "M2 u⊥ (m/s)"]
    fields = np.full((4, 2, len(depths), len(points)), np.nan)
    beds = np.zeros((2, len(points)))
    for j, (_, _, d) in enumerate(points):
        d = d[d[:, 1] >= spinup]
        beds[:, j] = d[:, 2].mean(), d[:, 3].mean()
        n_frames = len(np.unique(d[:, 0]))
        for i, z in enumerate(depths):
            rows = d[d[:, 4] == z]
            rows = rows[np.all(np.isfinite(rows[:, 5:13]), axis=1)]
            # A depth both columns reach only through part of the tide
            # (near either bed) would bias the means and the M2 fit
            if len(rows) < 0.9 * n_frames:
                continue
            tt = rows[:, 0] - d[0, 0]
            for m, (ct, cs, cu, cv) in enumerate([(5, 6, 7, 8), (9, 10, 11, 12)]):
                un = rows[:, cu] * nx + rows[:, cv] * ny
                fields[0, m, i, j] = rows[:, ct].mean()
                fields[1, m, i, j] = rows[:, cs].mean()
                mean, amp = fit(tt, un)
                hours = (tt[-1] - tt[0]) / 3600
                fields[2, m, i, j] = mean
                fields[3, m, i, j] = amp if hours >= 12.4 else np.nan
    print(f"Section {name}: {len(points)} points over {length / 1e3:.1f} km, "
          f"({lon0:.3f}, {lat0:.3f}) → ({lon1:.3f}, {lat1:.3f}); normal ({nx:+.2f} E, {ny:+.2f} N)")
    print(f"  bed: model {beds[0].min():.0f}–{beds[0].max():.0f} m, NorKyst {beds[1].min():.0f}–{beds[1].max():.0f} m")
    for f, label in enumerate(names):
        diff = fields[f, 0] - fields[f, 1]
        ok = np.isfinite(diff)
        if not ok.any():
            print(f"  {label}: no common depths")
            continue
        both = np.isfinite(fields[f, 0]) & np.isfinite(fields[f, 1])
        corr = np.corrcoef(fields[f, 0][both], fields[f, 1][both])[0, 1] if both.sum() > 2 else np.nan
        print(f"  {label:14s} bias {np.nanmean(diff):+.3f}  RMSE {math.sqrt(np.nanmean(diff ** 2)):.3f}  "
              f"model {np.nanmean(fields[f, 0]):.3f}  NorKyst {np.nanmean(fields[f, 1]):.3f}  "
              f"pattern corr {corr:.2f}")
        deep = diff[depths >= 10]
        if np.isfinite(deep).any():
            print(f"  {'':14s} below 10 m: bias {np.nanmean(deep):+.3f}  "
                  f"RMSE {math.sqrt(np.nanmean(deep ** 2)):.3f}")

    fig, axes = plt.subplots(4, 3, figsize=(14, 13), sharex=True, sharey=True)
    for f, label in enumerate(names):
        a, b = fields[f, 0], fields[f, 1]
        lo, hi = np.nanmin([np.nanmin(a), np.nanmin(b)]), np.nanmax([np.nanmax(a), np.nanmax(b)])
        if f == 2:
            hi = max(abs(lo), abs(hi))
            lo = -hi
        cmap = "RdBu_r" if f == 2 else "viridis"
        for c, (title, v) in enumerate([("model", a), ("NorKyst", b)]):
            ax = axes[f, c]
            m = ax.pcolormesh(dist, depths, v, vmin=lo, vmax=hi, cmap=cmap, shading="nearest")
            ax.plot(dist, beds[c], "k-", lw=1)
            ax.set_title(f"{title}: {label}")
            fig.colorbar(m, ax=ax)
        d = a - b
        r = np.nanmax(np.abs(d)) if np.isfinite(d).any() else 1
        m = axes[f, 2].pcolormesh(dist, depths, d, vmin=-r, vmax=r, cmap="RdBu_r", shading="nearest")
        axes[f, 2].plot(dist, beds[0], "k-", lw=1)
        axes[f, 2].plot(dist, beds[1], "k--", lw=1)
        axes[f, 2].set_title(f"model − NorKyst: {label}")
        fig.colorbar(m, ax=axes[f, 2])
    axes[0, 0].invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("distance (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth (m)")
    fig.suptitle(f"Section {name}: hours ≥ {spinup:g}; beds: model solid, NorKyst dashed")
    out = opts.get("out", str(folder / f"section_{name}.png"))
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
