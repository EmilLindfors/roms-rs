#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gmsh>=4.11",
# ]
# ///
"""
Regenerate the Gmsh test meshes in tests/data/gmsh/.

A small bay with a curved coastline in the south and an island, meshed with
Frontal-Delaunay for quads + Blossom recombination. Blossom still leaves a few
triangles; `Mesh.SubdivisionAlgorithm = 1` splits every element into quads
(halving the size), which gives the all-quadrilateral meshes the solver needs.
Physical groups, deliberately numbered off the reader's
numeric convention (1 wall, 2 open, 3 tidal, 4 river) so the names decide:

    coast   (curve, tag 11)  south coastline + island  -> BoundaryTag::Wall
    open    (curve, tag 12)  west and north sides      -> BoundaryTag::Open
    outflow (curve, tag 13)  east side                 -> BoundaryTag::Custom(13)
    water   (surface, tag 20)

The all-quad mesh is written as MSH 4.1 ASCII, MSH 4.1 binary and MSH 2.2
ASCII; the three must load to the same Mesh2D (tests/gmsh_mesh_test.rs). The
quad-dominant mesh (no subdivision, a few triangles left) must be rejected.

Usage:
    uv run scripts/gmsh_fixtures.py
"""

import math
import os

import gmsh

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "gmsh")

LX, LY = 2000.0, 1200.0  # bay extent (m)
COAST_AMP = 150.0  # amplitude of the southern coastline (m)
ISLAND = (1200.0, 650.0, 220.0, 150.0)  # centre x, y and semi-axes (m)
H = 320.0  # target element size before subdivision (m)


def build(subdivide):
    """All quads with subdivision; without it, the simple recombination of a
    Frontal-Delaunay mesh at half the size (the resolution of the subdivided
    mesh) leaves triangles: a quad-dominant mesh."""
    occ = gmsh.model.occ
    # Southern coastline: y = COAST_AMP (1 + sin(2π x / LX)) / 2 through a spline
    n = 9
    coast_pts = [
        occ.addPoint(LX * i / (n - 1),
                     0.5 * COAST_AMP * (1.0 + math.sin(2.0 * math.pi * i / (n - 1))), 0.0)
        for i in range(n)
    ]
    ne = occ.addPoint(LX, LY, 0.0)
    nw = occ.addPoint(0.0, LY, 0.0)
    south = occ.addSpline(coast_pts)
    east = occ.addLine(coast_pts[-1], ne)
    north = occ.addLine(ne, nw)
    west = occ.addLine(nw, coast_pts[0])
    outer = occ.addCurveLoop([south, east, north, west])
    cx, cy, a, b = ISLAND
    island = occ.addEllipse(cx, cy, 0.0, a, b)
    hole = occ.addCurveLoop([island])
    water = occ.addPlaneSurface([outer, hole])
    occ.synchronize()

    model = gmsh.model
    model.addPhysicalGroup(1, [south, island], tag=11, name="coast")
    model.addPhysicalGroup(1, [west, north], tag=12, name="open")
    model.addPhysicalGroup(1, [east], tag=13, name="outflow")
    model.addPhysicalGroup(2, [water], tag=20, name="water")

    size = H if subdivide else 0.5 * H
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    if subdivide:
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom full-quad
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)  # simple
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1 if subdivide else 0)
    model.mesh.generate(2)


def write(name, version, binary):
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    path = os.path.join(OUT, name)
    gmsh.write(path)
    print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path)} bytes)")


def main():
    os.makedirs(OUT, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    for subdivide in (True, False):
        gmsh.model.add("bay")
        build(subdivide)
        types, tags, _ = gmsh.model.mesh.getElements(2)
        for t, ts in zip(types, tags):
            name = gmsh.model.mesh.getElementProperties(t)[0]
            print(f"{len(ts)} x {name}")
        # 3 = 4-node quadrangle
        assert (list(types) == [3]) == subdivide, "unexpected element types"
        if subdivide:
            write("bay_quads_v41.msh", 4.1, False)
            write("bay_quads_v41_binary.msh", 4.1, True)
            write("bay_quads_v22.msh", 2.2, False)
        else:
            write("bay_quad_dominant_v41.msh", 4.1, False)
        gmsh.model.remove()
    gmsh.finalize()


if __name__ == "__main__":
    main()
