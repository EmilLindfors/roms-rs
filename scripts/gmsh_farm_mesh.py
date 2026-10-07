#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gmsh>=4.11",
# ]
# ///
"""
Build the farm-refined fjord mesh used by the local time stepping benchmark
(examples/local_time_stepping_farm.rs) and its gate test: the fjord arm of
docs/gmsh-meshes.md, 12 × 6 km with its open side to the west, refined to
≈ 20 m quads at a fish farm in the middle, growing to ≈ 450 m at the far
end (the size ramp would reach 800 m at 10 km).

Quads come from Frontal-Delaunay for quads + Blossom recombination +
subdivision (all-quad; subdivision halves the size, so the size field asks
for twice the target). Coordinates in local metres (origin at the SW
corner). Physical groups: "coast" (south, east, north), "open" (west),
"water".

With farm=S the quads at the farm are ≈ S m instead of 20 (the ramp to the
far size unchanged): `farm=50 out=tests/data/gmsh/fjord_farm_3d.msh` is the
mesh the viewer's 3D fjord runs on (viz/src/scenario.rs), whose one global
step the farm's quads set.

Usage:
    uv run scripts/gmsh_farm_mesh.py [out=tests/data/gmsh/fjord_farm.msh] [farm=20]
"""

import sys

import gmsh

LX, LY = 12_000.0, 6_000.0
FARM = (6_000.0, 3_000.0)
# Before subdivision (halved by it)
SIZE_FARM, SIZE_FAR = 40.0, 1_600.0
DIST_MIN, DIST_MAX = 300.0, 10_000.0


def main(out: str, size_farm: float) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fjord_farm")
    geo = gmsh.model.geo

    corners = [geo.addPoint(x, y, 0) for x, y in [(0, 0), (LX, 0), (LX, LY), (0, LY)]]
    lines = [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    loop = geo.addCurveLoop(lines)
    surface = geo.addPlaneSurface([loop])
    farm = geo.addPoint(*FARM, 0)
    geo.synchronize()
    gmsh.model.mesh.embed(0, [farm], 2, surface)

    gmsh.model.addPhysicalGroup(1, lines[:3], name="coast")
    gmsh.model.addPhysicalGroup(1, [lines[3]], name="open")
    gmsh.model.addPhysicalGroup(2, [surface], name="water")

    field = gmsh.model.mesh.field
    distance = field.add("Distance")
    field.setNumbers(distance, "PointsList", [farm])
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    field.setNumber(threshold, "SizeMin", size_farm)
    field.setNumber(threshold, "SizeMax", SIZE_FAR)
    field.setNumber(threshold, "DistMin", DIST_MIN)
    field.setNumber(threshold, "DistMax", DIST_MAX)
    field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(out)
    types, tags, _ = gmsh.model.mesh.getElements(2)
    print(f"{out}: {sum(len(t) for t in tags)} surface elements (types {list(types)})")
    gmsh.finalize()


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    # The quad size asked for is halved by the subdivision
    farm = 2.0 * float(args["farm"]) if "farm" in args else SIZE_FARM
    main(args.get("out", "tests/data/gmsh/fjord_farm.msh"), farm)
