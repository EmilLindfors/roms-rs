# Gmsh meshes

The 2D solver takes meshes of first-order, straight-sided quadrilaterals.
`read_gmsh_mesh` loads them from Gmsh MSH 4.1 files (ASCII or binary; Gmsh's
default format) and from MSH 2.2 ASCII files. This page covers the Gmsh settings
that produce such meshes and how the physical groups become boundary conditions.

## Coordinates

Mesh in metres, in a projected system (UTM zone 32/33 N for the Norwegian coast;
see `io::UtmProjection`). The reader keeps x and y and ignores z.

## All-quadrilateral meshes

Recombination alone often leaves a few triangles, and the reader rejects those.
To get only quads, recombine and then subdivide:

```
Mesh.Algorithm = 8;               // Frontal-Delaunay for quads
Mesh.RecombinationAlgorithm = 3;  // Blossom full-quad
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;    // split every element into quads: all-quad
Mesh.ElementOrder = 1;
```

Subdivision halves the element size, so set the size fields to twice the target
size.

The reader also rejects:
- degenerate or non-convex quads (their bilinear map has J ≤ 0);
- high-order elements;
- periodic and partitioned meshes.

Each error names the Gmsh element and node tags.

## Boundary groups

Put every boundary curve in a named physical curve, and put the water in a
physical surface. Once a model has any physical group, Gmsh saves only the
elements in physical groups, so without the surface the file has no quads.

| Group name (case-insensitive) | `BoundaryTag` |
|---|---|
| `coast`, `coastline`, `land`, `shore`, `shoreline`, `wall`, `walls`, `closed` | `Wall` |
| `open`, `ocean`, `sea`, `open_boundary` | `Open` |
| `tidal`, `tide`, `tides`, `tidal_forcing` | `TidalForcing` |
| `river`, `rivers` | `River` |
| any other name | `Custom(group number)` |

Groups without a name map by their number: 1 is a wall, 2 open, 3 tidal forcing,
4 river, 5 Dirichlet, 6 Neumann, and any other n is `Custom(n)`. A boundary edge
outside every group is a wall. A curve in two groups that map to different tags
is an error.

## Example: a fjord arm with a refined farm site

```
SetFactory("OpenCASCADE");
// Coastline polygon in UTM metres (e.g. from io::coastline), open to the sea
// along the western edge
Point(1) = {500000, 7050000, 0};  Point(2) = {512000, 7050000, 0};
Point(3) = {512000, 7056000, 0};  Point(4) = {500000, 7056000, 0};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("coast") = {1, 2, 3};
Physical Curve("open") = {4};
Physical Surface("water") = {1};

// 40 m quads (80 m before subdivision) within 500 m of the farm, 800 m offshore
Point(10) = {506000, 7053000, 0};
Field[1] = Distance; Field[1].PointsList = {10};
Field[2] = Threshold; Field[2].InField = 1;
Field[2].SizeMin = 80; Field[2].SizeMax = 1600;
Field[2].DistMin = 500; Field[2].DistMax = 4000;
Background Field = 2;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 3;
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;
```

Mesh and save it with `gmsh fjord.geo -2 -format msh41 -o fjord.msh`. Then load
it and dispatch the conditions by tag:

```rust
use dg_rs::boundary::{CharacteristicOBC, MultiBoundaryCondition2D, Reflective2D};
use dg_rs::mesh::read_gmsh_mesh;

let mesh = read_gmsh_mesh(std::path::Path::new("fjord.msh"))?;
let wall = Reflective2D::new();
let sea = CharacteristicOBC::new(tides); // e.g. BoundaryTides from a TidalAtlas
let bc = MultiBoundaryCondition2D::new(&wall).with_open(&sea);
```

The cages themselves do not need to be meshed. A cage is usually smaller than
an element, and `CageDrag2D` weights its drag onto the nodes it covers:

```rust
use dg_rs::source::{CageDrag2D, NetCage};

// Two 50 m ring cages with 20 m nets, solidity 0.25 (clean netting)
let cages = [
    NetCage::circular([506000.0, 7053000.0], 25.0, 20.0, 0.25),
    NetCage::circular([506070.0, 7053000.0], 25.0, 20.0, 0.25),
];
let drag = CageDrag2D::new(&mesh, &ops, &cages);
let physics = builder.with_cage_drag(drag).build();
```

## Test meshes

`scripts/gmsh_fixtures.py` builds the meshes in `tests/data/gmsh/` with the Gmsh
Python API (`uv run scripts/gmsh_fixtures.py`). It meshes a bay with a curved
coastline and an island and saves the result in the three supported formats,
plus a quad-dominant version that the reader must reject.
