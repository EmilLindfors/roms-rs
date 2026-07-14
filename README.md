# roms-rs

Highly experimental Discontinuous Galerkin (DG) solver for coastal ocean modeling, targeting simulation of currents along the Norwegian coast, written in Rust.

## Windows HDF5/netCDF-C Setup

The default feature set enables NetCDF I/O, so Windows builds need native HDF5 and netCDF-C libraries available to Cargo.

Use the Miniforge/conda-forge setup in [docs/windows-native-deps.md](docs/windows-native-deps.md). The important detail is to pin HDF5 to the supported 1.14 line:

```powershell
conda create -y -n roms-rs -c conda-forge "hdf5=1.14.4" "libnetcdf<4.10" pkg-config
conda env config vars set -n roms-rs HDF5_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs" NETCDF_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs"
conda activate roms-rs
cargo check
```

For work that does not need NetCDF I/O:

```powershell
cargo check --no-default-features --features parallel,simd
```

## Changelog

Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).
