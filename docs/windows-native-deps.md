# Windows Native Dependencies

The default feature set enables NetCDF I/O:

```powershell
cargo check
```

That pulls in the Rust `netcdf` crate, which depends on native netCDF-C and HDF5 libraries. On Windows, the most reliable setup is a small conda-forge environment because the `hdf5-metno-sys` build script understands conda's `Library\include`, `Library\lib`, and `Library\bin` layout.

## Install Miniforge

Install Miniforge with `winget`:

```powershell
winget install --id CondaForge.Miniforge3 --exact --scope user --accept-package-agreements --accept-source-agreements
```

Close and reopen PowerShell after the installer finishes. If `conda activate` is not available in a new shell, initialize PowerShell and reopen it again:

```powershell
& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" init powershell
```

## Create the Project Environment

Create a dedicated environment for this repository:

```powershell
conda create -y -n roms-rs -c conda-forge "hdf5=1.14.4" "libnetcdf<4.10" pkg-config
```

The HDF5 pin is intentional. `hdf5-metno-sys 0.10.1` accepts HDF5 `1.10.x`, `1.12.x`, and `1.14.0` through `1.14.5`. Newer conda-forge HDF5 `2.x` packages fail with:

```text
Invalid H5_VERSION: "2.1.0"
```

## Persist Build Variables

Store the native library root paths in the conda environment:

```powershell
conda env config vars set -n roms-rs HDF5_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs" NETCDF_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs"
```

Reactivate the environment so those variables are loaded:

```powershell
conda activate roms-rs
```

## Verify

From the repository root:

```powershell
cargo check
```

The default build should now find both HDF5 and netCDF-C. If Cargo cached a failed native build before the environment was fixed, clean those build scripts and retry:

```powershell
cargo clean -p hdf5-metno-sys -p netcdf-sys
cargo check
```

## One-Shell Alternative

If you do not want to activate the conda environment in the current shell, set the variables manually before building:

```powershell
$env:CONDA_PREFIX = "$env:USERPROFILE\miniforge3\envs\roms-rs"
$env:HDF5_DIR = $env:CONDA_PREFIX
$env:NETCDF_DIR = $env:CONDA_PREFIX
$env:PATH = "$env:CONDA_PREFIX\Library\bin;$env:CONDA_PREFIX\Scripts;$env:PATH"
cargo check
```

## Building Without NetCDF

For development that does not need NetCDF I/O, disable default features:

```powershell
cargo check --no-default-features --features parallel,simd
```

This avoids the native HDF5/netCDF-C dependency path entirely, but NetCDF-backed I/O will not be available.
