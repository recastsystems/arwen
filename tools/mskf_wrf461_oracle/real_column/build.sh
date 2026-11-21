#!/bin/bash
# Build the one-column WRF v4.7.1 MSKF driver in BUILD_DIR.
# usage: build.sh BUILD_DIR
# The WRF files are fetched byte-unmodified (public domain, sha256 checked
# against ../source-pin.json) and compiled with WRF's scalar-referee flags.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
build=$(realpath -m "$1"); mkdir -p "$build/wrf"; cd "$build"
declare -A pinned=(
  [module_cu_mskf.F]=4bca54b294def08ffc98ef69a30650c943e2c394f34802e75e50c7f1a2896fcb
  [module_cam_shr_kind_mod.F]=369bad7e4b528953dc4047ad38bf0b0d88120190f75541667be635c55edf1783
  [module_cam_error_function.F]=49fa938132d5d49da279a387315bca8ff23c5de332b939a1065e95a063371bf0
)
for f in "${!pinned[@]}"; do
  [ -f wrf/$f ] || curl -sSfL -o wrf/$f https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/phys/$f
  echo "${pinned[$f]}  wrf/$f" | sha256sum -c --quiet
done
defs="-DEM_CORE=1 -DNMM_CORE=0 -DDA_CORE=0 -DWRFPLUS=0 -DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4 -DNONSTANDARD_SYSTEM_SUBR -DUSE_ALLOCATABLES -Dwrfmodel -DBUILD_RRTMG_FAST=0 -DBUILD_SBM_FAST=1"
for f in module_cam_shr_kind_mod module_cam_error_function module_cu_mskf; do
  sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" wrf/$f.F > $f.G
  /lib/cpp -P -nostdinc $defs -traditional-cpp $f.G > $f.f90; rm -f $f.G
done
FF="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
gfortran -c $FF module_cam_shr_kind_mod.f90
gfortran -c $FF module_cam_error_function.f90
gfortran -c $FF "$here/../stub_wrf_error.F90"
gfortran -c $FF module_cu_mskf.f90
gfortran -c -O0 -ffree-form -ffree-line-length-none "$here/run_col.F90"
gfortran -o run_col module_cam_shr_kind_mod.o module_cam_error_function.o stub_wrf_error.o module_cu_mskf.o run_col.o
echo "built $build/run_col"
