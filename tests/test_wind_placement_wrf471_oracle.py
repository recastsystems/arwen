"""Native-table turbine placement against WRF v4.7.1's own placement code.

``mav_project_ll`` (gpuwm/core/kernels/wind_mav.cu) places a turbine of a
native WRF table as ``init_module_wind_fitch`` does: ``map_set`` anchored at
XLAT/XLONG(1,1), float32 ``latlon_to_ij``, then ``nint``.  The fixtures are
WRF's own answers on eleven farms (Lambert secant, tangent and southern;
Mercator, across the dateline; polar stereographic, both poles; nests up to
three deep), from the jacobbieker/woof fork's oracle (pull request 123,
tests/data/oracles/windfarm_placement_wrf471/SOURCE.md).  Every turbine must
land in WRF's cell, the 30-odd per farm planted on cell edges included.
"""
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

ROOT = Path(__file__).parent / "data" / "oracles" / "windfarm_placement_wrf471"
CASES = sorted(p.name for p in ROOT.iterdir() if (p / "grid.txt").is_file())


def _case(name):
    d = ROOT / name
    lines = (d / "grid.txt").read_text().split("\n")
    proj, tl1, tl2, stdlon, dx = lines[0].split()
    lat1, lon1 = (float(x) for x in lines[2].split())
    read = lambda f, t: np.fromfile(d / f, dtype=t)
    return (int(proj), float(tl1), float(tl2), float(stdlon), float(dx), lat1, lon1,
            read("lat.bin", "<f4"), read("lon.bin", "<f4"),
            read("ival.bin", "<i4"), read("jval.bin", "<i4"), read("ts_rx.bin", "<f4"))


def test_fixtures_are_the_eleven_cases():
    assert len(CASES) == 11
    for name in CASES:
        *_, lat, lon, ival, jval, rx = _case(name)
        assert lat.size == lon.size == ival.size == jval.size == rx.size > 0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_native_placement_lands_every_turbine_in_wrfs_cell(name):
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    proj, tl1, tl2, stdlon, dx, lat1, lon1, lat, lon, ival, jval, rx = _case(name)
    nt = lat.size
    # A grid large enough that no WRF cell falls outside it; the kernel
    # reads only the corner of the latitude and longitude arrays.
    nx = ny = int(max(ival.max(), jval.max())) + 2
    corner_lat = cp.full(1, lat1, cp.float32)
    corner_lon = cp.full(1, lon1, cp.float32)
    ll = cp.asarray(np.stack([lat, lon], axis=1).astype(np.float32))
    col = cp.empty(nt, cp.int32)
    get_kernel("wind_mav", "mav_project_ll")(
        ((nt + 127) // 128,), (128,),
        (ll, corner_lat, corner_lon, col, np.int32(nt), np.int32(nx), np.int32(ny),
         np.int32(proj), np.float32(dx), np.float32(tl1), np.float32(tl2), np.float32(stdlon)))
    col = cp.asnumpy(col)
    assert np.all(col >= 0)
    i, j = col % nx + 1, col // nx + 1
    edge = np.abs(np.abs(rx - np.trunc(rx)) - 0.5) < 1e-3
    bad = np.flatnonzero((i != ival) | (j != jval))
    assert bad.size == 0, (name, bad.size, int(edge[bad].sum()), nt)
