"""A WRF v4.8.0 real.exe input under gwd_opt = 1 carries ELVMAX to the KIM drag.

real.exe v4.8.0 writes ELVMAX next to VAR, CON, OA1-4 and OL1-4 whenever
gwd_opt = 1 (Registry.EM_COMMON, package gwd_used_1).  The wrf_480 KIM drag
reads it (wrfinput_terrain_drag_inputs requires it), so the native door must
admit the name instead of refusing the file as an unmapped variable.
"""
import netCDF4
import numpy as np
import pytest

from gpuwm.ingest import wrfinput as wi
from gpuwm.static.orographic import drag_static_fields, required_static_fields
from test_wrfinput_physics_fields import (
    _cfg, _dimensions, _input, _surface_values,
)


def _wrf480_gwd_input(tmp_path, gwdo_version, *, elvmax=True):
    cfg = _cfg(gwd_opt=1, gwdo_version=gwdo_version)
    shape = cfg.ny, cfg.nx
    values = _surface_values(cfg)
    values["HGT"] = np.full(shape, np.float32(253.27182))
    for index, name in enumerate(required_static_fields(0, 1)):
        values[name] = (np.arange(cfg.ny * cfg.nx, dtype=np.float32).reshape(shape)
                        + np.float32(index + .125))
    path = _input(tmp_path / "input", cfg, values)
    if elvmax:
        # Written the way real.exe v4.8.0 writes it: a mass-point 2-D field.
        values["ELVMAX"] = np.full(shape, np.float32(1873.5)) + values["HGT"]
        with netCDF4.Dataset(path, "a") as dataset:
            dataset.createVariable("ELVMAX", "f4", ("Time", "south_north", "west_east"))[0] = values["ELVMAX"]
    return cfg, values, path


def _read(path, cfg):
    return wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                            require_complete=False)


def test_wrf480_input_hands_elvmax_to_the_wrf480_drag(tmp_path):
    cfg, values, path = _wrf480_gwd_input(tmp_path, "wrf_480")
    static = wi.wrfinput_terrain_drag_inputs(_read(path, cfg), cfg)["terrain_drag_static"]
    assert set(static) == set(drag_static_fields(0, 1, "wrf_480")) | {"HGT_M"}
    np.testing.assert_array_equal(np.asarray(static["ELVMAX"], np.float32).view("u4"),
                                  values["ELVMAX"].view("u4"))


def test_wrf471_generation_reads_the_same_file_and_leaves_elvmax_unused(tmp_path):
    cfg, _, path = _wrf480_gwd_input(tmp_path, "wrf_471")
    static = wi.wrfinput_terrain_drag_inputs(_read(path, cfg), cfg)["terrain_drag_static"]
    assert set(static) == set(required_static_fields(0, 1)) | {"HGT_M"}


def test_wrf480_drag_without_elvmax_is_still_refused(tmp_path):
    cfg, _, path = _wrf480_gwd_input(tmp_path, "wrf_480", elvmax=False)
    with pytest.raises(ValueError, match="ELVMAX"):
        wi.wrfinput_terrain_drag_inputs(_read(path, cfg), cfg)


def test_an_unknown_variable_is_still_refused_next_to_elvmax(tmp_path):
    cfg, _, path = _wrf480_gwd_input(tmp_path, "wrf_480")
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createVariable("NEW_DRAG_STATE", "f4", ("Time", "south_north", "west_east"))[0] = 1.
    with pytest.raises(ValueError, match="NEW_DRAG_STATE"):
        _read(path, cfg)
