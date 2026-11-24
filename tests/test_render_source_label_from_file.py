"""A map's source line names the model that made the file.

The breakage: ``gpuwm render`` on ArWen Global export tapes labelled every
map "source: WOOF 3.0.5", the renderer's own brand and version, although
each tape names its model in ``GPUWM_MODEL_LABEL`` (gpuwm 3.0.5;
``gpuwm-global render`` passed its own label and was right).  A file that
names no model keeps exactly the label it had.
"""
from __future__ import annotations

from pathlib import Path

import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm import render, rustwx  # noqa: E402


def _tape(path: Path, **attrs) -> Path:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        for key, value in attrs.items():
            ds.setncattr(key, value)
    return path


def test_a_file_that_names_its_model_labels_its_maps(tmp_path):
    tape = _tape(tmp_path / "wrfout_d01_a.nc", GPUWM_MODEL_LABEL="ArWen Global",
                 GPUWM_VERSION="3.0.5")
    assert render.source_label_for(tape) == "ArWen Global"
    versioned = _tape(tmp_path / "wrfout_d01_b.nc",
                      GPUWM_MODEL_LABEL="ArWen Global",
                      GPUWM_MODEL_VERSION="0.4.0")
    assert render.source_label_for(versioned) == "ArWen Global 0.4.0"


def test_a_file_that_names_no_model_keeps_the_label_it_had(tmp_path):
    ours = _tape(tmp_path / "wrfout_d01_c.nc", GPUWM_VERSION="3.0.5")
    assert render.source_label_for(ours) == render.default_source_label()
    assert render.source_label_for(tmp_path / "missing.nc") == \
        render.default_source_label()


@pytest.mark.parametrize("label", ["", "   ", "a" * 33, "Model | other"])
def test_a_label_the_map_row_cannot_print_counts_as_none(tmp_path, label):
    tape = _tape(tmp_path / "wrfout_d01_d.nc", GPUWM_MODEL_LABEL=label)
    assert render.source_label_for(tape) == render.default_source_label()


def test_the_files_model_reaches_the_renderer_and_a_given_label_still_wins(
        monkeypatch, tmp_path):
    import subprocess

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(command, **kwargs):
        seen.append([str(part) for part in command])
        return Result()

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr("gpuwm.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))
    tape = _tape(tmp_path / "wrfout_d01_2026-08-30_18-00-00",
                 GPUWM_MODEL_LABEL="ArWen Global", GPUWM_VERSION="3.0.5")

    def label_of(command):
        return command[command.index("--source-label") + 1]

    render.render_wrfouts_rust([tape], products="mslp_10m_winds", timeidx=0,
                               outdir=tmp_path / "png", size=(800, 600))
    drawn = [command for command in seen if "--source-label" in command]
    assert drawn, seen
    assert label_of(drawn[-1]) == "ArWen Global"

    seen.clear()
    render.render_wrfouts_rust([tape], products="mslp_10m_winds", timeidx=0,
                               outdir=tmp_path / "png", size=(800, 600),
                               source_label="ArWen Global 0.3.0")
    drawn = [command for command in seen if "--source-label" in command]
    assert label_of(drawn[-1]) == "ArWen Global 0.3.0"
