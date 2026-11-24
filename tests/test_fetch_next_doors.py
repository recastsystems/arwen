"""A finished fetch names the next step of the model that reads it.

The breakage: the ArWen Global README's first step (``gpuwm fetch
--source gdas --cycle 2026-09-01T00 --hours 0 --mode full-file --out
cases/baseline-2026090100``) ended with "next: ... gpuwm prep --source
gdas ...", the regional route, with gpuwm-global installed (gpuwm 3.0.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm import fetch_routes, next_doors

ANALYSIS = "gdas.t00z.pgrb2.0p25.f000"


def _fetched(out: Path, *, mode: str = "full-file", area=None) -> Path:
    out.mkdir(parents=True)
    (out / ANALYSIS).write_bytes(b"GRIB")
    (out / fetch_routes.MANIFEST_NAME).write_text(json.dumps({
        "mode": mode, "area": area,
        "files": [{"name": ANALYSIS, "forecast_hour": 0, "role": "atmosphere"},
                  {"name": "gdas-series.tsv", "forecast_hour": None,
                   "role": "series"}]}), encoding="utf-8")
    (out / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps({
        "argv": ["--source", "gdas", "--input-list", str(out / "inputs.txt")],
        "caller_supplies": ["--wps-namelist", "--output-root"]}),
        encoding="utf-8")
    return out


@pytest.fixture
def with_global(monkeypatch, tmp_path):
    """gpuwm-global installed, shipping one config that reads the case."""

    configs = tmp_path / "shipped"
    configs.mkdir()
    for name, truncation in (
            ("arwen_global_gdas_t255_native_sl_si_24h", 255),
            ("arwen_global_gdas_t255_native_sl_si_24h_dt600", 255),
            ("arwen_global_gdas_t63_native_sl_si_24h", 63),
            ("arwen_global_gdas_t1023_native_sl_si_24h", 1023)):
        (configs / f"{name}.toml").write_text(
            '[arwen_global]\nname = "x"\n[grid]\n'
            f'truncation = {truncation}\n[initial]\nmode = "analysis"\n'
            f'analysis_grib = "cases/baseline-2026090100/{ANALYSIS}"\n',
            encoding="utf-8")
    # Another planet started from Earth's analysis: not a forecast to offer.
    (configs / "arwen_global_world_t255_x.toml").write_text(
        '[arwen_global]\nname = "w"\n[grid]\ntruncation = 255\n'
        '[initial]\nmode = "analysis"\n'
        f'analysis_grib = "cases/baseline-2026090100/{ANALYSIS}"\n'
        '[world]\nlabel = "x"\n', encoding="utf-8")
    (configs / "arwen_global_moist_smoke.toml").write_text(
        '[arwen_global]\nname = "y"\n[initial]\nmode = "analytic"\n',
        encoding="utf-8")
    monkeypatch.setattr(next_doors, "shipped_configs_folder",
                        lambda module=next_doors.GLOBAL_MODULE: configs)
    monkeypatch.setattr(next_doors, "installed", lambda module: True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _lines(out: Path) -> list[str]:
    return list(fetch_routes.prep_handoff_lines("gdas", out))


def test_a_whole_globe_analysis_names_the_global_door_first(with_global):
    out = _fetched(with_global / "cases" / "baseline-2026090100")
    lines = _lines(out)
    text = "\n".join(lines)
    assert lines[0] == ("fetch gdas: next, for ArWen Global (gpuwm-global, "
                        "installed):")
    assert "gpuwm-global go CONFIG --outdir OUT" in text
    # One config per grid, smallest first; the dt600 variant of T255 is not
    # listed beside its base, and the analytic smoke run is not listed.
    listed = [line.strip()[2:].strip() for line in lines
              if line.startswith("  #   ")]
    assert listed == ["arwen_global_gdas_t63_native_sl_si_24h",
                      "arwen_global_gdas_t255_native_sl_si_24h",
                      "arwen_global_gdas_t1023_native_sl_si_24h"]
    assert "4 shipped configs read this file" in text
    # The regional block is still there, under a heading naming its model.
    regional = [line for line in lines if "gpuwm prep " in line]
    assert len(regional) == 1
    assert ("fetch gdas: next, for a regional ArWen forecast: feed the mapped "
            "front door, source already bound:") in lines
    assert lines.index(
        "fetch gdas: next, for a regional ArWen forecast: feed the mapped "
        "front door, source already bound:") > 0


def test_an_analysis_no_shipped_config_reads_names_the_config_line(
        with_global):
    out = _fetched(with_global / "elsewhere")
    text = "\n".join(_lines(out))
    assert "gpuwm-global go CONFIG --outdir OUT" in text
    assert f'analysis_grib = "{(out / ANALYSIS).resolve().as_posix()}"' in text


def test_an_area_crop_cannot_start_the_global_model(with_global):
    out = _fetched(with_global / "crop", mode="nomads-cgi-subset",
                   area=[-130, 20, -60, 55])
    lines = _lines(out)
    assert not any("ArWen Global" in line for line in lines)
    assert lines[0].startswith("fetch gdas: next, for a regional ArWen forecast")


def test_without_gpuwm_global_only_the_regional_block_prints(
        tmp_path, monkeypatch):
    monkeypatch.setattr(next_doors, "installed", lambda module: False)
    out = _fetched(tmp_path / "case")
    lines = _lines(out)
    assert not any("ArWen Global" in line for line in lines)
    assert lines[0].startswith("fetch gdas: next, for a regional ArWen forecast")
