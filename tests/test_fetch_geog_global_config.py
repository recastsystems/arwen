"""``gpuwm fetch-geog --config`` takes an ArWen Global experiment.

The breakage: ``gpuwm fetch-geog --config
arwen_global_gdas_t255_native_sl_si_24h --list`` exited 2 with "no such
file or folder", so the geography download could not be scoped to a
global config (gpuwm 3.0.5, beside gpuwm-global 0.3.0).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from gpuwm import cli, geog_assets, next_doors

NAME = "arwen_global_gdas_t255_native_sl_si_24h"
GLOBAL_TOML = """\
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "t255"
[initial]
mode = "analysis"
analysis_grib = "cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000"
[physics]
mode = "arwen-native"
native_adapter_options = { radiation = "rrtmgp", orographic_drag = "kim-gwdo" }
"""


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """gpuwm-global installed, shipping NAME."""

    root = tmp_path / "shipped"
    root.mkdir()
    (root / f"{NAME}.toml").write_text(GLOBAL_TOML, encoding="utf-8")
    monkeypatch.setattr(next_doors, "shipped_configs_folder",
                        lambda module=next_doors.GLOBAL_MODULE: root)
    monkeypatch.chdir(tmp_path)
    return root


def test_a_shipped_name_is_found_in_the_installed_package_without_importing_it(
        tmp_path, monkeypatch):
    """The package's folder is read as data; its code never runs."""

    site = tmp_path / "site"
    package = site / "arwen_global"
    (package / "configs").mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise RuntimeError("the engine imported gpuwm-global")\n',
        encoding="utf-8")
    (package / "configs" / f"{NAME}.toml").write_text(
        GLOBAL_TOML, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "arwen_global", raising=False)
    monkeypatch.chdir(tmp_path)
    found = geog_assets.resolve_run_config(NAME)
    assert found == package / "configs" / f"{NAME}.toml"
    assert geog_assets.resolve_run_config(f"{NAME}.toml") == found
    # A path with a folder is the caller's own file, never the package's.
    assert geog_assets.resolve_run_config(f"configs/{NAME}") == Path(
        f"configs/{NAME}")
    assert "arwen_global" not in sys.modules


def _listed(capsys, *argv) -> tuple[int, dict]:
    code = cli.main(["fetch-geog", *argv, "--list"])
    out = capsys.readouterr().out
    start = out.find("{")
    return code, (json.loads(out[start:]) if start >= 0 else {})


def test_a_shipped_global_name_lists_its_geography(shipped, capsys):
    code, plan = _listed(capsys, "--config", NAME)
    assert code == 0
    assert plan["model"] == "ArWen Global"
    # The nine land datasets the global statics read, and the sub-grid
    # orography the drag reads.
    assert "orogwd_10m" in plan["datasets"]
    assert "topo_gmted2010_30s" in plan["datasets"]
    assert len(plan["datasets"]) >= 9
    assert plan["archive_bytes"] > 0


def test_a_global_config_by_path_lists_the_same_geography(
        shipped, tmp_path, capsys):
    path = tmp_path / "copy.toml"
    path.write_text(GLOBAL_TOML, encoding="utf-8")
    code, plan = _listed(capsys, "--config", str(path))
    assert code == 0
    assert "orogwd_10m" in plan["datasets"]


def test_without_drag_the_orography_dataset_is_not_asked_for(
        shipped, tmp_path, capsys):
    path = tmp_path / "nodrag.toml"
    path.write_text(GLOBAL_TOML.replace('"kim-gwdo"', '"none"'),
                    encoding="utf-8")
    code, plan = _listed(capsys, "--config", str(path))
    assert code == 0
    assert "orogwd_10m" not in plan["datasets"]


def test_synthetic_statics_need_no_geography(shipped, tmp_path, capsys):
    path = tmp_path / "smoke.toml"
    path.write_text('[arwen_global]\nname = "smoke"\n'
                    '[initial]\nmode = "analytic"\n', encoding="utf-8")
    code, plan = _listed(capsys, "--config", str(path))
    assert code == 0
    assert plan["datasets"] == []


def test_a_name_nothing_resolves_is_still_refused_by_name(shipped, capsys):
    code = cli.main(["fetch-geog", "--config", "no_such_config", "--list"])
    assert code == 2
    assert "no_such_config" in capsys.readouterr().err


def test_the_geography_root_is_the_one_global_statics_read(
        shipped, tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_WPS_GEOG", raising=False)
    assert geog_assets.global_config_geog_root({}) == geog_assets.default_geog_root()
    assert geog_assets.global_config_geog_root(
        {"statics": {"geog_root": str(tmp_path / "g")}}) == tmp_path / "g"
