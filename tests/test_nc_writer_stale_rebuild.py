"""A checkout whose netcdf-writer predates compressed frames rebuilds it, and
says why when it cannot.  The rebuild itself is exercised for real by
test_native_level_output on a machine whose shared copy is stale."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpuwm.io import nc_writer_bridge as bridge


@pytest.fixture(autouse=True)
def _fresh_outcome(monkeypatch):
    monkeypatch.setattr(bridge, "_REBUILD_OUTCOME", None)


def _no_cargo_run(*args, **kwargs):
    raise AssertionError("cargo must not run here")


def test_an_explicit_override_is_never_rebuilt(monkeypatch):
    if bridge.unavailable_reason() is not None:
        pytest.skip("no netcdf-writer library on this host")
    monkeypatch.setenv(bridge.NCWRITE_BRIDGE_ENV, str(bridge.resolve_ncwrite_bridge()))
    monkeypatch.setattr(subprocess, "run", _no_cargo_run)
    assert bridge._refresh_stale_checkout_library() == (
        "it was named by " + bridge.NCWRITE_BRIDGE_ENV)


def test_no_cargo_is_named_as_the_reason(monkeypatch):
    monkeypatch.setattr(bridge, "_cargo", lambda: None)
    monkeypatch.setattr(subprocess, "run", _no_cargo_run)
    why = bridge._rebuild_checkout_library(Path("stale.so"))
    assert why in ("no cargo was found on PATH or in ~/.cargo/bin",
                   "this install has no tools/rustwx checkout to build from")
    # One attempt per process: the answer is remembered.
    assert bridge._rebuild_checkout_library(Path("stale.so")) == why


def test_a_failed_build_is_reported_with_its_own_words(monkeypatch, tmp_path):
    from gpuwm import rustwx

    workspace = tmp_path / "rustwx"
    (workspace / "crates" / "netcdf-writer").mkdir(parents=True)
    (workspace / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    monkeypatch.setattr(rustwx, "crate_dir", lambda: workspace)
    monkeypatch.setattr(bridge, "_cargo", lambda: "cargo")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 101, stdout="", stderr="error: could not compile `netcdf-writer`"))
    why = bridge._rebuild_checkout_library(Path("stale.so"))
    assert "failed (exit 101)" in why and "could not compile" in why
