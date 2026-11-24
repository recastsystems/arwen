"""``GPUWM_BRIDGE_DIR`` moves the bridge folder, and doctor names it.

The breakage: the bridge folder was fixed at ``~/.gpuwm/bridges``, so a
second account sharing one install on a shared host found no bridges
(``gpuwm-global doctor`` exit 1, ten gaps) and had to download the same
452 MB again, with no override to read the first account's copy (gpuwm
3.0.5).
"""
from __future__ import annotations

from pathlib import Path

from gpuwm import bridges


def test_the_override_moves_the_bridge_folder_and_its_versioned_folders(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared-bridges"
    monkeypatch.setenv("GPUWM_BRIDGE_DIR", str(shared))
    assert bridges.BRIDGE_DIR_ENV == "GPUWM_BRIDGE_DIR"
    assert bridges.legacy_bridge_dir() == shared
    own = bridges.default_bridge_dir()
    assert own == shared or own.parent == shared
    assert Path.home() / ".gpuwm" not in (own, *own.parents)


def test_without_the_override_the_folder_is_the_home_one(monkeypatch):
    monkeypatch.delenv("GPUWM_BRIDGE_DIR", raising=False)
    assert bridges.legacy_bridge_dir() == Path.home() / ".gpuwm" / "bridges"


def test_a_blank_override_is_no_override(monkeypatch):
    monkeypatch.setenv("GPUWM_BRIDGE_DIR", "  ")
    assert bridges.legacy_bridge_dir() == Path.home() / ".gpuwm" / "bridges"


def test_doctor_names_the_override_whether_or_not_it_is_set(
        tmp_path, monkeypatch):
    from gpuwm import doctor

    monkeypatch.delenv("GPUWM_BRIDGE_DIR", raising=False)
    unset = bridges.staging_location_note()
    assert "GPUWM_BRIDGE_DIR" in unset
    assert "every account" in unset

    shared = tmp_path / "shared-bridges"
    monkeypatch.setenv("GPUWM_BRIDGE_DIR", str(shared))
    chosen = bridges.staging_location_note()
    assert "GPUWM_BRIDGE_DIR chose this folder" in chosen
    assert str(shared) in chosen
    # The doctor line that reports the staged estate carries the note.
    check = doctor._staged_estate_check()
    assert "GPUWM_BRIDGE_DIR" in check.detail
