"""``gpuwm fetch-geog`` stages where ``GPUWM_WPS_GEOG`` points.

The breakage (ArWen Global 0.4.0 second-account walk, 22 Nov 2026, on
gpuwm 3.0.5): on a host shared by two accounts, the second account
exported ``GPUWM_WPS_GEOG`` at the archive the first had staged, as the
shared-host page says, and its ``gpuwm fetch-geog`` ignored the variable
and staged a second copy into its own home: 2.7 GB fetched, 32 GB on
disk.  ``gpuwm mesh``, the static door and gpuwm-global's statics
already read the variable, so the stager was the one reader that did not.
"""
from __future__ import annotations

import os
import sys

import pytest

from gpuwm import cli, geog_assets


@pytest.fixture
def two_roots(tmp_path, monkeypatch):
    """A case-data root of this account's own, and a shared archive."""

    case = tmp_path / "case-data"
    shared = tmp_path / "srv" / "WPS_GEOG"
    shared.mkdir(parents=True)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(case))
    monkeypatch.setenv("GPUWM_WPS_GEOG", str(shared))
    monkeypatch.setenv("GPUWM_FETCH_LOCK_ROOT", str(tmp_path / "locks"))
    return case, shared


def test_the_variable_moves_the_default_root(two_roots):
    case, shared = two_roots
    assert geog_assets.default_geog_root() == shared


def test_unset_or_blank_keeps_the_case_data_default(two_roots, monkeypatch):
    case, _ = two_roots
    monkeypatch.setenv("GPUWM_WPS_GEOG", "  ")
    assert geog_assets.default_geog_root() == case / "WPS_GEOG"
    monkeypatch.delenv("GPUWM_WPS_GEOG")
    assert geog_assets.default_geog_root() == case / "WPS_GEOG"


def test_the_listing_names_the_shared_root_and_why(two_roots, capsys):
    _, shared = two_roots
    assert cli.main(["fetch-geog", "--list"]) == 0
    out = capsys.readouterr().out
    assert f"staging into {shared}" in out
    assert "$GPUWM_WPS_GEOG chose it" in out


def test_a_plain_fetch_stages_into_the_shared_root(two_roots, monkeypatch,
                                                   capsys):
    case, shared = two_roots
    seen = {}

    def fake_fetch_geog(*, root, **_):
        seen["root"] = root
        return 0

    monkeypatch.setattr(geog_assets, "fetch_geog", fake_fetch_geog)
    assert cli.main(["fetch-geog"]) == 0
    assert seen["root"] == shared
    assert not (case / "WPS_GEOG").exists()


def test_the_root_option_still_wins(two_roots, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_fetch_geog(*, root, **_):
        seen["root"] = root
        return 0

    monkeypatch.setattr(geog_assets, "fetch_geog", fake_fetch_geog)
    assert cli.main(["fetch-geog", "--root", str(tmp_path / "mine")]) == 0
    assert seen["root"] == tmp_path / "mine"


def test_a_complete_shared_archive_is_read_and_not_fetched(two_roots,
                                                           monkeypatch):
    """Everything valid: nothing downloads, and a repair this account
    cannot write is named rather than raised."""

    _, shared = two_roots
    monkeypatch.setattr(geog_assets, "validate_dataset_dir",
                        lambda root, name: (True, "verified"))

    def no_network(*_a, **_k):
        raise AssertionError("a complete archive must not download")

    def repair(root, *, progress):
        raise PermissionError(13, "Permission denied", str(root))

    monkeypatch.setattr(geog_assets, "repair_gmted_east_of_180", repair)
    lines: list[str] = []
    staged = geog_assets.fetch_geog(
        root=shared, datasets=(geog_assets.GMTED_DATASET,), source="hf",
        progress=lines.append, urlopen_fn=no_network)
    assert staged == 0
    assert any("east-of-180 repair is not written" in line for line in lines)
    assert any("already staged and valid" in line for line in lines)


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid")
                    or os.geteuid() == 0,
                    reason="folder modes only bind an account that is not root on POSIX")
def test_a_shared_archive_this_account_cannot_write_is_refused_with_the_remedy(
        two_roots, monkeypatch, capsys):
    _, shared = two_roots
    monkeypatch.setattr(geog_assets, "validate_dataset_dir",
                        lambda root, name: (False, "missing"))

    def no_network(*_a, **_k):
        raise AssertionError("refused before any download")

    monkeypatch.setattr(geog_assets, "_default_urlopen", no_network)
    shared.chmod(0o555)
    try:
        code = cli.main(["fetch-geog", "--datasets", geog_assets.GMTED_DATASET])
    finally:
        shared.chmod(0o755)
    assert code == 2
    err = capsys.readouterr().err
    assert "cannot write" in err and "GPUWM_WPS_GEOG" in err
    assert "Traceback" not in err


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid")
                    or os.geteuid() == 0,
                    reason="folder modes only bind an account that is not root on POSIX")
def test_a_download_folder_another_account_made_is_refused_with_the_remedy(
        two_roots, monkeypatch, capsys):
    """3.0.6 user confirmation, 24 Nov 2026: the root was group-writable, its
    .fetch-geog download folder was not, and the fetch got past the check made
    before any download, then stopped with a raw Permission denied."""

    _, shared = two_roots
    monkeypatch.setattr(geog_assets, "validate_dataset_dir",
                        lambda root, name: (False, "missing"))

    def no_network(*_a, **_k):
        raise AssertionError("refused before any download")

    monkeypatch.setattr(geog_assets, "_default_urlopen", no_network)
    downloads = shared / geog_assets.ARCHIVE_SUBDIR
    downloads.mkdir()
    downloads.chmod(0o555)
    try:
        code = cli.main(["fetch-geog", "--datasets", geog_assets.GMTED_DATASET])
    finally:
        downloads.chmod(0o755)
    assert code == 2
    err = capsys.readouterr().err
    assert "download folder" in err and "chmod g+w" in err and "GPUWM_WPS_GEOG" in err
    assert "Traceback" not in err and "os error" not in err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX folder modes")
def test_the_download_folder_takes_the_group_write_of_a_shared_root(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    downloads = shared / geog_assets.ARCHIVE_SUBDIR
    downloads.mkdir()
    downloads.chmod(0o755)
    geog_assets._share_with_root_group(downloads, shared)
    assert downloads.stat().st_mode & 0o7777 == 0o2775

    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o755)
    own = private / geog_assets.ARCHIVE_SUBDIR
    own.mkdir()
    own.chmod(0o755)
    geog_assets._share_with_root_group(own, private)
    assert own.stat().st_mode & 0o7777 == 0o755
