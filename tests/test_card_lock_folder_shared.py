"""The machine card lock folder is one every account can lock in.

The breakage: card locks keep runs of any account off one card, so they
live in one folder per machine (``<tmp>/gpuwm/locks``).  Up to 3.0.5 the
first account to make that folder (a fetch or a run) made it mode 755,
and a second account's card lock then stopped with a ``PermissionError``
traceback (reproduced with two accounts in a container, Nov 2026).
"""
from __future__ import annotations

import os
import stat
import tempfile

import pytest

from gpuwm import supervisor

pytestmark = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX folder modes, as an account that is not root")


@pytest.fixture
def machine_root(tmp_path, monkeypatch):
    monkeypatch.delenv(supervisor.GPU_LOCK_ROOT_ENV, raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path / "gpuwm" / "locks"


def test_the_machine_folder_and_its_lock_files_are_open_to_every_account(
        machine_root):
    lock = supervisor.GPUFileLock("GPU-demo", run_id="t").acquire()
    try:
        assert lock.path.parent == machine_root
        assert stat.S_IMODE(machine_root.stat().st_mode) == 0o1777
        assert stat.S_IMODE(lock.path.stat().st_mode) == 0o666
    finally:
        lock.release()


def test_a_folder_another_account_made_closed_is_refused_with_the_remedy(
        machine_root, monkeypatch):
    machine_root.mkdir(parents=True)
    machine_root.chmod(0o755)
    real = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    machine_root.chmod(0o555)
    try:
        with pytest.raises(supervisor.GPULockFolderRefused) as refused:
            supervisor.GPUFileLock("GPU-demo", run_id="t").acquire()
    finally:
        machine_root.chmod(0o755)
    text = str(refused.value)
    assert f"chmod 1777 {machine_root}" in text
    assert supervisor.GPU_LOCK_ROOT_ENV in text


def test_a_configured_folder_keeps_the_modes_its_owner_chose(
        tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    mine.mkdir(mode=0o750)
    mine.chmod(0o750)
    monkeypatch.setenv(supervisor.GPU_LOCK_ROOT_ENV, str(mine))
    lock = supervisor.GPUFileLock("GPU-demo", run_id="t").acquire()
    lock.release()
    assert stat.S_IMODE(mine.stat().st_mode) == 0o750


def test_a_link_planted_in_the_shared_folder_is_not_followed(
        machine_root, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    machine_root.mkdir(parents=True)
    machine_root.chmod(0o1777)
    planted = supervisor.default_lock_path("GPU-demo")
    planted.symlink_to(victim)
    with pytest.raises(OSError):
        supervisor.GPUFileLock("GPU-demo", run_id="t").acquire()
    assert victim.read_text(encoding="utf-8") == "keep me"
