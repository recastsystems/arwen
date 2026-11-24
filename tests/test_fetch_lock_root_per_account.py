"""The fetch lock folder belongs to one account, and a folder that cannot
be used is refused with its remedy.

The breakage: on a Linux host shared by two accounts, every ``gpuwm
fetch`` of the second account stopped with a ``PermissionError``
traceback on ``/tmp/gpuwm/locks/...``, a folder the first account had
made mode 755, and nothing named ``GPUWM_FETCH_LOCK_ROOT`` (gpuwm 3.0.5;
seen twice in Nov 2026).  The two-account run itself is in the 3.0.6 fix
page; these hold the mechanism.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import pytest

from gpuwm import fetch_guard

POSIX = os.name != "nt"


@pytest.fixture(autouse=True)
def clean_registry():
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()
    yield
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()


@pytest.fixture
def default_root(tmp_path, monkeypatch):
    """The default lock folder, with the temp folder moved into tmp_path."""

    monkeypatch.delenv(fetch_guard.LOCK_ROOT_ENV, raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "programdata"))
    return tmp_path


def test_the_default_lock_folder_is_this_accounts_own(default_root):
    root = fetch_guard.lock_root()
    if POSIX:
        assert root == default_root / f"gpuwm-{os.getuid()}" / "locks"
    else:
        assert root == default_root / "localappdata" / "gpuwm" / "locks"
    # Not the folder every account shared until 3.0.5.
    assert root != default_root / "gpuwm" / "locks"
    assert root != default_root / "programdata" / "gpuwm" / "locks"


@pytest.mark.skipif(not POSIX, reason="POSIX folder modes")
def test_the_default_lock_folder_is_made_owner_only(default_root, tmp_path):
    with fetch_guard.hold("fetch-out", tmp_path / "out", progress=lambda *_: None):
        pass
    parent = default_root / f"gpuwm-{os.getuid()}"
    assert (parent / "locks").is_dir()
    assert parent.stat().st_mode & 0o077 == 0
    assert (parent / "locks").stat().st_mode & 0o077 == 0


@pytest.mark.skipif(not POSIX, reason="POSIX account ids")
def test_a_per_account_folder_made_by_another_account_is_refused(
        default_root, tmp_path, monkeypatch):
    real = os.getuid()
    # The folder is this process's; pretend this process is another account.
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    root = default_root / f"gpuwm-{real + 1}" / "locks"
    (default_root / f"gpuwm-{real + 1}").mkdir()
    with pytest.raises(fetch_guard.FetchLockRootRefused) as refused:
        fetch_guard.ensure_lock_root(root)
    text = str(refused.value)
    assert "belongs to another account" in text
    assert fetch_guard.LOCK_ROOT_ENV in text


@pytest.mark.skipif(not POSIX or os.geteuid() == 0,
                    reason="POSIX modes, and root can write anything")
def test_a_lock_folder_this_account_cannot_write_is_refused_by_name(
        tmp_path, monkeypatch):
    shared = tmp_path / "theirs"
    shared.mkdir()
    shared.chmod(0o555)
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(shared / "locks"))
    try:
        with pytest.raises(fetch_guard.FetchLockRootRefused) as refused:
            with fetch_guard.hold("fetch-out", tmp_path / "out",
                                  progress=lambda *_: None):
                pass
    finally:
        shared.chmod(0o755)
    text = str(refused.value)
    assert str(shared / "locks") in text
    assert f"{fetch_guard.LOCK_ROOT_ENV}=" in text
    # A refusal, not the bare OS error the CLI printed as a traceback.
    assert isinstance(refused.value, RuntimeError)


@pytest.mark.skipif(not POSIX or os.geteuid() == 0,
                    reason="POSIX modes, and root can write anything")
def test_a_lock_file_another_account_left_unwritable_is_refused_by_name(
        tmp_path, monkeypatch):
    root = tmp_path / "group-locks"
    root.mkdir()
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(root))
    path = fetch_guard.lock_path("fetch-out", tmp_path / "out")
    path.write_bytes(b"\0")
    path.chmod(0o444)
    try:
        with pytest.raises(fetch_guard.FetchLockRootRefused) as refused:
            with fetch_guard.hold("fetch-out", tmp_path / "out",
                                  progress=lambda *_: None):
                pass
    finally:
        path.chmod(0o644)
    text = str(refused.value)
    assert path.name in text
    # The remedy keeps the shared lock folder: abandoning it would let the
    # two accounts' fetches interleave in the shared output folder.
    assert f"chmod a+rw {path}" in text
    assert f"Keep {fetch_guard.LOCK_ROOT_ENV} as it is" in text
    assert "export GPUWM_FETCH_LOCK_ROOT=~/.cache" not in text


@pytest.mark.skipif(not POSIX, reason="POSIX file modes")
def test_a_lock_file_made_in_a_shared_lock_folder_is_writable_by_every_account(
        tmp_path, monkeypatch):
    root = tmp_path / "group-locks"
    root.mkdir()
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(root))
    previous = os.umask(0o022)
    try:
        with fetch_guard.hold("fetch-out", tmp_path / "out",
                              progress=lambda *_: None):
            pass
    finally:
        os.umask(previous)
    path = fetch_guard.lock_path("fetch-out", tmp_path / "out")
    assert path.stat().st_mode & 0o666 == 0o666


@pytest.mark.skipif(not POSIX, reason="POSIX file modes")
def test_the_default_lock_folder_keeps_the_umask(default_root, tmp_path):
    previous = os.umask(0o022)
    try:
        with fetch_guard.hold("fetch-out", tmp_path / "out",
                              progress=lambda *_: None):
            pass
    finally:
        os.umask(previous)
    path = fetch_guard.lock_path("fetch-out", tmp_path / "out")
    assert path.stat().st_mode & 0o022 == 0


@pytest.mark.skipif(not POSIX or os.geteuid() == 0,
                    reason="POSIX modes, and root can write anything")
def test_an_output_folder_this_account_cannot_write_is_refused_first(
        tmp_path):
    theirs = tmp_path / "shared" / "gfs-latest"
    theirs.mkdir(parents=True)
    theirs.chmod(0o555)
    try:
        with pytest.raises(fetch_guard.OutputFolderRefused) as refused:
            fetch_guard.refuse_unwritable_output(theirs)
        # A folder not made yet is judged by the folder it would be made in.
        with pytest.raises(fetch_guard.OutputFolderRefused):
            fetch_guard.refuse_unwritable_output(theirs / "deeper")
    finally:
        theirs.chmod(0o755)
    text = str(refused.value)
    assert str(theirs) in text
    assert "umask 002" in text and "g+s" in text
    assert isinstance(refused.value, RuntimeError)
    fetch_guard.refuse_unwritable_output(theirs)
    fetch_guard.refuse_unwritable_output(tmp_path / "mine" / "new")


@pytest.mark.skipif(not POSIX or os.geteuid() == 0,
                    reason="POSIX modes, and root can write anything")
def test_the_fetch_refuses_an_unwritable_output_before_its_lock(tmp_path):
    from gpuwm import fetch

    theirs = tmp_path / "theirs"
    theirs.mkdir()
    theirs.chmod(0o555)
    try:
        with pytest.raises(fetch_guard.OutputFolderRefused):
            with fetch._fetch_out(theirs, progress=lambda *_: None):
                raise AssertionError("the fetch went ahead")
    finally:
        theirs.chmod(0o755)


def test_two_processes_of_one_account_still_exclude_each_other(
        default_root, tmp_path):
    target = tmp_path / "out"
    env = dict(os.environ)
    env.pop(fetch_guard.LOCK_ROOT_ENV, None)
    env.update({"TMPDIR": str(default_root), "TEMP": str(default_root),
                "TMP": str(default_root),
                "LOCALAPPDATA": str(default_root / "localappdata"),
                fetch_guard.LOCK_TIMEOUT_ENV: "0"})
    child = textwrap.dedent(f"""
        import sys
        from gpuwm import fetch_guard
        try:
            with fetch_guard.hold("fetch-out", {str(target)!r},
                                  progress=lambda *_: None):
                print("ACQUIRED")
        except fetch_guard.FetchLockBusy:
            print("BUSY")
    """)
    repo = str(Path(fetch_guard.__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    with fetch_guard.hold("fetch-out", target, progress=lambda *_: None):
        busy = subprocess.run([sys.executable, "-c", child], env=env,
                              capture_output=True, text=True, timeout=120)
    free = subprocess.run([sys.executable, "-c", child], env=env,
                          capture_output=True, text=True, timeout=120)
    assert busy.stdout.strip() == "BUSY", busy.stderr
    assert free.stdout.strip() == "ACQUIRED", free.stderr


# --- Another account's file in a sticky shared folder (fs.protected_regular) ---
#
# The breakage: Linux refuses an O_CREAT open of a file another account owns
# in a sticky world-writable folder unless the folder's owner made it,
# whatever the file's mode (fs.protected_regular, on by default on Ubuntu).
# Two accounts fetching at once on one host: the second died with
# "[Errno 13] Permission denied: '/tmp/gpuwm-fetch-connections/slot-1'" on a
# 0666 slot the first had made (3.0.6 candidate, 24 Nov 2026); the card lock
# folder is such a folder too.  One account cannot own two uids here, so the
# kernel's answer is stood in for: an O_CREAT open (without O_EXCL) of a file
# that exists is refused, as the kernel refuses another account.

from gpuwm import fetch_connections, shared_files  # noqa: E402


class _ProtectedRegular:
    """``os`` as the kernel answers another account in a sticky folder."""

    def __init__(self, refuse=()):
        self.refuse = {str(path) for path in refuse}
        self.opens = []

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, path, flags, mode=0o777):
        self.opens.append((str(path), flags))
        if str(path) in self.refuse:
            raise PermissionError(13, "Permission denied", str(path))
        if flags & os.O_CREAT and not flags & os.O_EXCL and os.path.exists(path):
            raise PermissionError(13, "Permission denied", str(path))
        return os.open(path, flags, mode)


@pytest.fixture
def protected(monkeypatch):
    def install(refuse=()):
        fake = _ProtectedRegular(refuse)
        monkeypatch.setattr(shared_files, "os", fake)
        monkeypatch.setattr(shared_files.time, "sleep", lambda _s: None)
        return fake
    return install


def test_another_accounts_file_is_opened_without_o_creat(protected, tmp_path):
    path = tmp_path / "slot-1"
    path.write_bytes(b"")
    fake = protected()
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    with pytest.raises(PermissionError):
        fake.open(path, flags, 0o666)  # what the open before 3.0.6's fix met
    os.close(shared_files.open_shared(path, flags, 0o666))
    assert not fake.opens[-1][1] & os.O_CREAT


def test_a_missing_file_is_made_exclusively(protected, tmp_path):
    path = tmp_path / "lock"
    fake = protected()
    os.close(shared_files.open_shared(path, os.O_RDWR | os.O_CREAT, 0o666))
    assert path.exists()
    assert fake.opens[-1][1] & os.O_CREAT and fake.opens[-1][1] & os.O_EXCL


def test_a_file_made_between_the_two_opens_is_opened_as_it_is(monkeypatch, tmp_path):
    path = tmp_path / "lock"
    seen = []

    class Racing(_ProtectedRegular):
        def open(self, p, flags, mode=0o777):
            seen.append(flags)
            if len(seen) == 1:
                raise FileNotFoundError(2, "No such file", str(p))
            if flags & os.O_EXCL and len(seen) == 2:
                path.write_bytes(b"")  # the other process wins the create
                raise FileExistsError(17, "File exists", str(p))
            return super().open(p, flags, mode)

    monkeypatch.setattr(shared_files, "os", Racing())
    os.close(shared_files.open_shared(path, os.O_RDWR | os.O_CREAT, 0o666))
    assert len(seen) == 3 and not seen[-1] & os.O_CREAT


def test_a_file_that_stays_closed_is_refused_after_brief_retries(protected, tmp_path):
    path = tmp_path / "lock"
    path.write_bytes(b"")
    fake = protected(refuse=[path])
    with pytest.raises(PermissionError):
        shared_files.open_shared(path, os.O_RDWR | os.O_CREAT, 0o666)
    assert len(fake.opens) == shared_files.PERMISSION_RETRIES + 1


def test_the_lock_and_slot_opens_take_another_accounts_file(protected, tmp_path, monkeypatch):
    monkeypatch.delenv(fetch_guard.LOCK_ROOT_ENV, raising=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    if POSIX:
        shared.chmod(0o1777)
    for name in ("fetch-out-x.lock", "slot-1", "gpu-x.lock"):
        (shared / name).write_bytes(b"\0")
        if POSIX:
            (shared / name).chmod(0o666)
    protected()
    fetch_guard._open_lock_file(shared / "fetch-out-x.lock").close()
    fetch_connections._open_slot(shared / "slot-1").close()
    if POSIX:
        from gpuwm import supervisor
        supervisor._open_shared_lock(shared / "gpu-x.lock").close()


def test_a_closed_slot_past_the_first_sends_the_fetch_to_its_own_slots(protected, tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "6")
    directory = tmp_path / "gpuwm-fetch-connections"
    directory.mkdir()
    (directory / "slot-3").write_bytes(b"")
    protected(refuse=[directory / "slot-3"])
    chosen = fetch_connections._usable_slot_directory(directory)
    assert chosen != directory and chosen.parent == directory.parent
    assert chosen.name.startswith("gpuwm-fetch-connections-")
