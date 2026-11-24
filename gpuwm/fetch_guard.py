"""Single-writer discipline and atomic publication for the fetch layer.

Every ``gpuwm fetch*`` flow writes into a directory that a later run --
or a *concurrent* run -- reads back as authority.  Two properties have
to hold for that to be safe, and neither is free:

**One writer at a time.**  The request-identity guard, the mutation of
the payloads, and the publication of the receipt that blesses them are
three separate steps.  Interleave two processes across them and the
final receipt can describe the other process's bytes.  Every mutating
flow therefore takes an exclusive, OS-enforced lock on its output root
first: a Windows byte-range lock / POSIX ``flock`` on a lock file kept
outside the output tree (so it never lands in a fetched directory and
never confuses the nonempty-output guard).  The kernel releases it when
the holder dies, which a cooperative sentinel file cannot promise --
a crashed fetch must not leave a directory permanently unfetchable.
The loser announces the wait, waits, and then refuses loudly naming the
holder; it never proceeds in parallel and never silently doubles work.
A caller that can see the holder's work (the bytes a download has
staged) passes it as ``holder_progress``, and the loser then waits as
long as that work keeps moving and refuses only a holder that stalls.

**Nothing is published half-written.**  Text receipts are written to a
temp that is unique per call -- a fixed ``.tmp`` is exactly the file two
writers collide on -- claimed exclusively under a compact random name
that never repeats the target's own (so a folder deep enough to hold a
file is deep enough to stage it), flushed, fsynced, atomically renamed,
and the containing directory is fsynced too where the platform allows
it.  Quarantine names are proven absent before the rename, so
moving evidence aside can never overwrite older evidence.

**A write this computer refuses is not a download that failed.**  A
folder too deep for Windows or a full disk used to surface through the
network handler as "download failed ...; a re-run resumes".
:func:`receive` and :class:`LocalWriteFailed` keep the two apart, and
:func:`local_write_refusal` names the path's length and the fix.

Nothing here deletes anything.  Quarantine moves aside; the lock file
is the only file this module creates on its own, and it lives in the
per-user lock root, not in the fetched output.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time

from gpuwm.shared_files import open_shared
from typing import Any, Callable

from gpuwm.explain import layered

#: How long a losing writer waits for the holder before refusing.  With a
#: ``holder_progress`` probe it is how long the holder may go without
#: showing progress, not how long its whole job may take.
DEFAULT_LOCK_TIMEOUT_S = 600.0

#: Override for the wait budget (seconds).  A test or a batch driver
#: that would rather fail fast than queue sets this.
LOCK_TIMEOUT_ENV = "GPUWM_FETCH_LOCK_TIMEOUT_S"

#: Override for the lock root: a folder this account can write.  Set it
#: to one folder that every account writing the same output folders can
#: write (a group folder) when two accounts fetch into one shared folder
#: at once; the default is a folder of this account's own.
LOCK_ROOT_ENV = "GPUWM_FETCH_LOCK_ROOT"

_POLL_S = 0.25

#: Re-entrancy and in-process exclusion.  The same *thread* may nest
#: ``hold()`` for one target -- the CLI takes it around the request
#: guard and the library function takes it again around the transfer --
#: and a second OS lock on one file from one process would deadlock
#: against itself on Windows, so nesting is counted rather than
#: re-locked.  A *different* thread is a different writer and has to
#: queue: the per-key ``RLock`` gives exactly that pair of behaviours,
#: and the OS lock underneath it excludes other processes.
_KEY_LOCKS: dict[str, threading.RLock] = {}
_HELD: dict[str, "_Entry"] = {}
_REGISTRY_GUARD = threading.Lock()


def _after_fork() -> None:
    """A child must acquire its own OS lock instead of inheriting re-entry."""
    global _KEY_LOCKS, _HELD, _REGISTRY_GUARD
    # Close inherited descriptors without LOCK_UN: flock attaches to the
    # shared open-file description and unlocking it would release the
    # parent's lock too. The parent's descriptor remains open.
    for entry in _HELD.values():
        entry.stream.close()
    _KEY_LOCKS = {}
    _HELD = {}
    _REGISTRY_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class FetchLockBusy(RuntimeError):
    """Another writer holds the output root; this one refuses.

    ``budget_s`` is the wait budget, ``waited_s`` how long this writer
    waited in all, and ``idle_s``, set only when the wait watched the
    holder's progress, how long the holder had shown none.  ``holder`` is
    what the holder recorded about itself.
    """

    def __init__(self, message: str = "", *, budget_s: float | None = None,
                 waited_s: float | None = None, idle_s: float | None = None,
                 holder: str = "") -> None:
        super().__init__(message)
        self.budget_s = budget_s
        self.waited_s = waited_s
        self.idle_s = idle_s
        self.holder = holder


class FetchLockRootRefused(RuntimeError):
    """The lock folder cannot be used by this account; the fetch refuses.

    A refusal with its remedy rather than a ``PermissionError``
    traceback.  The breakage it names: on a Linux host shared by two
    accounts, every ``gpuwm fetch`` of the second account stopped with
    ``PermissionError`` on ``/tmp/gpuwm/locks/...``, a folder the first
    account had created mode 755 (gpuwm 3.0.5, seen twice in Nov 2026).
    ``root`` is the folder that was refused.
    """

    def __init__(self, message: str = "", *, root: Path | None = None) -> None:
        super().__init__(message)
        self.root = root


class _Patience:
    """How much longer a losing writer waits.

    Without a probe the budget runs from the start of the wait.  With a
    ``holder_progress`` probe it runs from the last time what the probe
    returns changed, so a holder whose work is still moving is waited
    for however long it takes, and only one that has stalled for the
    whole budget is refused.
    """

    def __init__(self, budget_s: float, clock,
                 probe: Callable[[], object] | None) -> None:
        self.budget_s = budget_s
        self.probe = probe
        self._clock = clock
        self.started = clock()
        self._last_change = self.started
        self._mark = probe() if probe is not None else None
        self._now = self.started

    def expired(self) -> bool:
        self._now = self._clock()
        if self.probe is not None:
            mark = self.probe()
            if mark != self._mark:
                self._mark = mark
                self._last_change = self._now
        return self._now - self._last_change >= self.budget_s

    @property
    def waited_s(self) -> float:
        return self._now - self.started

    @property
    def idle_s(self) -> float:
        return self._now - self._last_change


class _Entry:
    __slots__ = ("stream", "depth")

    def __init__(self, stream) -> None:
        self.stream = stream
        self.depth = 0


def _key_lock(key: str) -> threading.RLock:
    with _REGISTRY_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _KEY_LOCKS[key] = lock
        return lock


def lock_root() -> Path:
    """Where lock files live: outside every fetched output tree.

    One folder per account.  ``GPUWM_FETCH_LOCK_ROOT`` wins when set.
    Otherwise POSIX uses ``<tmp>/gpuwm-<uid>/locks`` and Windows
    ``%LOCALAPPDATA%\\gpuwm\\locks``.  Until 3.0.5 every account on a
    host shared ``<tmp>/gpuwm/locks`` (``%PROGRAMDATA%`` on Windows).
    The first account created it with its own umask (755), and every
    other account's fetch then stopped with ``PermissionError``.  A
    folder per account cannot be taken by another one, and
    :func:`ensure_lock_root` refuses by name a folder that another
    account made first.

    The lock keeps two writers of ONE account out of one output folder.
    Two accounts writing the same shared folder at the same moment
    exclude each other only when they set ``GPUWM_FETCH_LOCK_ROOT`` to
    one folder that both can write.
    """

    override = os.environ.get(LOCK_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return Path(base) / "gpuwm" / "locks"
    return Path(tempfile.gettempdir()) / f"gpuwm-{os.getuid()}" / "locks"


def _lock_root_remedy(root: Path, problem: str) -> str:
    example = ("%LOCALAPPDATA%\\gpuwm\\locks" if os.name == "nt"
               else "~/.cache/gpuwm/locks")
    setter = (f"set {LOCK_ROOT_ENV}={example}" if os.name == "nt"
              else f"export {LOCK_ROOT_ENV}={example}")
    return layered(
        f"the fetch lock folder {root} {problem}, so this fetch cannot "
        "take its lock.\n"
        f"  remedy: point {LOCK_ROOT_ENV} at a folder this account can "
        f"write, then run the same command again:\n"
        f"    {setter}",
        "  why: a fetch takes a lock on its output folder before it "
        "writes, so two runs cannot interleave their files and their "
        "receipt.  The lock file lives outside the fetched folder, in "
        f"this account's lock folder.  {LOCK_ROOT_ENV} moves it; set it "
        "to one folder that every account can write when two accounts "
        "fetch into the same shared folder at the same time.")


def ensure_lock_root(root: Path | None = None) -> Path:
    """Make the lock folder (owner only, on POSIX) or refuse by name.

    Returns the folder.  Raises :class:`FetchLockRootRefused` when the
    folder cannot be made or written, and, for the default per-account
    folder on POSIX, when it is a link or belongs to another account
    (a folder of that name made first by someone else would let them
    hold or swap this account's lock files).  An override folder is the
    user's choice and only has to be writable.
    """

    root = lock_root() if root is None else Path(root)
    default = not os.environ.get(LOCK_ROOT_ENV)
    try:
        if default and os.name != "nt":
            # The per-account parent (<tmp>/gpuwm-<uid>) is made mode 700
            # so only this account can create or replace files in it.
            parent = root.parent
            parent.mkdir(mode=0o700, exist_ok=True)
            info = os.lstat(parent)
            import stat as _stat
            if _stat.S_ISLNK(info.st_mode) or not _stat.S_ISDIR(info.st_mode):
                raise FetchLockRootRefused(_lock_root_remedy(
                    parent, "is not a plain folder (a link or a file "
                    "stands there)"), root=parent)
            if info.st_uid != os.getuid():
                raise FetchLockRootRefused(_lock_root_remedy(
                    parent, f"belongs to another account (uid "
                    f"{info.st_uid}, this one is {os.getuid()})"),
                    root=parent)
            root.mkdir(mode=0o700, exist_ok=True)
        else:
            root.mkdir(parents=True, exist_ok=True)
    except FetchLockRootRefused:
        raise
    except PermissionError as error:
        raise FetchLockRootRefused(_lock_root_remedy(
            root, f"cannot be created by this account ({error.strerror})"),
            root=root) from None
    except FileExistsError:
        raise FetchLockRootRefused(_lock_root_remedy(
            root, "is a file, not a folder"), root=root) from None
    if not os.access(root, os.W_OK | os.X_OK):
        raise FetchLockRootRefused(_lock_root_remedy(
            root, "is not writable by this account"), root=root)
    return root


def _open_lock_file(path: Path):
    """Open, creating it if needed, the lock file at ``path``.

    In a lock folder named by ``GPUWM_FETCH_LOCK_ROOT`` on POSIX, a lock
    file this account creates is made writable by every account (0666),
    whatever the umask.  The breakage: with the variable set to one folder
    for two accounts, as the shared-host section of DATA.md says, the
    first account's fetch left its lock file at 0644, and every later
    fetch of the other account into the same output folder was refused
    at once instead of waiting (3.0.6 candidate, two accounts on one
    host, 23 Nov 2026).  The exclusion is the lock taken on the open
    file, not the file's mode, so the mode gives nothing away but the
    holder record (process id, folder, time) the file carries.
    """

    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    # Another account's lock file in a sticky shared folder is opened without O_CREAT, which Linux
    # would refuse whatever the file's mode (shared_files.open_shared).
    descriptor = open_shared(path, flags, 0o666)
    try:
        if os.name != "nt" and os.environ.get(LOCK_ROOT_ENV):
            info = os.fstat(descriptor)
            if (info.st_uid == os.getuid()
                    and info.st_mode & 0o666 != 0o666):
                try:
                    os.fchmod(descriptor, 0o666)
                except OSError:
                    pass  # the lock still works for this account
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _lock_file_remedy(path: Path, error: OSError) -> str:
    try:
        owner = f"account uid {path.stat().st_uid}"
    except (OSError, AttributeError):
        owner = "another account"
    change = (f"chmod a+rw {path}" if os.name != "nt"
              else f'icacls "{path}" /grant Users:M')
    return layered(
        f"the fetch lock folder {path.parent} holds {path.name}, which "
        f"this account cannot open for writing ({error.strerror}), so this "
        f"fetch cannot take its lock.  {owner.capitalize()} made it without "
        "write for others (gpuwm before 3.0.6 made lock files so).\n"
        "  remedy: the account that made it runs, once:\n"
        f"    {change}\n"
        "  or removes the file while no fetch is running; then run the "
        f"same command again.  Keep {LOCK_ROOT_ENV} as it is: one lock "
        "folder for both accounts is what makes each wait for the other.",
        "  why: a fetch takes a lock on its output folder before it writes, "
        "so two runs cannot interleave their files and their receipt.  "
        "This gpuwm makes every lock file it creates in a folder named by "
        f"{LOCK_ROOT_ENV} writable by every account.")


class OutputFolderRefused(RuntimeError):
    """The fetch's output folder cannot be written by this account.

    A refusal with its remedy, taken before the lock and before any
    download.  The breakage: a second account's ``gpuwm fetch --out``
    into a folder the first account had made (mode 755) downloaded, then
    ended in a ``PermissionError`` traceback from
    :func:`atomic_write_bytes` (3.0.5, and the 3.0.6 candidate, two
    accounts on one host, 23 Nov 2026).
    """

    def __init__(self, message: str = "", *, folder: Path | None = None) -> None:
        super().__init__(message)
        self.folder = folder


def refuse_unwritable_output(out: str | Path) -> None:
    """Raise :class:`OutputFolderRefused` when this account cannot write
    ``out`` (or, when it does not exist yet, the folder it would be made
    in).  POSIX only: on Windows a folder's read-only flag does not stop
    writes into it, and its access lists are left to the write itself.
    """

    if os.name == "nt":
        return
    folder = Path(out).expanduser()
    probe = folder
    while not probe.exists():
        if probe.parent == probe:
            return
        probe = probe.parent
    if not probe.is_dir() or os.access(probe, os.W_OK | os.X_OK):
        return
    try:
        owner = f"account uid {probe.stat().st_uid}"
    except OSError:
        owner = "another account"
    raise OutputFolderRefused(layered(
        f"this account cannot write {probe} (it belongs to {owner}), so "
        f"this fetch cannot write {folder}; nothing was downloaded.\n"
        "  remedy: fetch into a folder of this account's own (--out), or "
        "let the accounts share the folder through one group: the owner "
        "runs `chgrp -R GROUP FOLDER && chmod -R g+rwX FOLDER && "
        "find FOLDER -type d -exec chmod g+s {} +`, and every account "
        "fetches with `umask 002` (DATA.md, \"A host shared by several "
        "accounts\").",
        "  why: a fetch writes its files and its receipt into the output "
        "folder; a folder this account cannot write would fail only after "
        "the download."), folder=probe)


def lock_path(kind: str, target: str | Path) -> Path:
    """The lock file for ``kind`` over ``target``.

    Keyed by the *resolved* path, so ``--out .\\run`` and an absolute
    spelling of the same directory take the same lock, and a junction
    or symlink cannot split one directory into two writers.  The
    directory need not exist yet: ``Path.resolve()`` is non-strict.
    """

    resolved = Path(target).expanduser().resolve()
    key = str(resolved)
    if os.name == "nt":
        key = key.lower()  # NTFS is case-insensitive; the key must be too
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return lock_root() / f"{kind}-{digest}.lock"


def _is_contention(error: OSError) -> bool:
    """A held lock, as opposed to a disk, ACL, or descriptor failure."""

    if os.name == "nt":
        return error.errno in {errno.EACCES, errno.EDEADLK}
    return error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}


def _try_lock(stream) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if _is_contention(error):
            return False
        raise
    return True


def _unlock(stream) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Closing the stream releases the lock either way; a failure to
        # unlock explicitly must not mask the caller's own exception.
        pass


def describe_holder(path: Path) -> str:
    """Whatever the current holder recorded about itself, for refusals.

    The owner record sits past the locked byte, so it stays readable
    while the lock is held.  An unreadable or empty record is not an
    error -- the refusal simply says less.
    """

    try:
        with path.open("rb") as stream:
            stream.seek(1)
            raw = stream.read(4096)
    except OSError:
        return "an unidentified process"
    try:
        owner = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "an unidentified process"
    if not isinstance(owner, dict):
        return "an unidentified process"
    pid = owner.get("pid")
    since = owner.get("acquired_at_utc")
    target = owner.get("target")
    parts = [f"pid {pid}" if pid is not None else "an unidentified process"]
    if since:
        parts.append(f"since {since}")
    if target:
        parts.append(f"on {target}")
    return " ".join(parts)


def _timeout(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get(LOCK_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_LOCK_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{LOCK_TIMEOUT_ENV}={raw!r} is not a number of seconds"
        ) from error
    if value < 0:
        raise ValueError(f"{LOCK_TIMEOUT_ENV} cannot be negative")
    return value


class OutputLock:
    """Exclusive cross-process lock over one fetch output root.

    ``holder_progress`` is an optional probe of the holder's work (the
    bytes it has staged, say).  With one, the wait budget runs from the
    last time the probe's answer changed rather than from the start of
    the wait, so a slow but live holder is waited for to the end and
    only a stalled one is refused.  The concrete breakage it prevents:
    a flat budget refused a preparation waiting on another's 2.28 GB
    land-cover download that was still arriving, on any link slower
    than about 3.8 MB/s.
    """

    def __init__(self, kind: str, target: str | Path, *,
                 timeout_s: float | None = None, progress=print,
                 clock=time.monotonic, sleeper=time.sleep,
                 holder_progress: Callable[[], object] | None = None) -> None:
        self.kind = kind
        self.target = Path(target)
        self.path = lock_path(kind, target)
        self.timeout_s = _timeout(timeout_s)
        self._progress = progress
        self._clock = clock
        self._sleeper = sleeper
        self._holder_progress = holder_progress
        self._key = str(self.path).lower() if os.name == "nt" \
            else str(self.path)
        self._held = False

    def _busy(self, patience: _Patience | None = None) -> FetchLockBusy:
        holder = describe_holder(self.path)
        why = ("  why: two writers in one output directory can publish a "
               "receipt that describes the other one's bytes, so this run "
               "refuses rather than interleave.")
        if patience is None or patience.probe is None:
            return FetchLockBusy(layered(
                f"another gpuwm fetch is writing {self.target} "
                f"({holder}) and this run waited "
                f"{self.timeout_s:g} s for it.\n"
                "  remedy: wait for the other run to finish, fetch into a "
                f"different --out, or raise {LOCK_TIMEOUT_ENV} if the other "
                "run is expected to take longer.", why),
                budget_s=self.timeout_s,
                waited_s=(self.timeout_s if patience is None
                          else patience.waited_s),
                holder=holder)
        return FetchLockBusy(layered(
            f"another gpuwm fetch is writing {self.target} ({holder}) and "
            f"has shown no progress for {self.timeout_s:g} s; this run "
            f"waited {patience.waited_s:.0f} s for it in all.\n"
            "  remedy: find out why the other run stopped (a stalled "
            "connection, a stopped process) and let it finish or stop it, "
            f"or raise {LOCK_TIMEOUT_ENV} if it is expected to pause "
            "longer.", why),
            budget_s=self.timeout_s, waited_s=patience.waited_s,
            idle_s=patience.idle_s, holder=holder)

    def _announce_wait(self) -> None:
        if self._holder_progress is None:
            limit = f"waiting up to {self.timeout_s:g} s for it to finish"
        else:
            limit = ("waiting for it to finish; this run refuses only if "
                     f"it shows no progress for {self.timeout_s:g} s")
        self._progress(f"fetch: {self.target} is locked by "
                       f"{describe_holder(self.path)}; {limit}")

    def _acquire_key_lock(self, key_lock: threading.RLock) -> None:
        # Same thread: re-entrant, so nesting counts.  Another thread in
        # this process: a genuine second writer, so it queues here.
        if self.timeout_s <= 0:
            if not key_lock.acquire(blocking=False):
                raise self._busy()
            return
        if self._holder_progress is None:
            if not key_lock.acquire(timeout=self.timeout_s):
                raise self._busy()
            return
        patience = _Patience(self.timeout_s, self._clock,
                             self._holder_progress)
        announced = False
        while not key_lock.acquire(timeout=_POLL_S):
            if patience.expired():
                raise self._busy(patience)
            if not announced:
                self._announce_wait()
                announced = True

    def acquire(self) -> "OutputLock":
        key_lock = _key_lock(self._key)
        self._acquire_key_lock(key_lock)
        self._held = True
        try:
            with _REGISTRY_GUARD:
                entry = _HELD.get(self._key)
            if entry is not None and entry.depth > 0:
                entry.depth += 1
                return self
            self._take_os_lock()
        except BaseException:
            self._held = False
            key_lock.release()
            raise
        return self

    def _take_os_lock(self) -> None:
        ensure_lock_root(self.path.parent)
        try:
            stream = _open_lock_file(self.path)
        except PermissionError as error:
            # A shared GPUWM_FETCH_LOCK_ROOT whose lock file another
            # account left without write for others.
            raise FetchLockRootRefused(_lock_file_remedy(self.path, error),
                                       root=self.path.parent) from None
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            patience = _Patience(self.timeout_s, self._clock,
                                 self._holder_progress)
            announced = False
            while not _try_lock(stream):
                if patience.expired():
                    raise self._busy(patience)
                if not announced:
                    self._announce_wait()
                    announced = True
                self._sleeper(_POLL_S)
        except BaseException:
            stream.close()
            raise
        owner = json.dumps({
            "pid": os.getpid(),
            "kind": self.kind,
            "target": str(self.target),
            "acquired_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, sort_keys=True).encode("utf-8")
        try:
            stream.seek(1)
            stream.truncate()
            stream.write(owner)
            stream.flush()
        except OSError:
            pass  # the lock is what matters; the record is a courtesy
        entry = _Entry(stream)
        entry.depth = 1
        with _REGISTRY_GUARD:
            _HELD[self._key] = entry

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            with _REGISTRY_GUARD:
                entry = _HELD.get(self._key)
            if entry is not None:
                entry.depth -= 1
                if entry.depth <= 0:
                    with _REGISTRY_GUARD:
                        _HELD.pop(self._key, None)
                    _unlock(entry.stream)
                    entry.stream.close()
        finally:
            _key_lock(self._key).release()

    def __enter__(self) -> "OutputLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def hold(kind: str, target: str | Path, *, timeout_s: float | None = None,
         progress=print,
         holder_progress: Callable[[], object] | None = None) -> OutputLock:
    """``with hold('fetch-out', out):`` -- the single-writer contract."""

    return OutputLock(kind, target, timeout_s=timeout_s, progress=progress,
                      holder_progress=holder_progress)


class DirectoryProgress:
    """A ``holder_progress`` probe of the bytes a writer stages in a folder.

    The mark is every file's name, size and change time under ``root``
    (``.part`` staging included), so a download whose bytes keep arriving
    is a holder making progress.  The walk is redone at most every
    ``every_s`` seconds; between walks the last mark stands, which can
    only shorten a measured stall, never lengthen it.  The concrete
    breakage it prevents: a 36-member HRRR ensemble's member 4 refused
    after a flat 600 s wait on its sibling's download of the same cycle
    into the shared folder, which was still arriving at about 1 MiB/s
    (2026-10-24).
    """

    def __init__(self, root: str | Path, *, every_s: float = 2.0,
                 clock=time.monotonic) -> None:
        self.root = Path(root)
        self.every_s = every_s
        self._clock = clock
        self._taken = None
        self._mark: object = None

    def _walk(self) -> tuple:
        rows = []
        for directory, _folders, files in os.walk(self.root):
            for name in files:
                try:
                    stat = os.stat(os.path.join(directory, name))
                except OSError:
                    continue
                rows.append((directory, name, stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(rows))

    def __call__(self) -> object:
        now = self._clock()
        if self._taken is None:
            # The first call is the uncontended acquire's own: no walk.
            # A wait's first real mark counts once as progress.
            self._taken = now
            return None
        if now - self._taken >= self.every_s:
            self._taken = now
            self._mark = self._walk()
        return self._mark


def active_writer(kind: str, target: str | Path) -> bool:
    """Inspect an existing writer lock without creating files or waiting.

    Managed cache selection must queue behind an active partial download,
    rather than mistake its unfinished receipt for damage and duplicate it.
    An unreadable lock is conservatively treated as active; the real fetch
    still acquires its normal exclusive lock before it can write anything.
    """
    path = lock_path(kind, target)
    key = str(path).lower() if os.name == "nt" else str(path)
    with _REGISTRY_GUARD:
        entry = _HELD.get(key)
        if entry is not None and entry.depth > 0:
            return True
    try:
        stream = path.open("r+b")
    except FileNotFoundError:
        return False
    except OSError:
        return True
    with stream:
        try:
            if not _try_lock(stream):
                return True
            _unlock(stream)
            return False
        except OSError:
            return True


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------

def _fsync_dir(directory: Path) -> None:
    """Best-effort durability for the rename itself.

    POSIX needs the containing directory fsynced before a rename is
    durable.  Windows has no directory handle to fsync through the
    stdlib; there the file's own flush plus ``os.replace`` is what the
    platform offers, and this is a no-op rather than a pretence.
    """

    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


#: The longest name :func:`_staging_path` produces:
#: ``<tag, at most 8>-<10 hex>.tmp``.
STAGING_NAME_CHARS = 8 + 1 + 10 + 4

#: Fresh names tried before a publication gives up.  Each is 40 random
#: bits claimed with O_EXCL, so a second attempt already means another
#: writer holds the first name; eight in a row do not happen by chance.
_STAGING_ATTEMPTS = 8


def _staging_token() -> str:
    return secrets.token_hex(5)


def _staging_path(path: Path, tag: str, token: str | None = None) -> Path:
    """A compact sibling of ``path`` that one publication stages into.

    The name is the tag and a random token, never the target's own name.
    It used to be ``<name>.<tag>-<pid>-<time_ns>.tmp``, 35 characters
    longer than the file it publishes, so a folder deep enough to hold
    the published file could still be too deep for Windows (259
    characters without long paths) to create its staging copy: in an
    install 143 characters deep the geography resume record fit and its
    staging copy did not, and every geography setup failed there.
    Uniqueness now comes from :func:`atomic_write_bytes` claiming the
    name exclusively rather than from spelling out who wrote it.
    """

    return path.with_name(f"{tag[:8]}-{token or _staging_token()}.tmp")


def atomic_write_bytes(path: Path, payload: bytes, *,
                       tag: str = "publish") -> Path:
    """Publish ``payload`` at ``path`` or leave the old bytes alone.

    The staging file is claimed with O_EXCL under a fresh random name, so
    two publishers never share it -- the fixed ``<name>.tmp`` this
    replaced is the one file concurrent writers were guaranteed to
    collide on -- and a name another writer already holds is skipped
    rather than truncated.
    """

    for _ in range(_STAGING_ATTEMPTS):
        tmp = _staging_path(path, tag)
        try:
            stream = tmp.open("xb")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(
            errno.EEXIST,
            f"no free staging name for {path.name} after "
            f"{_STAGING_ATTEMPTS} attempts", str(path.parent))
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: Path, text: str, *,
                      tag: str = "publish") -> Path:
    """:func:`atomic_write_bytes` for UTF-8 text with LF newlines."""

    return atomic_write_bytes(path, text.encode("utf-8"), tag=tag)


# ---------------------------------------------------------------------------
# A write this computer refused is not a download that failed
# ---------------------------------------------------------------------------

#: The longest path Windows opens for a process when long paths are not
#: enabled on the machine: MAX_PATH (260) less the terminating null.
WINDOWS_PATH_LIMIT = 259

#: The widest process id a Windows staging name can carry: process ids
#: are 32-bit there, ten digits at most.  A measure of a path that holds
#: one uses this, since the limit above binds only on Windows.
WINDOWS_WIDEST_PID = 2 ** 32 - 1


def windows_path_limit() -> int | None:
    """:data:`WINDOWS_PATH_LIMIT` where it binds this process, else None.

    Python's own ``python.exe`` declares itself long-path aware, so the
    limit binds only on Windows and only while the machine's
    ``LongPathsEnabled`` switch is off, which is the Windows default.
    """

    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem"
                            ) as key:
            if winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1:
                return None
    except OSError:
        pass
    return WINDOWS_PATH_LIMIT


class LocalWriteFailed(Exception):
    """A write on this computer failed; carries the path it was writing.

    Deliberately not an ``OSError``: a transfer's own handler treats
    ``OSError`` as the network failing (a reset connection and a read
    timeout are both ``OSError``), and a local failure must never reach
    it.
    """

    def __init__(self, path: Path, error: OSError) -> None:
        super().__init__(str(error))
        self.path = path
        self.error = error


def receive(response, dest: Path, mode: str, *, block_bytes: int,
            on_bytes=None) -> None:
    """Stream an HTTP ``response`` into ``dest``; a failed write is local.

    Only the writes are wrapped.  Reads stay outside, so a reset
    connection or a timeout still reaches the caller as the network.
    ``on_bytes`` is called with the byte count written so far after each
    block, for a caller that reports progress.
    """

    try:
        sink = dest.open(mode)
    except OSError as error:
        raise LocalWriteFailed(dest, error) from error
    written = 0
    try:
        while block := response.read(block_bytes):
            try:
                sink.write(block)
            except OSError as error:
                raise LocalWriteFailed(dest, error) from error
            if on_bytes is not None:
                written += len(block)
                on_bytes(written)
    finally:
        try:
            sink.close()
        except OSError as error:
            raise LocalWriteFailed(dest, error) from error


def local_write_refusal(label: str, attempted: Path, error: OSError,
                        consequence: str) -> str:
    """The message for a write this computer would not do.

    Names the path's length, and where Windows' limit is what refused it,
    how many characters shorter the folder has to be.  The path goes
    last: a front end that shows one line cut to a few hundred
    characters still shows the reason and the fix, and the path itself
    can be longer than that line.  ``consequence`` says what the failure
    left behind for a re-run.
    """

    names = [str(attempted)] + [str(name) for name in
                                (error.filename, error.filename2) if name]
    path = max(names, key=len)
    length = len(path)
    limit = windows_path_limit()
    if limit is not None and length > limit:
        over = length - limit
        return (f"{label}: this computer cannot write a path of {length} "
                f"characters: Windows refuses paths longer than {limit} "
                "characters unless long paths are enabled.  This is not a "
                f"download failure.  Use a folder at least {over} "
                f"character{'' if over == 1 else 's'} shorter, or enable "
                f"Windows long paths; {consequence}.  Path: {path}")
    return (f"{label}: this computer could not write a path of {length} "
            f"characters ({error.strerror or error}).  This is not a "
            f"download failure; {consequence}.  Path: {path}")


# ---------------------------------------------------------------------------
# Quarantine (never deletes, never overwrites older evidence)
# ---------------------------------------------------------------------------

def aside_path(path: Path, tag: str = "rejected") -> Path:
    """A free ``<name>.<tag>-<stamp>`` beside ``path``.

    ``time.time_ns()`` alone is not free: two quarantines inside one
    clock tick, or two processes, can generate the same name, and
    ``os.replace`` onto an existing file overwrites it -- turning
    "nothing is ever deleted" into a lie.  This proves the name absent
    and disambiguates with a counter when it is not.
    """

    stamp = time.time_ns()
    candidate = path.with_name(f"{path.name}.{tag}-{stamp}")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = path.with_name(f"{path.name}.{tag}-{stamp}-{counter}")
    return candidate


def quarantine(path: Path, *, tag: str = "rejected") -> Path:
    """Move ``path`` aside to a proven-free name; returns the new path."""

    aside = aside_path(path, tag)
    os.replace(path, aside)
    return aside


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_S",
    "DirectoryProgress",
    "FetchLockBusy",
    "FetchLockRootRefused",
    "OutputFolderRefused",
    "LOCK_ROOT_ENV",
    "LOCK_TIMEOUT_ENV",
    "LocalWriteFailed",
    "OutputLock",
    "STAGING_NAME_CHARS",
    "WINDOWS_PATH_LIMIT",
    "WINDOWS_WIDEST_PID",
    "aside_path",
    "atomic_write_bytes",
    "atomic_write_text",
    "describe_holder",
    "ensure_lock_root",
    "hold",
    "local_write_refusal",
    "lock_path",
    "refuse_unwritable_output",
    "lock_root",
    "quarantine",
    "receive",
    "windows_path_limit",
]
