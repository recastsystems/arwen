"""Open a file in a folder several accounts share, as Linux lets a second account open it.

The breakage this prevents, named: Linux refuses an ``O_CREAT`` open of a file
another account owns in a sticky world-writable folder (``/tmp`` and the
folders gpuwm makes like it), unless the folder's owner made the file, whatever
the file's mode.  That is ``fs.protected_regular``, on by default on Ubuntu and
most distributions.  A second account's fetch died with "[Errno 13] Permission
denied: '/tmp/gpuwm-fetch-connections/slot-1'" on a 0666 slot the first
account had made (3.0.6 candidate, two accounts fetching at once on one
host, 24 Nov 2026), and a card lock file another account made in the shared
card lock folder was refused the same way.

An existing file is therefore opened without ``O_CREAT``; only a missing one
is created, with ``O_EXCL``, and a file that appears between the two calls is
opened as it is.  A file another account made a moment ago can still carry its
umask's mode for the few microseconds before that account opens it to every
account; such an open is tried again briefly before the refusal reaches the
caller, whose remedy names the account and the mode.
"""
from __future__ import annotations

import os
import time

#: How often, and how far apart, a refused open of a just-made file is tried again.
PERMISSION_RETRIES = 20
PERMISSION_RETRY_S = 0.01


def open_shared(path, flags: int, mode: int = 0o666) -> int:
    """``os.open(path, flags | O_CREAT, mode)`` that another account's file in a sticky folder does not refuse.

    Returns the descriptor.  ``flags`` may carry ``O_CREAT`` or not; ``O_EXCL``
    is not accepted (the caller wants the file whoever made it).  The mode of a
    file this call creates is ``mode`` less the umask, as for ``os.open``; the
    caller widens it if it must.
    """
    if flags & os.O_EXCL:
        raise ValueError("open_shared opens a file whoever made it; O_EXCL is not one of its flags")
    flags &= ~os.O_CREAT
    refused = None
    retries = 0
    for _ in range(1000):
        try:
            return os.open(path, flags)
        except FileNotFoundError:
            pass
        except PermissionError as error:
            refused = error
            retries += 1
            if retries > PERMISSION_RETRIES:
                raise
            time.sleep(PERMISSION_RETRY_S)
            continue
        try:
            return os.open(path, flags | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue  # made by another process between the two calls: open it as it is
        except PermissionError:
            if refused is not None:
                raise refused from None
            raise
    # A file made and removed under this call a thousand times over: nothing to open.
    raise FileNotFoundError(f"{path} was made and removed by another process on every one of 1000 opens")
