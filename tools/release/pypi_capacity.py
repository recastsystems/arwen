"""Read-only project storage preflight. Limits are bytes, configurable per project.

Public JSON counts all files in all releases, including yanked files. An exact
existing file costs no additional space on a retry. Invalid or unavailable
inventory refuses; no credentials, upload or deletion is used here.

The default limit is PyPI's own: warehouse/constants.py MAX_PROJECT_SIZE =
10 * ONE_GIB = 10,737,418,240 bytes, and Project.total_size_limit_value takes
the largest of that, a per-project grant and an organisation's limit, so the
effective limit is never below it. The file used to say 10,000,000,000, which
would have refused 2.9.11 (9.53 GB stored + 0.27 GB a release) although PyPI
takes it; the 2.8.9 intake recorded the project holding 10.69 GB, above
10,000,000,000, before its upload was refused. A granted increase is entered in
pypi_limits.json.

Four projects: gpuwm, gpuwm-data and, from 2.9.12, the kernel companions
gpuwm-kernels and gpuwm-kernels-startup. A project PyPI does not have yet
answers 404 to the public JSON; that is an empty inventory, not an unreadable
one (the 2.9.12 R14 preflight refused the sealed packet on the first kernels
wheel because only two projects were known).

headroom() and storage_line() are the early warning: R2 reports, every pass,
how many releases of the newest one's size still fit, long before R14 refuses
a sealed packet on cut day.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

PROJECTS = ("gpuwm", "gpuwm-data", "gpuwm-kernels", "gpuwm-kernels-startup")
LIMITS = Path(__file__).with_name("pypi_limits.json")


class CapacityError(ValueError):
    pass


def positive_bytes(value, label):
    if type(value) is not int or value < 0:
        raise CapacityError(f"{label}: expected nonnegative integer bytes, got {value!r}")
    return value


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def inventory(document, project):
    if normalize(document.get("info", {}).get("name", "")) != project:
        raise CapacityError(f"{project}: public JSON names a different project")
    releases = document.get("releases")
    if not isinstance(releases, dict):
        raise CapacityError(f"{project}: public JSON has no complete releases inventory")
    files = {}
    for version, rows in releases.items():
        if not isinstance(rows, list):
            raise CapacityError(f"{project}: invalid release {version}")
        for row in rows:
            try:
                name, sha = row["filename"], row["digests"]["sha256"]
                size = positive_bytes(row["size"], f"{project}/{name}")
            except (KeyError, TypeError) as exc:
                raise CapacityError(f"{project}: incomplete file inventory") from exc
            if not isinstance(name, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise CapacityError(f"{project}: invalid file identity")
            identity = (size, sha)
            if name in files and files[name] != identity:
                raise CapacityError(f"{project}: conflicting public file {name}")
            files[name] = identity
    return files


def evaluate(project, document, new_files, limit):
    limit = positive_bytes(limit, f"{project} limit")
    if not limit:
        raise CapacityError(f"{project}: limit must be positive")
    files = inventory(document, project)
    added, seen = 0, set()
    for row in new_files:
        name = row["filename"]
        if name in seen:
            raise CapacityError(f"{project}: duplicate proposed file {name}")
        seen.add(name)
        size = positive_bytes(row["bytes"], f"{project}/{name}")
        sha = row["sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise CapacityError(f"{project}: invalid proposed hash for {name}")
        if name in files:
            if files[name] != (size, sha):
                raise CapacityError(f"{project}: existing file differs: {name}")
        else:
            added += size
    current = sum(size for size, _ in files.values())
    projected = current + added
    return {"project": project, "current_bytes": current, "new_bytes": added,
            "projected_bytes": projected, "limit_bytes": limit,
            "headroom_bytes": limit - projected, "existing_files": len(files),
            "releases": len(document["releases"]), "status": "PASS" if projected <= limit else "REFUSED"}


def empty_project(project):
    """The inventory of a project PyPI does not have yet: no releases, no files."""
    return {"info": {"name": project}, "releases": {}, "urls": []}


def fetch_project(project):
    if project not in PROJECTS:
        raise CapacityError(f"unknown project {project}")
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # not on PyPI yet (a new companion project): nothing stored
            return empty_project(project)
        raise CapacityError(f"{project}: cannot read complete public JSON inventory: {exc}") from exc
    except Exception as exc:
        raise CapacityError(f"{project}: cannot read complete public JSON inventory: {exc}") from exc


#: Warn while there is still a release cycle to act: at this many releases of
#: room or fewer, the line says who decides (a limit-increase request is a
WARN_RELEASES = 3


def release_bytes(document, project):
    """Bytes the next release is priced at: the newest release's files.

    ``urls`` lists the files of ``info.version``; without it, the largest
    release so far (the cautious price).
    """
    latest = document.get("urls") or []
    size = sum(positive_bytes(row.get("size"), f"{project}/{row.get('filename')}") for row in latest)
    if size:
        return size
    return max((sum(row["size"] for row in rows) for rows in document["releases"].values()), default=0)


def headroom(project, document, limit):
    """Room left on PyPI, in bytes and in releases of the newest one's size."""
    limit = positive_bytes(limit, f"{project} limit")
    if not limit:
        raise CapacityError(f"{project}: limit must be positive")
    files = inventory(document, project)
    current = sum(size for size, _ in files.values())
    per_release = release_bytes(document, project)
    room = limit - current
    fits = max(room, 0) // per_release if per_release else None
    return {"project": project, "current_bytes": current, "limit_bytes": limit, "headroom_bytes": room,
            "release_bytes": per_release, "releases_that_fit": fits, "releases": len(document["releases"]),
            "warn": fits is not None and fits <= WARN_RELEASES}


def storage_report(*, fetch=fetch_project, limits=None):
    limits = json.loads(LIMITS.read_text(encoding="utf-8")) if limits is None else limits
    return [headroom(project, fetch(project), limits[project]) for project in PROJECTS]


def storage_line(row):
    used = (f"PyPI {row['project']}: {row['current_bytes'] / 1e9:.2f} of {row['limit_bytes'] / 1e9:.2f} GB "
            f"stored ({row['releases']} releases)")
    fits = row["releases_that_fit"]
    if fits is None:
        return used + "; no release to price the next one by"
    text = f"{used}; room for {fits} more of {row['release_bytes'] / 1e6:.0f} MB, this one included"
    if fits == 0:
        return (text + "; THIS RELEASE DOES NOT FIT: PyPI refuses it ('Project size too large', the 2.8.9 cut "
                "day) and R14 will refuse the packet. The package owner decides: a limit-increase request to pypi/support "
                "(owner action) or deleting old releases (approval per item)")
    if row["warn"]:
        return (text + "; LOW: get owner approval before the cut that would not fit: a limit-increase request to "
                "pypi/support (drafted, owner action) or deleting old releases (approval per item)")
    return text


def refusal(row):
    return (f"REFUSED: {row['project']} project storage: current={row['current_bytes']} bytes + "
            f"new={row['new_bytes']} bytes = {row['projected_bytes']} bytes; "
            f"limit={row['limit_bytes']} bytes; exceeds by {-row['headroom_bytes']} bytes")


def check_plan(plan, *, fetch=fetch_project, limits=None):
    limits = json.loads(LIMITS.read_text(encoding="utf-8")) if limits is None else limits
    unknown = sorted(set(plan) - set(PROJECTS))
    if unknown:
        raise CapacityError(f"unknown project {unknown[0]}")
    rows = [evaluate(project, fetch(project), plan[project], limits[project]) for project in PROJECTS if project in plan]
    return {"schema": "arwen.pypi-capacity.v1", "checked_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "REFUSED", "projects": rows}


def packet_project(name):
    """The PyPI project a packet file belongs to, from its distribution name.

    Wheels and sdists both start ``<distribution>-<version>``; the distribution
    is the project's name with ``-`` written ``_`` (gpuwm_kernels_startup-2.9.12.tar.gz
    is gpuwm-kernels-startup). Anything not one of PROJECTS refuses.
    """
    if Path(name).name != name or "-" not in name:
        raise CapacityError(f"unexpected PyPI packet filename {name}")
    project = normalize(name.split("-", 1)[0])
    if project not in PROJECTS:
        raise CapacityError(f"unexpected PyPI packet filename {name}")
    return project


def packet_plan(packet):
    manifest = packet / "PUBLICATION-ASSETS.json"
    data = manifest.read_bytes()
    rows = json.loads(data)["pypi"]["artifacts"]
    plan = {project: [] for project in PROJECTS}
    for row in rows:
        name = row["filename"]
        project = packet_project(name)
        path = packet / name
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["bytes"] or sha != row["sha256"]:
            raise CapacityError(f"{project}: packet file differs: {name}")
        plan[project].append(row)
    missing = [project for project in PROJECTS if not plan[project]]
    if missing:
        raise CapacityError("packet must include every PyPI project; missing " + ", ".join(missing))
    return plan, hashlib.sha256(data).hexdigest()


def check_packet(packet, *, fetch=fetch_project, limits=None, out=None):
    plan, manifest_sha = packet_plan(packet)
    receipt = check_plan(plan, fetch=fetch, limits=limits)
    receipt["manifest_sha256"] = manifest_sha
    if out:
        out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    failed = [refusal(row) for row in receipt["projects"] if row["status"] != "PASS"]
    if failed:
        raise CapacityError("\n".join(failed))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--limits", type=Path, default=LIMITS)
    parser.add_argument("--fixture-dir", type=Path, help="recorded public JSON, never a live approval")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fetch = (lambda p: json.loads((args.fixture_dir / f"{p}.json").read_text())) if args.fixture_dir else fetch_project
    try:
        check_packet(args.packet, fetch=fetch, limits=json.loads(args.limits.read_text()), out=args.out)
    except (CapacityError, KeyError, OSError, ValueError) as exc:
        print(str(exc))
        return 1
    print(f"PASS: every project fits ({', '.join(PROJECTS)}); receipt {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
