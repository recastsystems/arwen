"""What a finished fetch names as the next step, by the model that reads it.

``gpuwm fetch`` used to end with one ``next:`` block, the regional
preparation door (``gpuwm prep``), whatever the fetched data was for.  The
breakage it prevents: the ArWen Global README's first step is a
whole-globe GDAS analysis (``gpuwm fetch --source gdas --hours 0 --mode
full-file``), and the fetch then sent the reader to ``gpuwm prep``, the
regional route, which is not the global model's next step (gpuwm 3.0.5,
found in a fresh-user test of gpuwm-global 0.3.0).

A model that starts from a fetched folder is one row of :data:`ROWS`: the
model's name, the package and module that say it is installed, the
sources it starts from, and what it needs from the fetch.  A row is
printed only when its package is installed and the folder holds what it
needs, so a box with only the engine sees exactly the regional block it
saw before, under a heading that names the regional model.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Callable, Sequence

#: The heading of the regional block (``gpuwm prep``), the engine's own.
REGIONAL_MODEL = "a regional ArWen forecast"


@dataclass(frozen=True)
class NextDoor:
    """One model that starts from a fetched folder."""

    #: The model's public name, as the heading prints it.
    model: str
    #: The distribution a reader installs, named in the heading.
    package: str
    #: The importable module whose presence says the package is installed.
    module: str
    #: The fetch sources this model starts from.
    sources: tuple[str, ...]
    #: Returns the lines under the heading, or ``()`` when the folder does
    #: not hold what the model needs.  Called with (source, out, manifest).
    lines: Callable[[str, Path, dict], tuple[str, ...]]


def _manifest(out: Path) -> dict:
    from gpuwm.fetch_routes import MANIFEST_NAME

    try:
        return json.loads((Path(out) / MANIFEST_NAME).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def whole_globe_analysis(out: Path, manifest: dict) -> Path | None:
    """The f000 object of a whole-file fetch, or None.

    Whole-globe means whole archive objects (``mode = full-file``) with no
    area crop; the default GDAS transport is an area crop, and a crop
    cannot start a global model.
    """

    if manifest.get("mode") != "full-file" or manifest.get("area"):
        return None
    for entry in manifest.get("files") or ():
        name = str(entry.get("name") or "")
        if (entry.get("forecast_hour") == 0 and entry.get("role") != "series"
                and not name.endswith((".tsv", ".json", ".txt"))):
            return (Path(out) / name).resolve()
    return None


#: The module of gpuwm-global, the package that ships ArWen Global.
GLOBAL_MODULE = "arwen_global"

#: The suffix of a shipped experiment file.
CONFIG_SUFFIX = ".toml"


def shipped_configs_folder(module: str = GLOBAL_MODULE) -> Path | None:
    """The folder of experiment files an installed model package ships.

    gpuwm-global keeps its shipped experiments in the ``configs`` folder
    inside its package (``arwen_global/configs/<name>.toml``), the folder
    its own ``configs_dir.config_root()`` names.  The engine finds that
    folder by the package's location and reads the files as data; it
    imports and runs none of gpuwm-global's code, so no engine module
    depends on that package (``tests/test_obs_streams_in_engine.py``
    keeps it so).  None when the package is not installed or ships no
    such folder.
    """

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    for location in spec.submodule_search_locations or ():
        folder = Path(location) / "configs"
        if folder.is_dir():
            return folder
    return None


def shipped_config_names(folder: Path | None = None) -> tuple[str, ...]:
    """Every shipped experiment name, without its suffix, sorted."""

    folder = shipped_configs_folder() if folder is None else folder
    if folder is None or not folder.is_dir():
        return ()
    return tuple(sorted(path.stem for path in folder.glob(f"*{CONFIG_SUFFIX}")))


def resolve_shipped_config(value) -> Path | None:
    """The shipped experiment a bare name means, or None.

    The rule ``gpuwm-global`` applies to a name it is given: only a bare
    name (no folder) resolves, with or without ``.toml``, to the file of
    that name in the shipped folder.  A path with a folder is the
    caller's own file and never becomes the package's copy.
    """

    path = Path(str(value))
    if path.parent not in (Path("."), Path("")):
        return None
    folder = shipped_configs_folder()
    if folder is None:
        return None
    stem = (path.name[:-len(CONFIG_SUFFIX)]
            if path.name.endswith(CONFIG_SUFFIX) else path.name)
    candidate = folder / f"{stem}{CONFIG_SUFFIX}"
    return candidate if candidate.is_file() else None


def _configs_reading(analysis: Path) -> list[tuple[int, str]]:
    """Shipped ArWen Global configs whose [initial] analysis_grib is this
    file, read from the folder the command runs in, as (truncation, name).
    """

    import tomllib

    root = shipped_configs_folder()
    if root is None:
        return []
    named: list[tuple[int, str]] = []
    for name in shipped_config_names(root):
        try:
            raw = tomllib.loads((root / f"{name}.toml").read_text(
                encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        grib = (raw.get("initial") or {}).get("analysis_grib")
        if not grib:
            continue
        try:
            if Path(grib).expanduser().resolve() != analysis:
                continue
            if raw.get("world"):
                # Another planet's run starts from Earth's analysis too,
                # but it is not the forecast a reader of this hint wants.
                continue
            truncation = int((raw.get("grid") or {}).get("truncation") or 0)
        except (OSError, ValueError, TypeError):
            continue
        named.append((truncation, name))
    return named


def one_per_grid(named: Sequence[tuple[int, str]]) -> list[str]:
    """One config per truncation, smallest grid first.

    Of each truncation's configs, the base experiment: the one whose name
    the most others extend (``..._24h`` before ``..._24h_dt600``), the
    shorter name on a tie.
    """

    names = [name for _truncation, name in named]
    best: dict[int, tuple[int, int, str]] = {}
    for truncation, name in named:
        extended = sum(1 for other in names
                       if other != name and other.startswith(name + "_"))
        key = (-extended, len(name), name)
        if truncation not in best or key < best[truncation]:
            best[truncation] = key
    return [best[truncation][2] for truncation in sorted(best)]


def _arwen_global_lines(source: str, out: Path,
                        manifest: dict) -> tuple[str, ...]:
    analysis = whole_globe_analysis(out, manifest)
    if analysis is None:
        return ()
    shipped = _configs_reading(analysis)
    lines = []
    if len(shipped) == 1:
        name = shipped[0][1]
        lines.append(f"  gpuwm-global go {name} --outdir OUT")
        lines.append(f"  # {name} reads this file as its [initial] "
                     f"analysis_grib from {Path.cwd()}.")
    elif shipped:
        lines.append("  gpuwm-global go CONFIG --outdir OUT")
        lines.append(f"  # {len(shipped)} shipped configs read this file as "
                     f"their [initial] analysis_grib from {Path.cwd()}; one "
                     "per grid, smallest first:")
        lines.extend(f"  #   {name}" for name in one_per_grid(shipped))
        lines.append("  # `gpuwm-global configs` lists every config.")
    else:
        lines.append("  gpuwm-global go CONFIG --outdir OUT")
        lines.append("  # CONFIG names this file in its [initial] table:")
        lines.append(f'  #   analysis_grib = "{analysis.as_posix()}"')
        lines.append("  # `gpuwm-global configs` lists the shipped configs; "
                     "copy one and edit it.")
    lines.append("  # --outdir is yours; `gpuwm-global statics CONFIG` then "
                 "`gpuwm-global run CONFIG` run the same day one stage at "
                 "a time.")
    return tuple(lines)


#: Every model, other than the regional route, that starts from a fetched
#: folder.  A new model is a row here, not a new branch in the fetch.
ROWS: tuple[NextDoor, ...] = (
    NextDoor(model="ArWen Global", package="gpuwm-global",
             module=GLOBAL_MODULE, sources=("gdas",),
             lines=_arwen_global_lines),
)


def installed(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def model_lines(label: str, source: str, out: Path, *,
                rows: Sequence[NextDoor] | None = None,
                is_installed: Callable[[str], bool] | None = None
                ) -> tuple[str, ...]:
    """The blocks of every installed model this folder can start."""

    rows = ROWS if rows is None else rows
    is_installed = installed if is_installed is None else is_installed
    manifest = _manifest(out)
    lines: list[str] = []
    for row in rows:
        if source not in row.sources or not is_installed(row.module):
            continue
        body = row.lines(source, Path(out), manifest)
        if body:
            lines.append(f"fetch {label}: next, for {row.model} "
                         f"({row.package}, installed):")
            lines.extend(body)
    return tuple(lines)


__all__ = ["CONFIG_SUFFIX", "GLOBAL_MODULE", "NextDoor", "REGIONAL_MODEL",
           "ROWS", "installed", "model_lines", "one_per_grid",
           "resolve_shipped_config", "shipped_config_names",
           "shipped_configs_folder", "whole_globe_analysis"]
