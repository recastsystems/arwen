"""A module run with ``python -m`` reaches every name its code uses.

THE BREAKAGE THIS PREVENTS (gpuwm 3.0.4, and 3.0.5 at e8d7c8098d): a forecast
started as ``python -m gpuwm.prepared_single_domain_forecast`` -- the documented
door, and the way the ensemble and paired doors start every leg -- ran all its
steps and then stopped at its report with ``NameError: name
'_wrf_reference_receipt' is not defined``.  The helper was defined below
``if __name__ == "__main__": raise SystemExit(main())``, so a module run called
``main()`` before the ``def`` line was reached.  Imported (as ``gpuwm go``
does) it worked, which is why the cut's own runs did not see it.

Two tests: a CPU one over every engine module (a name the module's own code
uses is never defined below its main guard), and a card one that runs a
prepared forecast with ``python -m`` to its report.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "gpuwm"


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    names = [test.left, *test.comparators]
    return (any(isinstance(n, ast.Name) and n.id == "__name__" for n in names)
            and any(isinstance(n, ast.Constant) and n.value == "__main__"
                    for n in names))


def _late_names_used_above(tree: ast.Module) -> set[str]:
    """Names bound below the main guard and read by code above it."""
    body = tree.body
    guard = next((i for i, node in enumerate(body) if _is_main_guard(node)), None)
    if guard is None:
        return set()
    late: set[str] = set()
    for node in body[guard + 1:]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            late.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            late.update(t.id for t in targets if isinstance(t, ast.Name))
    # __all__ below the guard is read by star imports only, never by a run.
    late.discard("__all__")
    used = {node.id for part in body[:guard + 1] for node in ast.walk(part)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return late & used


def test_no_engine_module_uses_a_name_defined_below_its_main_guard():
    offenders = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = _late_names_used_above(tree)
        if names:
            offenders[str(path.relative_to(ROOT))] = sorted(names)
    assert offenders == {}, (
        "defined below `if __name__ == \"__main__\"` but used by the module's "
        "own code, so a `python -m` run raises NameError when it gets there: "
        f"{offenders}")


def test_the_prepared_forecast_helper_sits_above_the_guard():
    path = PACKAGE / "prepared_single_domain_forecast.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = next(i for i, node in enumerate(tree.body) if _is_main_guard(node))
    assert guard == len(tree.body) - 1, "the main guard is the module's last statement"
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_wrf_reference_receipt"
               for node in tree.body[:guard])


def test_a_prepared_forecast_run_with_python_m_reaches_its_report(tmp_path):
    """The real door on a card: ``GPUWM_MODULE_RUN_CASE`` names a JSON file
    ``{"argv": [... the forecast's arguments without --outdir ...],
    "timeout_seconds": 1800}`` for a small prepared forecast (keep it short
    with ``--run-seconds``)."""
    path = os.environ.get("GPUWM_MODULE_RUN_CASE")
    if not path:
        pytest.skip("GPUWM_MODULE_RUN_CASE names no prepared forecast")
    case = json.loads(Path(path).read_text())
    out = Path(case.get("outdir") or tmp_path / "module-run")
    command = [sys.executable, "-m", "gpuwm.prepared_single_domain_forecast",
               *case["argv"], "--outdir", str(out)]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=float(case.get("timeout_seconds", 1800)))
    assert "NameError" not in result.stderr, result.stderr[-3000:]
    assert result.returncode == 0, result.stderr[-3000:]
    # The door names a run folder under --outdir; a failed run writes a FAIL
    # report there with the error, so the status is what is checked.
    reports = sorted(out.glob("**/report.json"))
    assert reports, f"no report under {out}"
    report = json.loads(reports[-1].read_text())
    assert report["status"] == "PASS", report.get("error")
