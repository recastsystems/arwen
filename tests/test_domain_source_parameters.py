"""`gpuwm domain --source-parameters`: HAFS's storm id reaches [fetch] and the printed fetch.

THE BREAKAGE: the route table declares STORM required, with no default, for
hafs-a and hafs-b, and `gpuwm domain` had no --source-parameters, so the
[fetch] table it wrote and the fetch command it printed lacked the storm id
and the fetch refused ("--source-parameters requires STORM=VALUE").
Brought from jacobbieker/woof#344 (Apache-2.0).
"""
import tomllib

import pytest

from gpuwm.cli import main as cli_main


def _domain(tmp_path, *extra):
    out = tmp_path / "h.toml"
    rc = cli_main(["domain", "--point=25.0,-60.0", "--card", "16gb", "--root-dx", "6",
                   "--hours", "6", "--source", "hafs-a", "--cycle", "2025-10-15T00",
                   "--out", str(out), *extra])
    return rc, out


def test_storm_id_reaches_the_fetch_table_and_printed_command(tmp_path, capsys):
    # --explain prints the numbered steps, the fetch command among them.
    rc, out = _domain(tmp_path, "--source-parameters", "STORM=12l", "--explain")
    printed = capsys.readouterr().out
    assert rc == 0
    fetch = tomllib.loads(out.read_text(encoding="utf-8"))["fetch"]
    assert fetch["source_parameters"] == "STORM=12l"
    assert "--source-parameters STORM=12l" in printed


def test_a_missing_storm_id_is_refused_before_a_table_is_written(tmp_path, capsys):
    with pytest.raises((SystemExit, ValueError)) as caught:
        rc, out = _domain(tmp_path)
        assert rc != 0
        raise SystemExit(rc)
    text = capsys.readouterr()
    assert "STORM" in (str(caught.value) + text.out + text.err)
    assert not (tmp_path / "h.toml").exists()
