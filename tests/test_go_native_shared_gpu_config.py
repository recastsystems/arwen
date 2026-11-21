"""`gpuwm go CONFIG --native-shared-gpu` is refused, as --allow-shared-gpu is.

THE BREAKAGE: the flag is translated only by the WRF-input door
(gpuwm.wrfinput_go.validate_controls), so on a configuration launch it was
dropped without a word: the user asked for a shared card and got an
exclusive, supervised one.  Brought from jacobbieker/woof#634 (Apache-2.0).
"""
from gpuwm.cli import main as cli_main


def _config(tmp_path):
    out = tmp_path / "c.toml"
    assert cli_main(["domain", "--point=39.1,-94.6", "--card", "16gb", "--root-dx", "12",
                     "--hours", "3", "--source", "gfs", "--cycle", "2026-11-19T12",
                     "--out", str(out)]) == 0
    return out


def test_native_shared_gpu_is_refused_on_a_configuration_launch(tmp_path, capsys):
    config = _config(tmp_path)
    capsys.readouterr()
    rc = cli_main(["go", str(config), "--native-shared-gpu", "--dry-run"])
    text = capsys.readouterr()
    assert rc == 2, text.out + text.err
    assert "--native-shared-gpu requires WRF inputs" in text.out + text.err
