from tools.build_rw_wps_release import _stage_rw_wps_python_project

def test_cpu_preparation_stages_the_new_scheme_without_forecast_executors(tmp_path):
    result = _stage_rw_wps_python_project(tmp_path / "cpu-preparation")
    assert result["unresolved_internal_imports"] == []
    assert result["forecast_executor_files"] == []
    assert "gpuwm/core/model.py" not in result["files"]
    assert "gpuwm/core/physics.py" not in result["files"]
    assert "gpuwm/core/dycore.py" not in result["files"]
