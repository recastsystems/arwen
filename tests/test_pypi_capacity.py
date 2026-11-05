"""Recorded JSON fixtures and exact local bytes exercise project capacity without network."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.release import pypi_capacity as capacity


def row(name, size=260_000_000, sha="a" * 64):
    return {"filename": name, "bytes": size, "sha256": sha}


def public(project, size, *, name="old.whl", yanked=False):
    return {"info": {"name": project}, "releases": {"2.8.8": [
        {"filename": name, "size": size, "yanked": yanked, "digests": {"sha256": "a" * 64}}]}}


@pytest.fixture
def recorded():
    # Recorded reduced shape of PyPI's project JSON, with the historical totals
    # from the intake. No test reaches a network, even on an error path.
    return json.loads(Path(__file__).with_name("pypi_capacity_recorded.json").read_text())


def test_historical_over_limit_refusal_names_project_and_every_number(recorded):
    receipt = capacity.check_plan({"gpuwm": [row("gpuwm-new.whl")],
        "gpuwm-data": [row("gpuwm_data-new.whl", 190_000_000)]}, fetch=recorded.__getitem__)
    assert receipt["status"] == "REFUSED"
    assert receipt["projects"][1]["status"] == "PASS"
    assert capacity.refusal(receipt["projects"][0]) == (
        "REFUSED: gpuwm project storage: current=10690000000 bytes + new=260000000 bytes = "
        "10950000000 bytes; limit=10737418240 bytes; exceeds by 212581760 bytes")


def test_companion_can_refuse_independently_and_limit_increase_is_configurable(recorded):
    recorded["gpuwm-data"] = public("gpuwm_data", 10_600_000_000)
    plan = {"gpuwm": [row("gpuwm-new.whl")], "gpuwm-data": [row("gpuwm_data-new.whl", 190_000_000)]}
    receipt = capacity.check_plan(plan, fetch=recorded.__getitem__)
    assert [r["status"] for r in receipt["projects"]] == ["REFUSED", "REFUSED"]
    raised = capacity.check_plan(plan, fetch=recorded.__getitem__, limits=dict.fromkeys(capacity.PROJECTS, 20_000_000_000))
    assert raised["status"] == "PASS"


def test_exact_limit_is_allowed_yanked_files_count_and_retry_does_not_double_count():
    document = public("gpuwm", 100, yanked=True)
    document["urls"] = document["releases"]["2.8.8"]  # do not count latest urls twice
    assert capacity.evaluate("gpuwm", document, [row("new.whl", 10)], 110)["status"] == "PASS"
    assert capacity.evaluate("gpuwm", document, [row("old.whl", 100)], 100)["new_bytes"] == 0
    with pytest.raises(capacity.CapacityError, match="differs"):
        capacity.evaluate("gpuwm", document, [row("old.whl", 100, "b" * 64)], 1000)


@pytest.mark.parametrize("document", [{}, {"info": {"name": "gpuwm"}}, public("wrong", 10), public("gpuwm", -1)])
def test_incomplete_or_wrong_public_inventory_refuses(document):
    with pytest.raises(capacity.CapacityError):
        capacity.evaluate("gpuwm", document, [], 1000)


PACKET_FILES = ("gpuwm-9.0.whl", "gpuwm_data-9.0.whl", "gpuwm_kernels-9.0-py3-none-any.whl",
                "gpuwm_kernels-9.0.tar.gz", "gpuwm_kernels_startup-9.0-py3-none-any.whl", "gpuwm_kernels_startup-9.0.tar.gz")


def sealed(tmp_path, names=PACKET_FILES):
    artifacts = []
    for name in names:
        payload = b"sealed bytes " + name.encode()
        (tmp_path / name).write_bytes(payload)
        artifacts.append(row(name, len(payload), hashlib.sha256(payload).hexdigest()))
    (tmp_path / "PUBLICATION-ASSETS.json").write_text(json.dumps({"pypi": {"artifacts": artifacts}}))
    return artifacts


def new_projects(recorded):
    # The 2.9.12 shape: the kernel companions are not on PyPI yet, so their inventory is the 404's empty one.
    recorded["gpuwm-kernels"] = capacity.empty_project("gpuwm-kernels")
    recorded["gpuwm-kernels-startup"] = capacity.empty_project("gpuwm-kernels-startup")
    return recorded


def test_packet_hashes_are_verified_and_a_refusal_receipt_survives(tmp_path, recorded):
    artifacts = sealed(tmp_path)
    recorded = new_projects(recorded)
    out = tmp_path / "capacity.json"
    full = dict.fromkeys(capacity.PROJECTS, 10 * 1024 ** 3)
    full["gpuwm"] = 10_690_000_000  # the recorded total, to the byte
    with pytest.raises(capacity.CapacityError, match="gpuwm project storage"):
        capacity.check_packet(tmp_path, fetch=recorded.__getitem__, limits=full, out=out)
    assert json.loads(out.read_text())["status"] == "REFUSED"
    (tmp_path / artifacts[0]["filename"]).write_bytes(b"changed")
    with pytest.raises(capacity.CapacityError, match="packet file differs"):
        capacity.check_packet(tmp_path, fetch=lambda p: pytest.fail("network was attempted"))


def test_duplicates_conflicting_rows_and_invalid_limits_refuse():
    document = public("gpuwm", 10)
    with pytest.raises(capacity.CapacityError, match="duplicate"):
        capacity.evaluate("gpuwm", document, [row("new", 1), row("new", 1)], 100)
    document["releases"]["retry"] = [dict(document["releases"]["2.8.8"][0], size=11)]
    with pytest.raises(capacity.CapacityError, match="conflicting"):
        capacity.inventory(document, "gpuwm")
    for limit in (True, -1, 0, "10 GB"):
        with pytest.raises(capacity.CapacityError):
            capacity.evaluate("gpuwm", public("gpuwm", 10), [], limit)


def test_default_limit_is_pypis_own_10_gib():
    # warehouse/constants.py MAX_PROJECT_SIZE = 10 * ONE_GIB; the effective limit is the largest of it and any
    # grant, so a decimal 10,000,000,000 refused 2.9.11 although PyPI takes it.
    limits = json.loads(capacity.LIMITS.read_text(encoding="utf-8"))
    assert limits == dict.fromkeys(capacity.PROJECTS, 10 * 1024 ** 3)


def latest(project, stored, newest):
    document = public(project, stored)
    document["releases"]["2.9.9"] = [{"filename": "new.whl", "size": newest, "yanked": False,
                                      "digests": {"sha256": "b" * 64}}]
    document["urls"] = document["releases"]["2.9.9"]
    return document


def test_headroom_prices_the_next_release_by_the_newest_and_warns_a_cycle_ahead():
    # PyPI JSON on 2026-10-29: gpuwm 9,533,260,024 B stored, 2.9.9 = 273,464,174 B.
    document = latest("gpuwm", 9_533_260_024 - 273_464_174, 273_464_174)
    row = capacity.headroom("gpuwm", document, 10 * 1024 ** 3)
    assert (row["current_bytes"], row["release_bytes"], row["releases_that_fit"], row["warn"]) == (
        9_533_260_024, 273_464_174, 4, False)
    assert capacity.storage_line(row) == ("PyPI gpuwm: 9.53 of 10.74 GB stored (2 releases); "
                                          "room for 4 more of 273 MB, this one included")
    low = capacity.headroom("gpuwm", document, 10_000_000_000)
    assert low["releases_that_fit"] == 1 and "LOW: get owner approval" in capacity.storage_line(low)
    full = capacity.headroom("gpuwm", document, 9_600_000_000)
    assert full["releases_that_fit"] == 0 and "THIS RELEASE DOES NOT FIT" in capacity.storage_line(full)
    del document["urls"]  # no newest files listed: priced at the largest release
    assert capacity.headroom("gpuwm", document, 10 * 1024 ** 3)["release_bytes"] == 9_533_260_024 - 273_464_174


def test_kernel_companion_packet_files_map_to_their_projects_and_a_404_project_is_empty(tmp_path, recorded):
    # 2.9.12 R14: "unexpected PyPI packet filename gpuwm_kernels-2.9.12-py3-none-any.whl" refused the sealed packet
    # because only gpuwm and gpuwm-data were known. The distribution name decides the project.
    assert capacity.PROJECTS == ("gpuwm", "gpuwm-data", "gpuwm-kernels", "gpuwm-kernels-startup")
    assert capacity.packet_project("gpuwm_kernels-2.9.12-py3-none-any.whl") == "gpuwm-kernels"
    assert capacity.packet_project("gpuwm_kernels-2.9.12.tar.gz") == "gpuwm-kernels"
    assert capacity.packet_project("gpuwm_kernels_startup-2.9.12-py3-none-any.whl") == "gpuwm-kernels-startup"
    assert capacity.packet_project("gpuwm_kernels_startup-2.9.12.tar.gz") == "gpuwm-kernels-startup"
    assert capacity.packet_project("gpuwm-2.9.12-py3-none-win_amd64.whl") == "gpuwm"
    assert capacity.packet_project("gpuwm_data-2.9.12.tar.gz") == "gpuwm-data"
    for name in ("gpuwm_bridges-2.9.12.zip", "other-1.0.whl", "sub/gpuwm-1.0.whl", "gpuwm"):
        with pytest.raises(capacity.CapacityError, match="unexpected PyPI packet filename"):
            capacity.packet_project(name)

    sealed(tmp_path)
    plan, _ = capacity.packet_plan(tmp_path)
    assert {project: [r["filename"] for r in rows] for project, rows in plan.items()} == {
        "gpuwm": ["gpuwm-9.0.whl"], "gpuwm-data": ["gpuwm_data-9.0.whl"],
        "gpuwm-kernels": ["gpuwm_kernels-9.0-py3-none-any.whl", "gpuwm_kernels-9.0.tar.gz"],
        "gpuwm-kernels-startup": ["gpuwm_kernels_startup-9.0-py3-none-any.whl", "gpuwm_kernels_startup-9.0.tar.gz"]}

    # A project PyPI does not have yet (HTTP 404) is an empty inventory, not an unreadable one.
    empty = capacity.empty_project("gpuwm-kernels")
    assert capacity.inventory(empty, "gpuwm-kernels") == {}
    receipt = capacity.check_packet(tmp_path, fetch=new_projects(recorded).__getitem__,
                                    limits=dict.fromkeys(capacity.PROJECTS, 20_000_000_000))
    by_project = {r["project"]: r for r in receipt["projects"]}
    assert [r["project"] for r in receipt["projects"]] == list(capacity.PROJECTS)
    assert by_project["gpuwm-kernels"]["status"] == "PASS"
    assert (by_project["gpuwm-kernels"]["current_bytes"], by_project["gpuwm-kernels"]["existing_files"],
            by_project["gpuwm-kernels"]["releases"]) == (0, 0, 0)
    assert by_project["gpuwm-kernels-startup"]["new_bytes"] == sum(
        len(b"sealed bytes " + n.encode()) for n in PACKET_FILES if n.startswith("gpuwm_kernels_startup-"))
    line = capacity.storage_line(capacity.headroom("gpuwm-kernels", empty, 10 * 1024 ** 3))
    assert line == "PyPI gpuwm-kernels: 0.00 of 10.74 GB stored (0 releases); no release to price the next one by"


def test_fetch_treats_404_as_empty_and_other_http_errors_refuse(monkeypatch):
    import io
    import urllib.error

    def opener(code):
        def urlopen(url, timeout):
            raise urllib.error.HTTPError(url, code, "status", {}, io.BytesIO(b""))
        return urlopen

    monkeypatch.setattr(capacity.urllib.request, "urlopen", opener(404))
    assert capacity.fetch_project("gpuwm-kernels-startup") == capacity.empty_project("gpuwm-kernels-startup")
    monkeypatch.setattr(capacity.urllib.request, "urlopen", opener(503))
    with pytest.raises(capacity.CapacityError, match="cannot read complete public JSON inventory"):
        capacity.fetch_project("gpuwm-kernels-startup")
    with pytest.raises(capacity.CapacityError, match="unknown project"):
        capacity.fetch_project("gpuwm-bridges")


def test_packet_missing_a_project_refuses_and_names_it(tmp_path):
    sealed(tmp_path, ("gpuwm-9.0.whl", "gpuwm_data-9.0.whl"))
    with pytest.raises(capacity.CapacityError, match="missing gpuwm-kernels, gpuwm-kernels-startup"):
        capacity.packet_plan(tmp_path)
    with pytest.raises(capacity.CapacityError, match="unknown project gpuwm-bridges"):
        capacity.check_plan({"gpuwm-bridges": []}, fetch=lambda p: pytest.fail("network was attempted"),
                            limits=dict.fromkeys(capacity.PROJECTS, 1))
