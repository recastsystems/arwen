"""The resident Noah-MP route publishes the census a seam caller checks.

THE BREAKAGE THIS PREVENTS (gpuwm 3.0.4, and 3.0.5 at e8d7c8098d): every hex
forecast on the ArWen physics route with Noah-MP stopped at step 0.  The
resident (default) surface route published, through the seam's
``last_noahmp_census``:

* a ``glacier_path`` on every step, also with no glacier column, while the
  host route and the seam's docstring name it only when glacier columns ran
  (hex: "census/provenance differs from executed layout authority");
* its column counts as 0-d device arrays, which the step receipt (JSON)
  cannot hold ("Object of type ndarray is not JSON serializable");
* on a mesh with glacier columns, the token "noahmp-glacier/resident-cuda",
  which is not the CUDA glacier unit's token that the caller checks.

The counts stay on the device inside the step (the step makes no host read and
stays CUDA-graph capturable, test_noahmp_surface_device); the seam property is
where they become the receipt.  The existing seam test pins the host layout
route (test_noahmp_glacier_gpu), so it never saw the resident route's census.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

CUDA_GLACIER = "noahmp-glacier/cuda (gpuwm/core/kernels/noahmp_glacier.cu)"


def _executed_layout(classification):
    """What a seam caller builds from the classification receipt, as hex's
    ArWen route does (noahmp_routing_census)."""
    expected = {"land": classification["sflx_land_columns"],
                "water": classification["open_water_columns"],
                "sea_ice": classification["sea_ice_columns"],
                "glacier": classification["glacier_columns"]}
    if expected["glacier"]:
        expected["glacier_path"] = CUDA_GLACIER
    return expected


def test_the_published_census_is_a_receipt():
    from gpuwm.core.noahmp_runtime import published_noahmp_census

    device_like = {"land": np.array(3), "water": np.array(1),
                   "sea_ice": np.array(0), "glacier": np.array(0),
                   "glacier_path": CUDA_GLACIER}
    published = published_noahmp_census(device_like)
    assert published == {"land": 3, "water": 1, "sea_ice": 0, "glacier": 0}
    assert all(type(value) is int for value in published.values())
    json.dumps(published)
    device_like["glacier"] = np.array(2)
    assert published_noahmp_census(device_like)["glacier_path"] == CUDA_GLACIER
    assert published_noahmp_census(None) is None


@pytest.mark.parametrize("glacier", [True, False], ids=["glacier", "no-glacier"])
def test_the_resident_route_census_is_the_receipt_a_caller_checks(glacier):
    pytest.importorskip("cupy")
    from test_noahmp_glacier_gpu import ISICE, SURFACE, _phase1, _profile, _seam

    overrides = {}
    if not glacier:
        overrides["ivgtyp"] = np.where(SURFACE["ivgtyp"] == ISICE, 10,
                                       SURFACE["ivgtyp"]).astype(np.int32)
    seam = _seam(**overrides)
    # the default route: the resident (device) layout
    assert seam._driver.noahmp_params.uses_device_surface_layout()
    _phase1(seam, _profile())
    census = seam.last_noahmp_census
    json.dumps(census)  # the step receipt
    assert census == _executed_layout(seam.surface_classification)
    assert all(type(census[key]) is int
               for key in ("land", "water", "sea_ice", "glacier"))
    assert ("glacier_path" in census) is glacier
