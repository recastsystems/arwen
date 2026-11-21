"""Every Kain-Fritsch family kernel floors the downdraft-capped rain flux.

When the downdraft cap sets DDINC = TRPPT/TDER, the float64 answer for
TRPPT - TDER is 0, but binary32 TDER*DDINC can round above TRPPT and the
bare difference is minus a few ulp.  PRATEC is then negative and RAINC
accumulates below zero.  kf.cu has carried the floor since d547dc8c3; MSKF
(cu_physics = 11) and KF-CuP (cu_physics = 10) carried the bare line
(jacobbieker/woof pull request 375 fixed MSKF; KF-CuP is the same line).
This fails if any of the three loses the floor.
"""

import re
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"


@pytest.mark.parametrize("unit, flux", [("kf.cu", "pptflx"), ("mskf.cu", "PPTFLX"),
                                        ("kfcup.cu", "PPTFLX")])
def test_the_capped_rain_flux_is_floored_at_zero(unit, flux):
    text = (KERNELS / unit).read_text(encoding="utf-8")
    trppt, tder = ("trppt", "tder") if flux.islower() else ("TRPPT", "TDER")
    bare = re.findall(rf"^\s*{flux}\s*=\s*{trppt}\s*-\s*{tder}\s*;", text, re.M)
    floored = re.findall(
        rf"^\s*{flux}\s*=\s*fmaxf\(\s*{trppt}\s*-\s*{tder}\s*,\s*0\.0f\s*\)\s*;", text, re.M)
    assert not bare, f"{unit}: bare {trppt} - {tder} can go below zero"
    assert len(floored) == 1, f"{unit}: the floored rain flux is missing"
