"""The same-words speed levers taken from jacobbieker/woof (Apache-2.0), held
in place.  Each guards a measured cost on an RTX 5090 (HRRR 21 May
2024 18Z, 1 h, through `gpuwm go`, 0 history words different from 3.0.4):

- #226: RRTMG legacy's forecast call reads its device guards once at its end
  (without it every ra=4 call waits on the host once per guard flag);
- #369: nwp_hail skips pow(d, mu) when the size distribution has mu == 0;
- #529: MSKF keeps 256 columns per SM in flight (64 left half of each SM
  idle: 21.6 against 15.1-15.4 s for steps 10-240 of a cu_physics=11 hour);
- #219: f64cr's error-free transformations are rounded intrinsics, so a unit
  compiled with --fmad=true cannot fuse them.
"""
from __future__ import annotations

import re
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"


def test_legacy_rrtmg_call_defers_its_device_guards():
    from gpuwm.core import rrtmg_legacy
    call = rrtmg_legacy.RRTMGLegacyRadiation.__call__
    assert getattr(call, "__wrapped__", None) is not None, (
        "RRTMGLegacyRadiation.__call__ lost @_with_deferred_device_checks")
    assert call.__code__.co_name == "wrapper"


def test_nwp_hail_skips_the_power_for_zero_shape():
    text = (KERNELS / "nwp_hail.cu").read_text(encoding="utf-8")
    assert "const bool shaped=mu!=0.f;" in text
    assert "(shaped?nh_pow64(d,(double)mu):1.)" in text


def test_mskf_tile_fills_both_resident_blocks():
    from gpuwm.core import mskf
    assert mskf.tile_columns(10**6, 170) == 170 * 256
    assert mskf.tile_columns(100, 170) == 100
    # the aerosol path keeps its own base
    assert mskf.tile_columns(10**6, 10, aerosol=True) in (10 * 64, 10 * 128)


def test_f64cr_error_free_transformations_are_rounded_intrinsics():
    text = (KERNELS / "f64cr.cuh").read_text(encoding="utf-8")
    assert "#define WCR_ADD(a, b) __dadd_rn((a), (b))" in text
    assert "#define WCR_MUL(a, b) __dmul_rn((a), (b))" in text
    body = text[text.index("WCR_HD wcr_dd wcr_2sum("):text.index("WCR_HD wcr_dd wcr_neg(")]
    assert not re.search(r"[^_A-Za-z]\w+\s*[-+*]\s*\w+", body.replace("WCR_FMA(a, b, -p)", "")), body
