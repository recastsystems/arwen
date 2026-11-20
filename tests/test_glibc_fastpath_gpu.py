"""The float-float fast paths (kernels/glibc_fastpath.cuh) never change a word.

The routed header copies -- gfk_* (glibc_flt32.cuh) and mynn_glibc_*
(mynn_libm.cuh) -- are each compiled twice from the kernel tree's own text:
routed exactly as the loader routes them with GPUWM_GLIBC_FASTPATH=1, and
unchanged (the original binary64 copy), and the two are compared

* on all 2^32 binary32 arguments for logf and expf;
* on all 2^32 x for every constant exponent the kernel sources pass to a
  routed powf copy, and on 2^30 random (x, y) pairs in three y ranges;

in every compile mode the routed units use.  Declared tolerance: 0 differing
words.  The copies inside larger units (rlw_* and rsw_* in rrtmg_lw/sw.cu,
r_* in noahmp_fluxprep/leaves.cu) share the same fast path; their check is
the forecast's output words, identical with and without it.  A second check proves the float pair tables are glibc's binary64
tables split exactly as the header states (hi = float(d), lo = float(d - hi)).
"""
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

KDIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"

MODES = {
    "fmad=false ftz=false": ("--fmad=false", "--ftz=false"),
    "fmad=true ftz=false": ("--fmad=true", "--ftz=false"),
    "fmad=true ftz=true": ("--fmad=true", "--ftz=true"),
}

#: routed powf copies whose call sites are scanned for constant exponents
POW_COPIES = ("gfk_pow", "mynn_glibc_powf", "rlw_pow", "r_pow")

DRIVER = r'''
extern "C" __global__ void chk(int which, unsigned long long base, float yc, int mode,
                               unsigned int* out) {
    unsigned long long t = base + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long z = t * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
    unsigned int iy = (unsigned int)(z >> 32);
    float x = __uint_as_float((unsigned int)t), y = yc;
    if (which == 3) {                       // random (x, y): positive x of every magnitude
        x = __uint_as_float((unsigned int)z & 0x7fffffffu);
        if (mode == 0) y = __uint_as_float(iy);
        else if (mode == 1) y = ((float)(iy >> 8) / 16777216.0f) * 8.0f - 4.0f;
        else y = ((float)(iy >> 8) / 16777216.0f) * 200.0f - 100.0f;
    }
    float r = which == 0 ? COPY_LOGF(x) : which == 1 ? COPY_EXPF(x) : COPY_POWF(x, y);
    out[t - base] = __float_as_uint(r);
}
'''


#: header -> its logf, expf, powf copies
FAMILIES = {"glibc_flt32.cuh": ("gfk_log", "gfk_exp", "gfk_pow"),
            "mynn_libm.cuh": ("mynn_glibc_logf", "mynn_glibc_expf", "mynn_glibc_powf")}


def _module(cp, options, routed, header):
    names = FAMILIES[header]
    from gpuwm.core.kernels import route_glibc_copies
    copy = (KDIR / header).read_text()
    text = ("typedef float real;\n"
            + ((KDIR / "glibc_fastpath.cuh").read_text() + route_glibc_copies(copy, header) if routed else copy)
            + "".join(f"#define COPY_{f} {n}\n" for f, n in zip(("LOGF", "EXPF", "POWF"), names))
            + DRIVER)
    return cp.RawModule(code=text, options=("-std=c++17",) + options).get_function("chk")


def constant_exponents():
    """Every literal exponent a kernel source passes to a routed powf copy."""
    lit = re.compile(r"\b(" + "|".join(POW_COPIES) + r")\s*\(\s*[^,()]+(?:\([^()]*\))?\s*,\s*"
                     r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)f?\s*\)")
    found = set()
    for path in KDIR.glob("*.cu*"):
        for m in lit.finditer(path.read_text(errors="replace")):
            found.add(float(np.float32(float(m.group(2)))))
    return sorted(found)


def _differing(cp, header, which, mode_opts, yc=0.0, mode=0, total=1 << 32):
    fast, orig = _module(cp, mode_opts, True, header), _module(cp, mode_opts, False, header)
    chunk = 1 << 28
    a = cp.empty(chunk, cp.uint32)
    b = cp.empty(chunk, cp.uint32)
    bad = 0
    for base in range(0, total, chunk):
        for k, buf in ((fast, a), (orig, b)):
            k((chunk // 256,), (256,), (np.int32(which), np.uint64(base), np.float32(yc), np.int32(mode), buf))
        bad += int(cp.count_nonzero(a != b))
    return bad


@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("header", list(FAMILIES))
@pytest.mark.parametrize("mode", list(MODES))
def test_logf_expf_every_argument(mode, header):
    cp = pytest.importorskip("cupy")
    assert _differing(cp, header, 0, MODES[mode]) == 0
    assert _differing(cp, header, 1, MODES[mode]) == 0


@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("header", list(FAMILIES))
@pytest.mark.parametrize("mode", list(MODES))
def test_powf_every_x_for_each_constant_exponent(mode, header):
    cp = pytest.importorskip("cupy")
    exponents = constant_exponents()
    assert exponents, "no constant exponents found"
    bad = {y: _differing(cp, header, 2, MODES[mode], yc=y) for y in exponents}
    assert not any(bad.values()), {y: n for y, n in bad.items() if n}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("yrange", [0, 1, 2], ids=["any-bits", "abs-lt-4", "abs-lt-100"])
@pytest.mark.parametrize("header", list(FAMILIES))
def test_powf_random_pairs(mode, yrange, header):
    cp = pytest.importorskip("cupy")
    assert _differing(cp, header, 3, MODES[mode], mode=yrange, total=1 << 30) == 0


def _c_array(text, name):
    body = re.search(name + r"\s*\[\s*\d*\s*\]\s*=\s*\{(.*?)\}", text, re.S).group(1)
    return [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]


def _f32(v):
    return struct.unpack("f", struct.pack("f", float.fromhex(v.rstrip("f")) if "0x" in v else float(v.rstrip("f"))))[0]


def test_float_pair_tables_are_glibc_split_exactly():
    """hi = float(d) and lo = float(d - hi) for glibc's binary64 logf tables."""
    fast = (KDIR / "glibc_fastpath.cuh").read_text()
    exact = (KDIR / "glibc_flt32.cuh").read_text()
    for pair, ref in (("GFP_LOGF_INVC", "GFK_LOGF_INVC"), ("GFP_LOGF_LOGC", "GFK_LOGF_LOGC")):
        d = [float.fromhex(v) for v in _c_array(exact, ref)]
        hi = [_f32(v) for v in _c_array(fast, pair + "_H")]
        lo = [_f32(v) for v in _c_array(fast, pair + "_L")]
        assert len(d) == len(hi) == len(lo) == 16
        for dv, h, l in zip(d, hi, lo):
            assert h == float(np.float32(dv)), (pair, dv)
            assert l == float(np.float32(dv - h)), (pair, dv)
