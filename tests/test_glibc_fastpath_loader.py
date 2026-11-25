"""GPUWM_GLIBC_FASTPATH: the loader's routing of the glibc libm copies.

Off (the default) leaves every composed source untouched; on routes exactly
the modules that compose a copy, never Noah; a copy the router cannot find
fails closed instead of compiling unrouted.
"""
import pytest

from gpuwm.core import kernels as K

ROUTED = ("mynn_pbl", "mynn_surface", "noahmp_fluxprep", "noahmp_leaves", "kfcup", "gd")
NOAH = ("noah", "noah_mosaic")


@pytest.mark.parametrize("name", ROUTED + NOAH)
def test_default_composes_no_fast_path(monkeypatch, name):
    monkeypatch.delenv("GPUWM_GLIBC_FASTPATH", raising=False)
    assert "glibc_fastpath_" not in K.module_source(name)


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
def test_only_the_value_1_enables_it(monkeypatch, value):
    monkeypatch.setenv("GPUWM_GLIBC_FASTPATH", value)
    assert not K.glibc_fastpath_enabled("mynn_pbl")
    assert "glibc_fastpath_" not in K.module_source("mynn_pbl")


@pytest.mark.parametrize("name", ROUTED)
def test_enabled_routes_each_copy_once(monkeypatch, name):
    monkeypatch.setenv("GPUWM_GLIBC_FASTPATH", "1")
    source = K.module_source(name)
    assert source.count("#define GLIBC_FASTPATH_CUH") == 1
    assert source.count("GLIBC_FASTPATH_UNSURE) return gfp_r;") >= 2


@pytest.mark.parametrize("name", NOAH)
def test_noah_keeps_the_original_copies_when_enabled(monkeypatch, name):
    monkeypatch.setenv("GPUWM_GLIBC_FASTPATH", "1")
    assert not K.glibc_fastpath_enabled(name)
    assert "glibc_fastpath_" not in K.module_source(name)


def test_a_copy_the_router_cannot_find_fails_closed():
    text = (K._KDIR / "glibc_flt32.cuh").read_text(encoding="utf-8")
    broken = text.replace("gfk_exp(float", "gfk_exp_renamed(float", 1)
    assert broken != text
    with pytest.raises(RuntimeError, match="gfk_exp not found"):
        K.route_glibc_copies(broken, "glibc_flt32.cuh")


def test_routing_inserts_exactly_one_early_out_per_copy():
    for filename, copies in K.GLIBC_FASTPATH_COPIES.items():
        text = (K._KDIR / filename).read_text(encoding="utf-8")
        routed = K.route_glibc_copies(text, filename)
        assert routed.count("GLIBC_FASTPATH_UNSURE) return gfp_r;") == len(copies), filename
        # removing the inserted lines gives the file back byte for byte
        lines = [l for l in routed.split("\n")
                 if not l.startswith("    { float gfp_r = glibc_fastpath_")
                 and l != "      if (__float_as_uint(gfp_r) != GLIBC_FASTPATH_UNSURE) return gfp_r; }"]
        assert "\n".join(lines) == text, filename
