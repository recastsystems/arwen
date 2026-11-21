"""MSKF's negative convective ice reaches RTE+RRTMGP in WRF's order.

The defect: every 12 km Morrison + multiscale-KF run through ``gpuwm go``
(GFS 21 May 2024 12Z, 40N 93W) stopped after a few hours with "RRTMGP input
validation failed: qi is non-finite or negative".  The negative ice is WRF's
own: at a shallow plume's first freezing level DTFRZNEW
(module_cu_mskf.F:7091-7093) sublimates ``qs - qu`` of ice that is not there,
and WRF writes ``qi_KF = -5.48e-6``.  WRF's radiation driver adds
``qi_cu*cldfra_cu`` (module_radiation_driver.F:1345) and RRTMG then reads
``max(0, qi)`` (module_ra_rrtmg_lw.F:12091, module_ra_rrtmg_sw.F:10692).  The
RTE+RRTMGP adapter merged the cumulus cloud after its own reads and refused
the sum.

The fixture is that column from the failing run, replayed through the
byte-unmodified WRF v4.7.1 MSKF_CPS (tools/mskf_wrf461_oracle/real_column).
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = Path("tests/data/oracles/mskf/negative_ice_column.npz")
F = np.float32
_F3 = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten",
       "rucuten", "rvcuten", "cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu",
       "udr_kf", "ddr_kf", "uer_kf", "der_kf")
_F2 = ("raincv", "pratec", "nca", "cutop", "cubot", "timec_kf")


def test_fixture_is_wrfs_own_negative_shallow_ice():
    fx = np.load(FIXTURE)
    qi = fx["wrf_qi_cu"]
    k = int(np.argmin(qi))
    assert qi[k] < F(-5e-6)
    assert fx["wrf_cldfra_sh"][k] > 0 and fx["wrf_cldfra_dp"].max() == 0
    # The shallow plume: a held count of CUDT*60 and no surface rain.
    assert fx["wrf_nca"][0] == F(300.0) and fx["wrf_raincv"][0] == 0


@pytest.fixture
def wrf_exact_numerics(monkeypatch):
    import gpuwm.numerics as numerics
    import gpuwm.wrf_exact as exact
    from gpuwm.core.kernels import no_ftz
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    monkeypatch.setattr(exact, "ENABLED", True)
    monkeypatch.setattr(numerics, "_mode", "strict")
    no_ftz.get_kernel_no_ftz.cache_clear()
    no_ftz.load_module_no_ftz.cache_clear()
    yield
    no_ftz.get_kernel_no_ftz.cache_clear()
    no_ftz.load_module_no_ftz.cache_clear()


@requires_gpu
def test_strict_port_writes_wrfs_negative_ice_word_for_word(wrf_exact_numerics):
    import cupy as cp
    from gpuwm.core import mskf

    fx = np.load(FIXTURE)
    nz = fx["in_t"].size
    shape = (nz, 1, 1)
    inputs = {n: cp.asarray(fx["in_" + n].reshape(shape))
              for n in ("u", "v", "t", "qv", "p", "rho", "pi", "dz8w")}
    inputs["w0avg"] = cp.asarray(fx["in_w0avg_pre"].reshape(shape))
    mskf.update_w0avg(inputs["w0avg"], cp.asarray(fx["in_w"].reshape(nz + 1, 1, 1)),
                      dt=60.0, cudt_minutes=5.0, stepcu=5, adaptive=False)
    for n in ("hfx", "ust", "pblh", "zol"):
        inputs[n] = cp.asarray(fx["in_" + n].reshape(1, 1))
    outputs = {n: cp.full(shape, -7777.0, cp.float32) for n in mskf.OUTPUT_3D}
    outputs.update({n: cp.full((1, 1), -7777.0, cp.float32) for n in mskf.OUTPUT_2D})
    outputs["nca"] = cp.full((1, 1), -100.0, cp.float32)
    outputs["cu_act"] = cp.zeros((1, 1), cp.int32)
    status = mskf.launch_mskf(inputs, outputs, dx=12000.0, dt=60.0, cudt_minutes=5.0,
                              phase=mskf.MskfPhase(False, True, True, True),
                              aercu_opt=0, kf_edrates=1)
    assert status == 0
    got = {n: cp.asnumpy(outputs[n]).ravel() for n in _F3 + _F2}
    got["w0avg"] = cp.asnumpy(inputs["w0avg"]).ravel()
    for name in _F3 + _F2 + ("w0avg",):
        np.testing.assert_array_equal(got[name].view("u4"), fx["wrf_" + name].view("u4"),
                                      err_msg=name)


def _radiation(qi_cu):
    """One RTE+RRTMGP call on the fixture column, MSKF's cloud fed back."""
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    fx = np.load(FIXTURE)
    p, t, qv = (fx["in_" + n].astype(F) for n in ("p", "t", "qv"))
    nz = p.size
    plev = np.empty(nz + 1, F)
    plev[0] = p[0] + F(0.5) * (p[0] - p[1])
    plev[1:nz] = F(0.5) * (p[:-1] + p[1:])
    plev[nz] = F(5000.0)
    shape = (nz, 1, 1)
    col = lambda a, n=nz: cp.asarray(np.asarray(a, F).reshape(n, 1, 1))
    exner = fx["in_pi"].astype(F)
    zeros = cp.zeros(shape, cp.float32)
    atmosphere = {"pressure": col(p), "p_interface": col(plev, nz + 1),
                  "temperature": col(t), "theta": col(t / exner), "exner": col(exner),
                  "qv": col(qv), "qc": zeros, "qi": zeros}
    feedback = {"cldfra_dp": col(fx["wrf_cldfra_dp"]), "cldfra_sh": col(fx["wrf_cldfra_sh"]),
                "qc_cu": col(fx["wrf_qc_cu"]), "qi_cu": col(qi_cu), "qs_cu": zeros,
                "efcs": zeros, "efis": zeros, "efss": zeros}
    atmosphere["cumulus_feedback"] = feedback
    fields = {"tsk": cp.full((1, 1), float(t[0]) + 2.0, cp.float32),
              "albedo": cp.full((1, 1), 0.17, cp.float32),
              "emiss": cp.full((1, 1), 0.95, cp.float32),
              "xland": cp.ones((1, 1), cp.float32)}
    state = SimpleNamespace(elapsed_seconds=21600.0, qc=zeros, qi=zeros, qs=zeros, qr=zeros,
                            nc=zeros, nr=zeros, ni=zeros, ns=zeros,
                            effc=cp.full(shape, 10.0, cp.float32),
                            effi=cp.full(shape, 25.0, cp.float32),
                            effs=cp.full(shape, 100.0, cp.float32))
    radiation = RRTMGPRadiation(datetime(2024, 5, 21, 12), cp.full((1, 1), 41.0, cp.float32),
                                cp.full((1, 1), -101.0, cp.float32))
    result = radiation(atmosphere=atmosphere, fields=fields, state=state,
                       cfg=SimpleNamespace(mp_physics=10, cu_physics=11, aercu_opt=0,
                                           cu_rad_feedback=True, icloud=1, dt=60.0,
                                           radt=12.0, radt_minutes=12.0, bl_pbl_physics=1,
                                           icloud_bl=0))
    return {n: cp.asnumpy(getattr(result, n)) for n in
            ("swdown", "glw", "rthratensw", "rthratenlw")}


@requires_gpu
def test_rrtmgp_reads_mskf_negative_ice_as_wrf_rrtmg_does():
    fx = np.load(FIXTURE)
    qi_cu = fx["wrf_qi_cu"]
    assert qi_cu.min() < 0
    # Before: ValueError "RRTMGP input validation failed: qi is non-finite or
    # negative".  After: WRF's max(0, qi + qi_cu*cldfra_cu), so the column
    # radiates exactly as with that layer's ice at zero.
    negative = _radiation(qi_cu)
    floored = _radiation(np.maximum(qi_cu, F(0)))
    for name in negative:
        assert np.isfinite(negative[name]).all(), name
        np.testing.assert_array_equal(negative[name], floored[name], err_msg=name)
