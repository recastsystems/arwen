"""AFWA precipitation type follows WRF's literal cloud-top test.

WRF 4.7.1 phys/module_diag_afwa.F:1888 reads
``IF (rh(i,k,j) .gt. 80 .and. saturation(i,j) .eqv. .false.)``.  Fortran
binds ``.and.`` tighter than ``.eqv.``, so the test is
``(rh > 80 .and. saturation) .eqv. .false.``: the cloud top moves down at
every level that is not a saturated continuation, not only where RH first
passes 80%.  The kernel used ``rh > 80 && !saturation``, which types a cold
column with a dry layer under a saturated top as snow where WRF gives
freezing rain.  The reference below transcribes lines 1857-1980 as WRF
compiles them.
"""
import numpy as np
import pytest
from conftest import requires_gpu

CCN, MELT = np.float32(264.15), np.float32(50.0)


def _rh(p, t, qv):
    q = qv / (1.0 + qv)
    qs = 379.90516 / p * np.exp(17.2693882 * (t - 273.16) / (t - 35.86))
    return float(np.clip(100.0 * q / qs, 1.0, 100.0))


def _qv(p, t, rh):
    qs = 379.90516 / p * np.exp(17.2693882 * (t - 273.16) / (t - 35.86))
    q = rh / 100.0 * qs
    return q / (1.0 - q)


def _wrf_precip_type(t, p, q, z, dz, ht, t2, sw):
    """module_diag_afwa.F precip_type_diagnostics for one column, k=0 at the
    surface; returns precip_type (0 none, 1 rain, 2 snow, 3 ice) and mod_2m_tmp."""
    nz = len(t)
    mod = t2 + sw / 100.0
    maxtmp, melt, ptype, saturation = mod, 0.0, 0, False
    cloud_top_tmp = np.nan
    if mod > 275.15:
        return 1, mod
    top = nz - 1
    for k in range(nz - 1, -1, -1):
        if z[k] - ht > 0.0:
            maxtmp = max(maxtmp, t[k])
            rh = _rh(p[k], t[k], q[k])
            if ((rh > 80) and saturation) == False:  # noqa: E712 (Fortran .eqv. .false.)
                cloud_top_tmp, top, saturation, ptype = t[k], k, True, 2
            if ((rh <= 70) and saturation) == True:  # noqa: E712
                saturation = False
    if cloud_top_tmp <= CCN and maxtmp <= 273.15:
        ptype = 2
    for k in range(top, -1, -1):
        if z[k] - ht > 0.0:
            if cloud_top_tmp == t[k] and cloud_top_tmp > CCN:
                ptype = 1
            if ptype in (2, 3) and t[k] > 273.15:
                melt += 9.8 * (((t[k] - 273.15) / 273.15) * dz[k])
                if melt > MELT:
                    ptype, melt = 1, 0.0
            if t[k] <= 273.15 and melt > MELT / 4.0 and ptype in (2, 3):
                ptype, melt = 3, 0.0
            if ptype == 1 and t[k] <= CCN:
                ptype = 3
    return ptype, mod


def _columns():
    """(t, rh) columns, surface first.  The first is the defect's own case: a
    saturated cold top over a dry layer whose lowest level is warmer than the
    nucleation temperature."""
    cold_top_dry_below = ([268.0, 266.0, 262.0, 256.0, 250.0, 245.0, 242.0, 240.0],
                          [40.0, 40.0, 40.0, 40.0, 40.0, 95.0, 95.0, 95.0])
    rng = np.random.default_rng(471)
    columns = [cold_top_dry_below]
    for _ in range(255):
        surface = rng.uniform(262.0, 280.0)
        lapse = rng.uniform(2.0, 7.0)
        warm_nose = rng.uniform(0.0, 8.0) * (rng.random() < 0.5)
        t = [surface - lapse * k + warm_nose * np.exp(-((k - 2) / 1.2) ** 2) for k in range(8)]
        rh = list(rng.choice([40.0, 75.0, 95.0], size=8))
        columns.append((t, rh))
    return columns


@pytest.mark.gpu
@requires_gpu
def test_ptype_matches_wrf_literal_cloud_top_test():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    columns = _columns()
    n, nz = len(columns), 8
    t = np.zeros((nz, n), np.float32); p = np.zeros_like(t); q = np.zeros_like(t)
    z = np.zeros_like(t); dz = np.full_like(t, 250.0)
    for c, (tc, rhc) in enumerate(columns):
        for k in range(nz):
            p[k, c] = 95000.0 - 6000.0 * k
            t[k, c] = tc[k]
            q[k, c] = _qv(float(p[k, c]), float(t[k, c]), rhc[k])
            z[k, c] = 125.0 + 250.0 * k
    ht = np.zeros(n, np.float32)
    t2 = np.array([tc[0] - 0.5 for tc, _ in columns], np.float32)
    t2[0] = 271.0
    sw = np.zeros(n, np.float32)
    precip = np.full(n, 1.5, np.float32)
    out = [cp.zeros(n, cp.float32) for _ in range(5)]
    args = [cp.asarray(a) for a in (t, p, q, z, dz, ht, t2, sw, precip)]
    get_kernel('afwa_diagnostics', 'afwa_ptype')(
        ((n + 127) // 128,), (128,),
        (*args, *out, CCN, MELT, np.int32(nz), np.int32(n)))
    rain, snow, ice, frz, snowfall = (cp.asnumpy(a) for a in out)
    got = np.select([ice > 0, snow > 0, rain > 0, frz > 0], [3, 2, 1, 1], 0)
    want, liquid_warm = [], []
    for c in range(n):
        kind, mod = _wrf_precip_type(t[:, c].astype(np.float64), p[:, c], q[:, c],
                                     z[:, c], dz[:, c], 0.0, float(t2[c]), 0.0)
        want.append(kind)
        liquid_warm.append(mod > 273.15)
    want = np.array(want)
    assert want[0] == 1 and not liquid_warm[0]  # WRF: freezing rain
    assert got[0] == 1 and frz[0] == precip[0] and snow[0] == 0.0
    assert np.array_equal(got, want), np.flatnonzero(got != want)
    assert np.array_equal(rain > 0, (want == 1) & np.array(liquid_warm))
    assert np.array_equal(frz > 0, (want == 1) & ~np.array(liquid_warm))
    assert np.array_equal(snowfall > 0, want == 2)
