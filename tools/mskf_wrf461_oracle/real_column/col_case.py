"""Build one-column MSKF cases from a probe dump and run them on the GPU port.

python col_case.py DUMP.npz OUTDIR J,I [J,I ...]

For each column: writes OUTDIR/col_f32.bin (the Fortran driver's input, one
record per column) and OUTDIR/gpu_out.npz (the port's outputs for the same
inputs). The pre-call W0AVG and interface W are constructed so that WRF's
own fixed-step W0AVG update (done inside MSKF_CPS) and the port's
update_w0avg kernel see the same words: W0AVG_pre = the dumped (post-update)
mean, W chosen with 0.5*(W(k)+W(k+1)) = that mean.
"""
import sys
import numpy as np

dump, outdir = sys.argv[1], sys.argv[2]
cols = [tuple(int(x) for x in s.split(",")) for s in sys.argv[3:]]
d = np.load(dump)
nz = d["in_t"].shape[0]
f32 = np.float32

LEV = ("u", "v", "t", "qv", "p", "rho", "pi", "dz8w")
recs = []
cases = []
for j, i in cols:
    lev = {n: d["in_" + n][:, j, i].astype(f32) for n in LEV}
    th = (lev["t"] / lev["pi"]).astype(f32)
    target = d["in_w0avg"][:, j, i].astype(f32)
    w = np.zeros(nz + 1, f32)
    for k in range(nz):
        w[k + 1] = f32(f32(2.0) * target[k]) - w[k]
    sfc = [f32(d["in_" + n][j, i]) for n in ("hfx", "ust", "pblh", "zol")]
    rec = np.concatenate([lev["u"], lev["v"], lev["t"], th, lev["qv"], lev["p"], lev["rho"],
                          lev["pi"], lev["dz8w"], target, w, np.array(sfc, f32)]).astype(f32)
    recs.append(rec)
    cases.append((lev, th, target, w, sfc))
np.concatenate(recs).astype("<f4").tofile(f"{outdir}/col_f32.bin")
with open(f"{outdir}/ncol.txt", "w") as fh:
    fh.write(f"{len(cols)} {nz}\n")

# GPU port, one column per launch, the forecast's settings for this case.
import cupy as cp
from gpuwm.core import mskf

out_all = {}
for c, (lev, th, target, w, sfc) in enumerate(cases):
    shape = (nz, 1, 1)
    inputs = {n: cp.asarray(lev[n].reshape(shape)) for n in LEV}
    inputs["w0avg"] = cp.asarray(target.reshape(shape))
    mskf.update_w0avg(inputs["w0avg"], cp.asarray(w.reshape(nz + 1, 1, 1)), dt=60.0,
                      cudt_minutes=5.0, stepcu=5, adaptive=False)
    for n, v in zip(("hfx", "ust", "pblh", "zol"), sfc):
        inputs[n] = cp.full((1, 1), v, cp.float32)
    outputs = {n: cp.full(shape, -7777.0, cp.float32) for n in mskf.OUTPUT_3D}
    outputs.update({n: cp.full((1, 1), -7777.0, cp.float32) for n in mskf.OUTPUT_2D})
    outputs["nca"] = cp.full((1, 1), -100.0, cp.float32)
    outputs["cu_act"] = cp.zeros((1, 1), cp.int32)
    st = mskf.launch_mskf(inputs, outputs, dx=12000.0, dt=60.0, cudt_minutes=5.0,
                          phase=mskf.MskfPhase(False, True, True, True), aercu_opt=0,
                          kf_edrates=1)
    for n, v in outputs.items():
        out_all[f"c{c}_{n}"] = cp.asnumpy(v).ravel()
    out_all[f"c{c}_status"] = np.array([st])
    out_all[f"c{c}_w0avg"] = cp.asnumpy(inputs["w0avg"]).ravel()
np.savez(f"{outdir}/gpu_out.npz", **out_all)
print("gpu done", len(cases), "columns, nz", nz)
