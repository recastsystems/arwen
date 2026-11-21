"""Pack one replayed column and WRF's answer into a test fixture.

python make_fixture.py RUN_DIR COLUMN OUT.npz

RUN_DIR holds col_f32.bin / ncol.txt (written by col_case.py) and wrf_out.bin
(written by run_col, the byte-unmodified WRF v4.7.1 MSKF_CPS).
"""
import sys

import numpy as np

run, column, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ncol, nz = (int(x) for x in open(f"{run}/ncol.txt").read().split())
LEV = ("u", "v", "t", "th", "qv", "p", "rho", "pi", "dz8w", "w0avg_pre")
F3 = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten", "rucuten",
      "rvcuten", "cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu", "udr_kf", "ddr_kf", "uer_kf",
      "der_kf", "w0avg")
F2 = ("raincv", "pratec", "nca", "cutop", "cubot", "timec_kf")
recl = 10 * nz + (nz + 1) + 4
rec = np.fromfile(f"{run}/col_f32.bin", "<f4").reshape(ncol, recl)[column]
per = len(F3) * nz + len(F2)
wrf = np.fromfile(f"{run}/wrf_out.bin", "<f4").reshape(ncol, per)[column]
arrays = {f"in_{n}": rec[i * nz:(i + 1) * nz] for i, n in enumerate(LEV)}
arrays["in_w"] = rec[10 * nz:11 * nz + 1]
for i, n in enumerate(("hfx", "ust", "pblh", "zol")):
    arrays[f"in_{n}"] = rec[11 * nz + 1 + i:11 * nz + 2 + i]
o = 0
for n in F3 + F2:
    size = nz if n in F3 else 1
    arrays[f"wrf_{n}"] = wrf[o:o + size]
    o += size
np.savez(out, **{k: np.ascontiguousarray(v, "<f4") for k, v in arrays.items()})
print(out, {k: v.shape for k, v in arrays.items()})
