"""Compare the port's outputs (gpu_out.npz) with WRF's (wrf_out.bin), word for word."""
import sys
import numpy as np

d = sys.argv[1]
ncol, nz = (int(x) for x in open(f"{d}/ncol.txt").read().split())
F3 = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten", "rucuten",
      "rvcuten", "cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu", "udr_kf", "ddr_kf", "uer_kf",
      "der_kf", "w0avg")
F2 = ("raincv", "pratec", "nca", "cutop", "cubot", "timec_kf")
per = len(F3) * nz + len(F2)
wrf = np.fromfile(f"{d}/wrf_out.bin", "<f4").reshape(ncol, per)
gpu = np.load(f"{d}/gpu_out.npz")
total = differ = 0
for c in range(ncol):
    line = [f"col {c}:"]
    o = 0
    for n in F3 + F2:
        size = nz if n in F3 else 1
        w = wrf[c, o:o + size]
        g = gpu[f"c{c}_{n}"][:size]
        o += size
        nd = int((w.view(np.int32) != g.view(np.int32)).sum())
        total += size
        differ += nd
        if nd:
            line.append(f"{n}:{nd} differ (max |d| {np.abs(w.astype(np.float64) - g).max():.3e})")
    w_qi = wrf[c, F3.index("qi_cu") * nz:(F3.index("qi_cu") + 1) * nz]
    g_qi = gpu[f"c{c}_qi_cu"]
    line.append(f"qi_cu min WRF {w_qi[w_qi > -7000].min() if (w_qi > -7000).any() else 'untouched'}"
                f" port {g_qi[g_qi > -7000].min() if (g_qi > -7000).any() else 'untouched'}")
    print(" ".join(str(x) for x in line))
print(f"{differ} of {total} words differ")
