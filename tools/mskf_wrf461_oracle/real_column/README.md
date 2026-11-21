# MSKF on real forecast columns, against WRF v4.7.1

Replays columns captured from a running forecast through both the GPU port
(`gpuwm.core.mskf.launch_mskf`) and WRF's own byte-unmodified `MSKF_CPS`, and
compares every output word.

1. Capture: run a forecast with `probe_sitecustomize.py` on `PYTHONPATH` as
   `sitecustomize.py` and `MSKF_PROBE_LOG=/path/probe.log`. It logs the minima
   of the MSKF outputs, the radiation merges and the RRTMGP input per call, and
   saves the first MSKF call with a negative `qi_cu` (inputs and outputs) to
   `probe.log-mskf.npz`. `MSKF_PROBE_NOLEDGER=1` also makes every health site
   stop on its own step.
2. `python col_case.py probe.log-mskf.npz RUN J,I [J,I ...]` writes
   `RUN/col_f32.bin` and runs the port on the same columns (`RUN/gpu_out.npz`).
   WRF updates W0AVG inside `MSKF_CPS`, so the pre-call W0AVG is the dumped mean
   and W is built so that 0.5*(W(k)+W(k+1)) equals it: both sides form the same
   update. Settings are the 12 km case's: dt 60 s, cudt 5 min fixed, Morrison
   phases, aercu_opt 0, trigger 1, NCA -100.
3. `build.sh BUILD; (cd RUN && BUILD/run_col)` writes `RUN/wrf_out.bin`.
4. `python compare.py RUN` counts differing words per field.

Run the port with `GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict` for a word-for-word
comparison; the default fast numerics differ from WRF by a few ULP.

Result on 20 Nov 2026 (GFS 21 May 2024 12Z, 12 km at 40N 93W, the first call
with a negative `qi_cu`, 7 columns: the shallow column with the negative ice and
six active deep and shallow columns): **0 of 5,873 words differ** in strict
mode, and WRF itself writes `qi_KF = -5.4766e-6` at the shallow plume's cloud
base. The negative comes from DTFRZNEW (module_cu_mskf.F:7091-7093): at the first
freezing level it sublimates `qs(liquid formula) - qu(table)` of ice whether or
not there is any. `make_fixture.py` packed that column into
`tests/data/oracles/mskf/negative_ice_column.npz`.
