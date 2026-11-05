# Port licence rulings — 2026-11-01

The audit covers all 56 current port reports and integration notes in
`~/lanes/downloads/WRF-PORTS-2026-11-01/`. The complete frozen report census,
item-by-item decisions and command evidence are delivered in `HOLDS.md`,
`lic-audit.md` and `evidence/lic-audit/` in that directory.

AER RRTMG's BSD-3-Clause grant clears the four stale tuning-control holds:
`radius_liquid`, `radius_ice`, `radius_snow` and `overlap_break_fraction` in the
`physics.rrtmg` family. The requested `port/port-tuning-knobs` branch carries
the registry correction, actual GPU checks and `docs/rrtmg-controls-proof.md`.
Existing full AER/WRF notices remain. These are coefficient experiments, not
new stock WRF namelist switches. Licence clearance does not upgrade underlying
scheme maturity or certify whole-forecast bitwise native equivalence.

The historical CCPP UGWP v1 “no licence” conclusion is wrong. Its archived
provenance pins CCPP revision `242a2fc0fce0fe71772e2b1a7dc76408fc96c6a7`.
That exact upstream revision grants Apache-2.0, and all eight archived oracle
source hashes match the official files. `physics/GWD/ecmwf_ngw.F90:47` also
explicitly grants Apache-2.0. Licence notice, exact-pin hash/notice review and
manifest are retained under `docs/licence-audit-20261101/`. The old archive
already retained the upstream licence. This clears a licence objection; it
does not import or requalify the historical solver on the current candidate.

The general Noah-MP lineage uncertainty is superseded by eligible WRF-v4 source.
It must not be used to hold ordinary snow, runoff, crop, irrigation or groundwater
work. The current Noah-MP report distinguishes unfinished eligible code from the
separately assigned SNICAR implementation and optical-table restriction.

External CESM current/future aerosol files still lack an established grant for
the exact files inventoried by the aerosol lane. Positive MP40/CU11 `aercu_opt`
and `aercu_fct` qualification share that dependency; table-free zero modes are
licence-clear. The old blanket MP40 hold is superseded. These data and SNICAR
belong to separate lanes and were not read for porting or copied by this audit.

The CAM5 lane's proposed native glibc cosine copy has LGPL-2.1-or-later terms.
That is not a sale prohibition or an absent licence. The lane's permissive
CORE-MATH replacement already ships; its remaining native replay and speed
failures are numerical work. The GFDL logarithm likewise already uses a
permissive helper. CTSM option6 has a missing external library, not an established
licence prohibition. Neither finding justifies calling the WRF physics itself
unshippable.

No additional genuinely restricted component outside SNICAR/CESM was found.
No papers rewrite was necessary. If such a component is found later, the ruling
requires fresh work from papers/public data without reading restricted source
or tables for porting, using the same WRF selector and documentation/receipts
labelled “written from papers, not bitwise to WRF”; option-off output must remain
byte-identical. Proof must include a stock-namelist whole `gpuwm go` forecast,
Rust maps against native WRF and observations, measured/explained differences,
GPU timing within 1.2x of the nearest scheme, and nests/restart/ensemble wiring.

Audit command: `python3 <lanes>/lic-audit/audit_reports.py`.
Licence verification: `python3 <lanes>/lic-audit/verify_licences.py`
checks retained hashes; `--refresh` additionally verifies all eight official
UGWP source hashes without retaining algorithm source. Exact URLs are in the
committed manifest and notice-review files. No runtime production code changes
in this audit branch.
