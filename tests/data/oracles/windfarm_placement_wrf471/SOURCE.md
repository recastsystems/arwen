# Where these fixtures come from

They are the placement part of the WRF v4.7.1 wind-farm layout oracle of the
public jacobbieker/woof fork (pull request 123, by GitHub user jacobbieker,
Apache-2.0), at the fork's merge commit bb2eb8ee0. WRF's own
`share/module_llxy.F` and `init_module_wind_fitch`'s placement loop, compiled
byte-unmodified in real(4) (see PROVENANCE.md, the fork's record, kept as it
is), placed each turbine of eleven synthetic farms. Only the placement words
are kept here, per case:

- `grid.txt`: line 1 `map_proj truelat1 truelat2 stand_lon dx`, line 3
  XLAT/XLONG(1,1), the corner WRF anchors `map_set` at;
- `lat.bin`, `lon.bin`: the float32 turbine positions WRF read;
- `ts_rx.bin`, `ts_ry.bin`: WRF's float32 `latlon_to_ij` result, 1-based;
- `ival.bin`, `jval.bin`: `nint` of those, the cell WRF uses.

All raw little-endian, sizes in each MANIFEST.txt. PROVENANCE.md describes
the fork's whole fixture set, of which the table and drag-coefficient parts
are not carried here.
