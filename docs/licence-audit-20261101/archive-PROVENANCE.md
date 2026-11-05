Synthetic column oracle, not a forecast.

CCPP commit 242a2fc0fce0fe71772e2b1a7dc76408fc96c6a7.
64 columns, 72 levels, model top 100 Pa.
Compiler and commands: tools/ugwp_v1_oracle/build.sh and run_columns.F90.
Columns cover latitude -90..90, zero and nonzero terrain offsets, jet winds and critical-level filtering. Terrain offsets do not qualify the orographic drag.
The upstream solver rejects a two-level column through an out-of-bounds level-3 read.
