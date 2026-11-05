Compiled kernel packs for gpuwm. This required companion has the same version as the engine. The loader verifies each image against its source and compiler identity. Apache-2.0; third-party notices are included.

Since 2.9.12 it also carries gpuwm's native-WRF verification oracles (`verify_oracles/`, with a SHA-256 `MANIFEST.json`), the reference captures `gpuwm verify` and the parity checks grade the GPU ports against. They are copied in at build time from the engine's own `gpuwm/data` tree; forecasts do not read them.
