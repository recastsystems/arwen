# Provenance: clean-snicar-layers (Noah-MP opt_alb=3)

Lane `lane/noahmp-clean-clean-snicar-layers`, base `ccbb7f5a284` (port/all, which
carries lic-snicar and spd/snicar). Every routine here is **re-implemented from
published descriptions, not bitwise to upstream**. No file from NCAR/noahmp, from
the WRF 4.8.0 noahmp submodule, from CLM/CTSM or from any other SNICAR source was
opened, cloned, read or quoted for this change. The sources are papers only.

## What the base already had

lic-snicar plus spd/snicar already carry a three-layer adding-doubling solve
(Flanner et al. 2021 Appendix A) over soil or four glacier ice layers. They also
feed layer-resolved absorbed solar into the snow/soil heat solve: kernel
`noahmp_tsnosoi_snow_papers` puts the sub-surface absorption into the source term
of the tridiagonal temperature solve. This lane adds the two missing choices: the
Toon 1989 two-stream solver and the five-band mode. Both use the same
layer-absorption hand-off, so layer heating works under every new choice.

## Options

| Option | Values | Default | Label |
| --- | --- | --- | --- |
| `snicar_solver_opt` | 1 adding-doubling, 2 Toon 1989 two-stream | 1 | 2 is re-implemented from published descriptions, not bitwise to upstream |
| `snicar_bandnumber_opt` | 480 bins, 5 bands | 480 | 5 is re-implemented from published descriptions, not bitwise to upstream |

Both are read from `&noah_mp` or from TOML, and only when `opt_alb=3`. At their
defaults they are absent from every identity (config echo, checkpoint identity,
restart identity, forecast receipt). Default runs therefore keep their bytes. The
default 480-bin AD path launches the unchanged `snow_papers_albedo` image, with the
same 138 registers and 1136 B frame. Any new choice launches the separate image
`snow_papers_albedo_layers`.

Scope: glacier-ice columns always use the 480-bin adding-doubling solver. Only that
solver carries the Fresnel ice interface of Whicker et al. 2022. This applies to
both new options, and the identity records it.

## Routine by routine

`gpuwm/core/kernels/snow_papers.cu`:

- **`toon_land`**: Toon et al. 1989, JGR 94(D13):16287, multi-layer two-stream.
  - **Delta scaling.** Joseph et al. 1976 with f = g²: τ* = (1-ωf)τ, ω* = (1-f)ω/(1-ωf), g* = g/(1+g). Flanner et al. 2021 (GMD 14:7673) sect. 2.1 says every SNICAR version applies it.
  - **γ1..γ4.** From Toon 1989 Table 1, with Eddington in the visible and hemispheric mean in the near-IR. Dang et al. 2019 (TC 13:2325) sect. 2.1 describes this pairing for SNICAR. Table 1 gives no beam coefficient for hemispheric mean, so the near-IR beam source uses the quadrature γ3 = (1-√3 g μ0)/2. This is a declared choice of ours.
  - **Equations.** The two-stream equations are Toon 1989 eqs 1-2. The layer solution has homogeneous modes e^{±λτ}, with λ = √(γ1²-γ2²) and Γ = γ2/(γ1+λ). The particular solutions C± follow Toon 1989 eqs 23-24, with πF_s = 1/μ0, so the incident beam flux is 1. The modes are written in a scaled form (P decays from the layer top, Q from the layer bottom), so that no exponential exceeds 1.
  - **Boundaries and interfaces** (Toon 1989 sect. 3). At the top, F↓ = 0 for beam illumination and 1 for diffuse. Both up and down diffuse fluxes are continuous at every interface. At the bottom, F↑ = R_s(F↓ + direct). Toon rearranges this 2N-equation system into a tridiagonal one. With at most three snow layers we eliminate the same 2N x 2N system directly with partial pivoting. The equations are identical; only the rounding differs.
  - **Singularity at λ = 1/μ0.** Dang et al. 2019 sect. 7 shifts μ0 by ±0.02. We shift by a relative 1e-5, which is enough to clear the removable singularity in double precision.
  - **Absorption.** Each layer's absorption is the difference in net flux (diffuse down plus direct minus up) across it. The ground takes the net flux at the bottom, so absorption plus albedo equals 1.
- **`ad_land`**: the existing adding-doubling arithmetic (`slab`, `add`, `netflux`; Flanner et al. 2021 Appendix A eqs A1-A20; Dang et al. 2019) as a routine for the dispatch paths. A test shows it reproduces the inline path exactly (`snow_ad_probe`).
- **`bin_layer_optics`**: the same per-bin layer optics as the inline 480-bin path. It reads the existing generated Mie table (our Rust Lorenz-Mie code with CC0 refractiveindex.info inputs), the He et al. 2017 fits for internal BC and dust, and external impurities by mass. It contains no new physics.
- **Five-band mode** (in `snow_albedo_body<true>`):
  - **Band edges.** 0.3-0.7, 0.7-1.0, 1.0-1.2, 1.2-1.5 and 1.5-5 µm, from Dang et al. 2019 sect. 7. Our table starts at 0.2 µm, but the 0.2-0.3 µm bins carry zero G173 weight.
  - **Band optics.** A Chandrasekhar mean, after Dang et al. 2019 eqs 15a-c, Thomas & Stamnes 1999 eq 9.27 and Flanner et al. 2007. Extinction and asymmetry are means weighted by incident flux. Single-scattering albedo is weighted by flux times α_λ, the hemispheric albedo of optically thick snow. We compute α_λ in **`thick_albedo`** as the semi-infinite delta-Eddington limit (u-1)/(u+1) of the layer's own optics; this is our choice.
  - **Direct and diffuse sets.** Two sets of band properties are formed, one with the G173 direct spectrum and one with the G173 diffuse spectrum. Dang 2019 sect. 7 describes one set per illumination.
  - **Broadband totals.** These use the same G173 band weights as the 480-bin path.
  - **Determinism.** Each (illumination, band, layer) task runs on one thread in a fixed bin order, so the result is deterministic.
  - **Reading of eqs 15b-c.** We read eqs 15b-c as flux-weighted means; the extracted text of the paper is ambiguous about their denominators.
- **`column_solve`**, `snow_column_probe`: the dispatch and a test probe.

Python and config:
- `gpuwm/core/snow_papers.py` validates the options and dispatches the two images.
- `identity(options)` lists non-default choices only.
- `gpuwm/config.py` holds the two fields, `NOAHMP_OPTION_IDENTITY_EVIDENCE` and `NOAHMP_OPTION_VALUES`, so `&noah_mp` import works.
- `gpuwm/checkpoint_identity.py` keeps default values out of identities.
- `noahmp_runtime.py` puts the non-default choices into the restart identity.
- `noahmp_selected_frames.py` and `preflight.py` price frames bound to the source: 1136 B for the default image and 2000 B for the layers image. Both were read on sm_120 with NVRTC 13.4, fast and strict.

## Data

No new data. The optics are the existing generated table (SHA256 `0e0a3779...`).

## Sources

- Toon, O. B., McKay, C. P., Ackerman, T. P., Santhanam, K. (1989), JGR 94(D13), 16287-16301.
- Joseph, J. H., Wiscombe, W. J., Weinman, J. A. (1976), J. Atmos. Sci. 33, 2452-2459.
- Dang, C., Zender, C. S., Flanner, M. G. (2019), The Cryosphere 13, 2325-2343 (CC BY 4.0).
- Flanner, M. G. et al. (2021), GMD 14, 7673-7704 (CC BY 4.0), sect. 2.1 and Appendix A.
- Flanner, M. G., Zender, C. S., Randerson, J. T., Rasch, P. J. (2007), JGR 112, D11202.
- Thomas, G. E., Stamnes, K. (1999), Radiative Transfer in the Atmosphere and Ocean, eq. 9.27.
- Lin, T.-S. et al. (2025), JHM 26(2), doi:10.1175/JHM-D-24-0082.1, and Lin's public RAL talk, June 2024: the solver and band choices offered in Noah-MP v5 (option set only).
- Whicker, C. A. et al. (2022), TC 16, 1197: why glacier columns keep AD.
