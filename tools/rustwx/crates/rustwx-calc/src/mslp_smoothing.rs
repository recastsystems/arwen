//! Sea-level pressure over high terrain, smoothed the way operational map
//! graphics smooth it, so a 3 km map's isobars over the mountains are
//! readable and the lows and highs over flat ground are untouched.
//!
//! THE DEFECT THIS FIXES (hex RRFS v2 conus3km maps, 2026-10-29; the same
//! reduction runs on every ArWen frame).  A sea-level reduction carries the
//! surface pressure down through a fictitious column whose temperature comes
//! from the lowest model level (NCEP Shuell on MPAS-bridge frames,
//! [`rw_mpas::mslp`]; wrf-python's DCOMPUTESEAPRS on WRF frames).  At 3 km
//! the lowest level resolves every valley cold pool and every sunlit ridge,
//! and over 1.5-3 km of terrain a 1 K difference in that temperature moves
//! the reduced pressure by about 1.4 hPa.  The result was right where the
//! ground is low (the conus3km f001 low within ~1 hPa of the model's own
//! diagnostic) and a dense tangle of small closed isobars over the Rockies,
//! the Great Basin and the Sierra Madre.
//! Measured on the conus3km 2023-03-31 12Z frame (1996 x 1301, dx 3.02 km):
//! 56 isobar crossings (4 hPa) per 1,000 grid edges above 1,500 m against
//! 11.5 over the 300-1,000 m Plains and 6.3 over the low East.
//!
//! WHAT OPERATIONAL PRODUCTS DO.  The fictitious column below a mountain is
//! a convention, not a measurement, so operational graphics draw a smoothed
//! sea-level pressure there: the RUC/RAP/HRRR "MAPS" reduction (Benjamin, S.
//! G., and P. A. Miller, 1990: An alternative sea level pressure reduction
//! and a statistical comparison of geostrophic wind estimates with observed
//! surface winds. Mon. Wea. Rev., 118, 2099-2116) replaces the surface
//! temperature by the 700 hPa temperature carried down at 6.5 K/km, to keep
//! the local surface temperature out of the reduction; and grid graphics
//! smooth the field with the classic nine-point smoother, applied as many
//! times as the grid spacing needs (GEMPAK's SM9S: 1/4 centre, 1/8 sides,
//! 1/16 corners, i.e. the 1-2-1 binomial filter in each direction).  On the
//! frame above the MAPS reduction alone still left 30.6 crossings per 1,000
//! edges above 1,500 m: the reduction halves the tangle, the smoothing is
//! what removes it, so the reduction is kept and the smoothing added.
//!
//! THE FILTER HERE.
//!
//! * The field is smoothed by `n` passes of the nine-point smoother, missing
//!   values (off-mesh, boundary rings) excluded and renormalised.  `n` passes
//!   of the 1-2-1 filter are a binomial approximation to a Gaussian of
//!   standard deviation `dx sqrt(n / 2)`, so `n = 2 (L / dx)^2` gives the same
//!   physical smoothing length `L` on any grid: 40 km by default, which is
//!   about 350 passes at 3 km and 19 at 13 km.  `n` is capped at
//!   [`MAX_PASSES`] (a 1 km grid then smooths over about 14 km).
//! * The smoothed field replaces the reduced one by the terrain weight
//!   `w = (z_s - 500 m) / (1,500 m - 500 m)`, clamped to [0, 1]:
//!   `p = p + w (p_smooth - p)`.  Ground at or below 500 m keeps its reduced
//!   pressure word for word, so lows and highs over the Plains, the coasts
//!   and the ocean do not move; above 1,500 m the map draws the smoothed
//!   field.  A frame with no ground above 500 m is returned untouched.
//! * Measured on the conus3km frame: crossings above 1,500 m 56 -> 10.0 per
//!   1,000 edges (Plains 11.5 -> 11.0), largest change 9.2 hPa above 1,500 m
//!   and 2.2 hPa at 500-1,000 m, none below 500 m; the Plains low's minimum
//!   is unchanged (993.29 hPa).  At 25 km more small closed rings stayed
//!   over the Uintas and the Wasatch (12.8 crossings).
//!
//! COARSE AND LATITUDE-LONGITUDE GRIDS (ArWen Global, 22 Nov 2026).  The
//! same tangle covered Tibet and the Himalaya, the Andes, the Rockies,
//! Greenland and Antarctica on the global 0.5 degree maps, already at F000.
//! Two causes, both here: 40 km on a 55.6 km grid is `n = 1` pass, which
//! leaves grid-scale noise standing; and the spacing was read at the grid
//! centre, the equator, so the rows near the poles, a few km apart east to
//! west, were smoothed over far less ground than the rows near the equator,
//! and the box stopped at the 180 degree seam.  Now:
//!
//! * the smoothing length is never less than [`MIN_LENGTH_IN_SPACINGS`]
//!   grid lengths ([`effective_length_m`]);
//! * a regular latitude-longitude grid ([`latlon_rows`]) is smoothed north
//!   to south on its latitude step and, row by row, east to west on that
//!   row's own spacing, and a grid that goes round the Earth wraps across
//!   its seam ([`smooth_latlon_over_high_terrain`]);
//! * measured on the T255 2026-09-01 00Z frame by
//!   `rw-wrfbatch/examples/mslp_terrain_check.rs`: crossings above 1,500 m
//!   outside the polar caps 298.6 -> about 98 per 1,000 edges (low ground
//!   88.5), poleward of 60 degrees 112 -> about 63, and 0 points changed at
//!   or below 500 m.
//!
//! Every renderer that draws sea-level pressure through this function takes
//! the same treatment: ArWen and stock WRF frames, ArWen Global tapes, and
//! hex frames converted from the mesh (the hex package draws its maps with
//! this renderer).
//!
//! `RUSTWX_MSLP_TERRAIN_SMOOTH_KM` sets `L` in km; `0` draws the unsmoothed
//! reduction (for a before/after with one binary).
//!
//! Lives here (moved from rw-wrfbatch) so the any-point column store's sea-level pressure plane (rw-soundings
//! `--columns`) is the map's own, word for word; the renderer re-exports it.

use rayon::prelude::*;

/// Terrain at or below this height keeps its reduced pressure unchanged (m).
pub const SMOOTH_FLOOR_M: f32 = 500.0;
/// Terrain at or above this height takes the smoothed pressure in full (m).
pub const SMOOTH_FULL_M: f32 = 1_500.0;
/// The default smoothing length (km): the standard deviation of the
/// Gaussian the repeated nine-point smoother approximates.
pub const DEFAULT_LENGTH_KM: f64 = 40.0;
/// The most nine-point passes one frame is given.
pub const MAX_PASSES: usize = 400;
/// The shortest smoothing length, in grid lengths, any grid is given.
/// Measured on the ArWen Global T255 2026-09-01 00Z frame (0.5 degree
/// tape, 4 hPa isobar crossings per 1,000 edges above 1,500 m outside the
/// polar caps, raw reduction 298.6): 116.5 at 2.5 grid lengths, 97.4 at 3.6
/// (low ground 88.5), 83.1 at 5; 3.5 is where the high ground reaches the
/// density of the low ground.
pub const MIN_LENGTH_IN_SPACINGS: f64 = 3.5;

/// The smoothing length the environment asks for, in metres (0 = off).
pub fn smoothing_length_m() -> f64 {
    std::env::var("RUSTWX_MSLP_TERRAIN_SMOOTH_KM")
        .ok()
        .and_then(|text| text.trim().parse::<f64>().ok())
        .filter(|km| km.is_finite() && *km >= 0.0)
        .unwrap_or(DEFAULT_LENGTH_KM)
        * 1_000.0
}

/// Nine-point passes for a smoothing length `length_m` on spacing `dx_m`.
pub fn passes_for(length_m: f64, dx_m: f64) -> usize {
    if !(length_m > 0.0) || !(dx_m > 0.0) || !dx_m.is_finite() {
        return 0;
    }
    let ratio = length_m / dx_m;
    ((2.0 * ratio * ratio).round() as usize).clamp(1, MAX_PASSES)
}

/// The terrain weight: 0 at and below [`SMOOTH_FLOOR_M`], 1 at and above
/// [`SMOOTH_FULL_M`], linear between; missing terrain is 0.
pub fn terrain_weight(terrain_m: f32) -> f32 {
    if !terrain_m.is_finite() {
        return 0.0;
    }
    ((terrain_m - SMOOTH_FLOOR_M) / (SMOOTH_FULL_M - SMOOTH_FLOOR_M)).clamp(0.0, 1.0)
}

/// The grid spacing (m) at the centre of a lat/lon-located grid: the mean
/// great-circle distance to the next point east and north.
pub fn centre_spacing_m(lat_deg: &[f32], lon_deg: &[f32], nx: usize, ny: usize) -> Option<f64> {
    if nx < 2 || ny < 2 || lat_deg.len() != nx * ny || lon_deg.len() != nx * ny {
        return None;
    }
    let (j, i) = ((ny - 1) / 2, (nx - 1) / 2);
    let at = |jj: usize, ii: usize| (lat_deg[jj * nx + ii] as f64, lon_deg[jj * nx + ii] as f64);
    let gc = |(a1, o1): (f64, f64), (a2, o2): (f64, f64)| {
        let (a1, o1, a2, o2) = (a1.to_radians(), o1.to_radians(), a2.to_radians(), o2.to_radians());
        let h = ((a2 - a1) / 2.0).sin().powi(2) + a1.cos() * a2.cos() * ((o2 - o1) / 2.0).sin().powi(2);
        2.0 * 6_371_000.0 * h.sqrt().min(1.0).asin()
    };
    let dx = 0.5 * (gc(at(j, i), at(j, i + 1)) + gc(at(j, i), at(j + 1, i)));
    (dx.is_finite() && dx > 0.0).then_some(dx)
}

/// Smooth `values` (one plane, row-major `ny` x `nx`) over high `terrain`
/// with `passes` nine-point passes.  Returns the number of grid points
/// changed.
pub fn smooth_over_high_terrain(
    values: &mut [f32],
    terrain: &[f32],
    nx: usize,
    ny: usize,
    passes: usize,
) -> usize {
    let cells = nx.saturating_mul(ny);
    if passes == 0 || nx < 3 || ny < 3 || values.len() != cells || terrain.len() != cells {
        return 0;
    }
    // The bounding box of the weighted ground, widened by the smoother's
    // reach (one point per pass) so every weighted point sees exactly what
    // a whole-grid pass would give it.
    let (mut j0, mut j1, mut i0, mut i1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for j in 0..ny {
        for i in 0..nx {
            if terrain_weight(terrain[j * nx + i]) > 0.0 {
                j0 = j0.min(j);
                j1 = j1.max(j);
                i0 = i0.min(i);
                i1 = i1.max(i);
            }
        }
    }
    if j0 == usize::MAX {
        return 0;
    }
    let reach = passes + 1;
    let (j0, j1) = (j0.saturating_sub(reach), (j1 + reach).min(ny - 1));
    let (i0, i1) = (i0.saturating_sub(reach), (i1 + reach).min(nx - 1));
    let (bx, by) = (i1 - i0 + 1, j1 - j0 + 1);

    // Numerator (finite values) and denominator (finite mask) are smoothed
    // alike; their ratio is the smoother over the points that exist.
    let mut sum: Vec<f64> = vec![0.0; bx * by];
    let mut mass: Vec<f64> = vec![0.0; bx * by];
    sum.par_chunks_mut(bx)
        .zip(mass.par_chunks_mut(bx))
        .enumerate()
        .for_each(|(jj, (srow, mrow))| {
            let row = &values[(j0 + jj) * nx + i0..(j0 + jj) * nx + i0 + bx];
            for (k, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    srow[k] = v as f64;
                    mrow[k] = 1.0;
                }
            }
        });
    let mut scratch_s = vec![0.0f64; bx * by];
    let mut scratch_m = vec![0.0f64; bx * by];
    for _ in 0..passes {
        one_two_one_x(&sum, &mut scratch_s, bx);
        one_two_one_x(&mass, &mut scratch_m, bx);
        one_two_one_y(&scratch_s, &mut sum, bx, by);
        one_two_one_y(&scratch_m, &mut mass, bx, by);
    }

    let changed = std::sync::atomic::AtomicUsize::new(0);
    values
        .par_chunks_mut(nx)
        .enumerate()
        .skip(j0)
        .take(by)
        .for_each(|(j, row)| {
            let jj = j - j0;
            let mut n = 0usize;
            for (k, value) in row[i0..i0 + bx].iter_mut().enumerate() {
                let w = terrain_weight(terrain[j * nx + i0 + k]);
                let m = mass[jj * bx + k];
                if w > 0.0 && value.is_finite() && m > 0.0 {
                    let smooth = sum[jj * bx + k] / m;
                    let blended = (*value as f64 + w as f64 * (smooth - *value as f64)) as f32;
                    if blended != *value {
                        *value = blended;
                        n += 1;
                    }
                }
            }
            changed.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        });
    changed.into_inner()
}

/// One 1-2-1 pass along rows; points beyond the box count as missing.
fn one_two_one_x(src: &[f64], dst: &mut [f64], bx: usize) {
    dst.par_chunks_mut(bx).zip(src.par_chunks(bx)).for_each(|(out, row)| {
        for i in 0..bx {
            let left = if i > 0 { row[i - 1] } else { 0.0 };
            let right = if i + 1 < bx { row[i + 1] } else { 0.0 };
            out[i] = 0.25 * left + 0.5 * row[i] + 0.25 * right;
        }
    });
}

/// One 1-2-1 pass along columns; points beyond the box count as missing.
fn one_two_one_y(src: &[f64], dst: &mut [f64], bx: usize, by: usize) {
    dst.par_chunks_mut(bx).enumerate().for_each(|(j, out)| {
        let mid = &src[j * bx..(j + 1) * bx];
        let up = (j > 0).then(|| &src[(j - 1) * bx..j * bx]);
        let down = (j + 1 < by).then(|| &src[(j + 1) * bx..(j + 2) * bx]);
        for i in 0..bx {
            out[i] = 0.25 * up.map_or(0.0, |r| r[i]) + 0.5 * mid[i] + 0.25 * down.map_or(0.0, |r| r[i]);
        }
    });
}

/// The smoothing length a grid of spacing `dx_m` is given: the asked
/// length, but never less than [`MIN_LENGTH_IN_SPACINGS`] grid lengths
/// (`0` stays off).  At or below 11.4 km spacing this is the default 40 km
/// itself, so 3 km and 9 km maps are drawn exactly as before; a 13 km grid
/// is smoothed over 45.5 km instead of 40.
pub fn effective_length_m(length_m: f64, dx_m: f64) -> f64 {
    if !(length_m > 0.0) || !length_m.is_finite() {
        return 0.0;
    }
    if !(dx_m > 0.0) || !dx_m.is_finite() {
        return length_m;
    }
    length_m.max(MIN_LENGTH_IN_SPACINGS * dx_m)
}

/// The most east-west passes one row of a latitude-longitude grid is
/// given.  Rows near a pole are a few km long per point, so they need many
/// more passes than [`MAX_PASSES`] for the same length in km.
pub const MAX_ROW_PASSES: usize = 4_000;

/// Earth radius (m) the latitude-longitude spacings are measured with.
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// A regular latitude-longitude grid: one latitude per row, one longitude
/// per column, even steps.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonRows {
    /// Each row's latitude (degrees).
    pub row_lat_deg: Vec<f64>,
    /// The longitude step (degrees, positive).
    pub dlon_deg: f64,
    /// The latitude step (degrees, positive).
    pub dlat_deg: f64,
    /// The rows go once round the Earth, so the first and last columns are
    /// neighbours.
    pub periodic: bool,
}

fn wrap180(degrees: f64) -> f64 {
    let wrapped = (degrees + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

/// The rows of a regular latitude-longitude grid, or `None` for any other
/// grid (a projected grid keeps the centre-spacing smoother).
pub fn latlon_rows(lat_deg: &[f32], lon_deg: &[f32], nx: usize, ny: usize) -> Option<LatLonRows> {
    if nx < 3 || ny < 3 || lat_deg.len() != nx * ny || lon_deg.len() != nx * ny {
        return None;
    }
    let step = wrap180(lon_deg[1] as f64 - lon_deg[0] as f64);
    if !(step.abs() > 1e-6) || !step.is_finite() {
        return None;
    }
    let tolerance = 1e-3_f64.max(1e-3 * step.abs());
    for i in 1..nx {
        let d = wrap180(lon_deg[i] as f64 - lon_deg[i - 1] as f64);
        if (d - step).abs() > tolerance {
            return None;
        }
    }
    let mut row_lat_deg = Vec::with_capacity(ny);
    for j in 0..ny {
        let row_lat = lat_deg[j * nx] as f64;
        for i in 0..nx {
            let k = j * nx + i;
            if (lat_deg[k] as f64 - row_lat).abs() > tolerance
                || wrap180(lon_deg[k] as f64 - lon_deg[i] as f64).abs() > tolerance
            {
                return None;
            }
        }
        row_lat_deg.push(row_lat);
    }
    let dlat = (row_lat_deg[1] - row_lat_deg[0]).abs();
    if !(dlat > 1e-6) {
        return None;
    }
    for j in 1..ny {
        if ((row_lat_deg[j] - row_lat_deg[j - 1]).abs() - dlat).abs() > tolerance.max(1e-3 * dlat) {
            return None;
        }
    }
    let dlon = step.abs();
    let periodic = (nx as f64 * dlon - 360.0).abs() < 0.5 * dlon;
    Some(LatLonRows { row_lat_deg, dlon_deg: dlon, dlat_deg: dlat, periodic })
}

/// Smooth a regular latitude-longitude plane over high terrain by the
/// physical length `length_m` ([`effective_length_m`] of the north-south
/// spacing): `n = 2 (L / dy)^2` north-south passes, and on each row
/// `2 (L / dx_row)^2` east-west passes with `dx_row` the row's own spacing,
/// `dy cos(latitude)`, so a row near a pole is smoothed over the same km as
/// one at the equator.  A grid that goes round the Earth wraps east-west, so
/// the column at the seam is smoothed with its neighbours across it.  The
/// blend and the terrain weight are [`smooth_over_high_terrain`]'s: ground
/// at or below [`SMOOTH_FLOOR_M`] keeps its value word for word.  Returns the
/// number of grid points changed.
pub fn smooth_latlon_over_high_terrain(
    values: &mut [f32],
    terrain: &[f32],
    nx: usize,
    ny: usize,
    rows: &LatLonRows,
    length_m: f64,
) -> usize {
    let cells = nx.saturating_mul(ny);
    if !(length_m > 0.0) || nx < 3 || ny < 3 || values.len() != cells || terrain.len() != cells
        || rows.row_lat_deg.len() != ny
    {
        return 0;
    }
    let (mut j0, mut j1) = (usize::MAX, 0usize);
    for j in 0..ny {
        if terrain[j * nx..(j + 1) * nx].iter().any(|&z| terrain_weight(z) > 0.0) {
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
    }
    if j0 == usize::MAX {
        return 0;
    }
    let dy = EARTH_RADIUS_M * rows.dlat_deg.to_radians();
    let dlon_m = EARTH_RADIUS_M * rows.dlon_deg.to_radians();
    let length = effective_length_m(length_m, dy);
    let passes_y = passes_for(length, dy);
    let reach = passes_y + 1;
    let (j0, j1) = (j0.saturating_sub(reach), (j1 + reach).min(ny - 1));
    let by = j1 - j0 + 1;

    let mut sum: Vec<f64> = vec![0.0; nx * by];
    let mut mass: Vec<f64> = vec![0.0; nx * by];
    let periodic = rows.periodic;
    sum.par_chunks_mut(nx)
        .zip(mass.par_chunks_mut(nx))
        .enumerate()
        .for_each(|(jj, (srow, mrow))| {
            let j = j0 + jj;
            for (k, &v) in values[j * nx..(j + 1) * nx].iter().enumerate() {
                if v.is_finite() {
                    srow[k] = v as f64;
                    mrow[k] = 1.0;
                }
            }
            let dx = dlon_m * rows.row_lat_deg[j].to_radians().cos().abs();
            let passes_x = if dx > 0.0 && dx.is_finite() {
                let ratio = length / dx;
                ((2.0 * ratio * ratio).round() as usize).clamp(1, MAX_ROW_PASSES)
            } else {
                MAX_ROW_PASSES
            };
            let mut scratch = vec![0.0f64; nx];
            for _ in 0..passes_x {
                one_two_one_row(srow, &mut scratch, periodic);
                srow.copy_from_slice(&scratch);
                one_two_one_row(mrow, &mut scratch, periodic);
                mrow.copy_from_slice(&scratch);
            }
        });
    let mut scratch_s = vec![0.0f64; nx * by];
    let mut scratch_m = vec![0.0f64; nx * by];
    for _ in 0..passes_y {
        one_two_one_y(&sum, &mut scratch_s, nx, by);
        one_two_one_y(&mass, &mut scratch_m, nx, by);
        std::mem::swap(&mut sum, &mut scratch_s);
        std::mem::swap(&mut mass, &mut scratch_m);
    }

    let changed = std::sync::atomic::AtomicUsize::new(0);
    values
        .par_chunks_mut(nx)
        .enumerate()
        .skip(j0)
        .take(by)
        .for_each(|(j, row)| {
            let jj = j - j0;
            let mut n = 0usize;
            for (i, value) in row.iter_mut().enumerate() {
                let w = terrain_weight(terrain[j * nx + i]);
                let m = mass[jj * nx + i];
                if w > 0.0 && value.is_finite() && m > 0.0 {
                    let smooth = sum[jj * nx + i] / m;
                    let blended = (*value as f64 + w as f64 * (smooth - *value as f64)) as f32;
                    if blended != *value {
                        *value = blended;
                        n += 1;
                    }
                }
            }
            changed.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        });
    changed.into_inner()
}

/// One 1-2-1 pass along one row; the ends wrap when `periodic`, otherwise
/// points beyond them count as missing.
fn one_two_one_row(src: &[f64], dst: &mut [f64], periodic: bool) {
    let n = src.len();
    for i in 0..n {
        let left = if i > 0 { src[i - 1] } else if periodic { src[n - 1] } else { 0.0 };
        let right = if i + 1 < n { src[i + 1] } else if periodic { src[0] } else { 0.0 };
        dst[i] = 0.25 * left + 0.5 * src[i] + 0.25 * right;
    }
}

/// The map's sea-level pressure from a reduced plane: smoothed over high
/// terrain by the environment's (or the default) smoothing length, never
/// less than [`MIN_LENGTH_IN_SPACINGS`] grid lengths.  A regular
/// latitude-longitude grid is smoothed row by row at each row's own spacing
/// and, when it goes round the Earth, across its seam
/// ([`smooth_latlon_over_high_terrain`]); any other grid on its centre
/// spacing.  Returns the number of grid points changed.
pub fn display_mslp(
    values: &mut [f32],
    terrain: &[f32],
    lat_deg: &[f32],
    lon_deg: &[f32],
    nx: usize,
    ny: usize,
) -> usize {
    if let Some(rows) = latlon_rows(lat_deg, lon_deg, nx, ny) {
        return smooth_latlon_over_high_terrain(values, terrain, nx, ny, &rows, smoothing_length_m());
    }
    let Some(dx) = centre_spacing_m(lat_deg, lon_deg, nx, ny) else {
        return 0;
    };
    smooth_over_high_terrain(
        values,
        terrain,
        nx,
        ny,
        passes_for(effective_length_m(smoothing_length_m(), dx), dx),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_ground_keeps_its_pressure_word_for_word() {
        // a noisy field over a plain at 400 m beside a 2,000 m plateau
        let (nx, ny) = (40, 30);
        let mut values = Vec::new();
        let mut terrain = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let noise = if (i + j) % 2 == 0 { 600.0 } else { -600.0 };
                values.push(100_000.0 + 10.0 * i as f32 + noise);
                terrain.push(if i < 20 { 400.0 } else { 2_000.0 });
            }
        }
        let before = values.clone();
        let changed = smooth_over_high_terrain(&mut values, &terrain, nx, ny, 30);
        assert_eq!(changed, 20 * ny);
        for j in 0..ny {
            for i in 0..20 {
                assert_eq!(values[j * nx + i].to_bits(), before[j * nx + i].to_bits());
            }
        }
        // the plateau's checkerboard is gone: neighbours within 1 Pa of each other
        // away from the edge, where the gradient is 10 Pa per point
        for j in 5..ny - 5 {
            for i in 25..nx - 5 {
                let d = (values[j * nx + i] - values[j * nx + i + 1]).abs();
                assert!(d < 20.0, "{j} {i} {d}");
            }
        }
    }

    #[test]
    fn a_frame_without_high_ground_is_untouched() {
        let (nx, ny) = (10, 10);
        let mut values: Vec<f32> = (0..100).map(|k| 100_000.0 + k as f32).collect();
        let before = values.clone();
        let terrain = vec![499.0; 100];
        assert_eq!(smooth_over_high_terrain(&mut values, &terrain, nx, ny, 50), 0);
        assert_eq!(values, before);
    }

    #[test]
    fn missing_points_stay_missing_and_do_not_leak() {
        let (nx, ny) = (12, 12);
        let mut values = vec![101_000.0f32; nx * ny];
        for j in 0..ny {
            values[j * nx] = f32::NAN;
        }
        let terrain = vec![3_000.0; nx * ny];
        smooth_over_high_terrain(&mut values, &terrain, nx, ny, 20);
        for j in 0..ny {
            assert!(values[j * nx].is_nan());
            for i in 1..nx {
                // a uniform field stays uniform: missing points are renormalised away
                assert!((values[j * nx + i] - 101_000.0).abs() < 0.01, "{}", values[j * nx + i]);
            }
        }
    }

    #[test]
    fn the_box_gives_what_the_whole_grid_gives() {
        // high ground in a corner of a larger grid: the windowed computation
        // must equal a whole-grid one at every weighted point
        let (nx, ny) = (60, 50);
        let mut values = Vec::new();
        let mut terrain = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                values.push(100_000.0 + ((i * 7 + j * 13) % 17) as f32 * 50.0);
                terrain.push(if i > 40 && j > 30 { 2_500.0 } else { 0.0 });
            }
        }
        let mut windowed = values.clone();
        smooth_over_high_terrain(&mut windowed, &terrain, nx, ny, 5);
        // whole grid by hand
        let mut s: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let mut m = vec![1.0f64; nx * ny];
        let (mut ts, mut tm) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
        for _ in 0..5 {
            one_two_one_x(&s, &mut ts, nx);
            one_two_one_x(&m, &mut tm, nx);
            one_two_one_y(&ts, &mut s, nx, ny);
            one_two_one_y(&tm, &mut m, nx, ny);
        }
        for k in 0..nx * ny {
            if terrain[k] > 0.0 {
                let expected = (s[k] / m[k]) as f32;
                assert!((windowed[k] - expected).abs() < 1e-3, "{k}");
            } else {
                assert_eq!(windowed[k], values[k]);
            }
        }
    }

    #[test]
    fn passes_follow_the_physical_length() {
        assert_eq!(passes_for(40_000.0, 3_000.0), 356);
        assert_eq!(passes_for(40_000.0, 13_000.0), 19);
        assert_eq!(passes_for(40_000.0, 1_000.0), MAX_PASSES);
        assert_eq!(passes_for(0.0, 3_000.0), 0);
        assert_eq!(terrain_weight(500.0), 0.0);
        assert_eq!(terrain_weight(1_000.0), 0.5);
        assert_eq!(terrain_weight(2_000.0), 1.0);
        assert_eq!(terrain_weight(f32::NAN), 0.0);
    }

    /// A global 0.5 degree plane: rows from -89.75 to 89.75, columns from
    /// -180 by 0.5, as the ArWen Global tape writes it.
    fn global_half_degree() -> (Vec<f32>, Vec<f32>, usize, usize) {
        let (nx, ny) = (720usize, 360usize);
        let mut lat = Vec::with_capacity(nx * ny);
        let mut lon = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                lat.push(-89.75 + 0.5 * j as f32);
                lon.push(-180.0 + 0.5 * i as f32);
            }
        }
        (lat, lon, nx, ny)
    }

    /// Noise of a six-point wavelength in both directions, the scale of the
    /// rings the 0.5 degree maps showed (one 1-2-1 pass leaves 56 % of it).
    fn ring_noise(i: usize, j: usize) -> f32 {
        let phase = std::f32::consts::TAU / 6.0;
        300.0 * ((i as f32 * phase).sin() + (j as f32 * phase).sin())
    }

    fn amplitude(values: &[f32], nx: usize, rows: std::ops::Range<usize>, cols: &[usize]) -> f32 {
        let mut most = 0f32;
        for j in rows {
            for &i in cols {
                most = most.max((values[j * nx + i] - 101_000.0).abs());
            }
        }
        most
    }

    #[test]
    fn coarse_grids_are_smoothed_over_several_grid_lengths() {
        // 40 km on the 0.5 degree tape was one pass; fine grids keep 40 km.
        assert_eq!(effective_length_m(40_000.0, 3_000.0), 40_000.0);
        assert_eq!(effective_length_m(40_000.0, 9_000.0), 40_000.0);
        assert_eq!(effective_length_m(40_000.0, 55_600.0), 3.5 * 55_600.0);
        assert!(passes_for(effective_length_m(40_000.0, 55_600.0), 55_600.0) >= 24);
        assert_eq!(effective_length_m(0.0, 55_600.0), 0.0);
    }

    #[test]
    fn a_global_latlon_grid_is_recognised_and_a_projected_one_is_not() {
        let (lat, lon, nx, ny) = global_half_degree();
        let rows = latlon_rows(&lat, &lon, nx, ny).expect("a latitude-longitude grid");
        assert!(rows.periodic);
        assert!((rows.dlon_deg - 0.5).abs() < 1e-9 && (rows.dlat_deg - 0.5).abs() < 1e-9);
        // A window over the Americas, and one across the date line, are not periodic.
        let window = |lon0: f32| {
            let (nx, ny) = (40usize, 20usize);
            let lat: Vec<f32> = (0..nx * ny).map(|k| 20.0 + 0.5 * (k / nx) as f32).collect();
            let lon: Vec<f32> = (0..nx * ny)
                .map(|k| {
                    let l = lon0 + 0.5 * (k % nx) as f32;
                    if l >= 180.0 { l - 360.0 } else { l }
                })
                .collect();
            latlon_rows(&lat, &lon, nx, ny)
        };
        assert_eq!(window(-130.0).map(|r| r.periodic), Some(false));
        assert_eq!(window(170.0).map(|r| r.periodic), Some(false));
        // A projected grid: latitude changes along a row.
        let (nx, ny) = (10usize, 10usize);
        let lat: Vec<f32> = (0..nx * ny).map(|k| 30.0 + 0.1 * (k / nx) as f32 + 0.01 * (k % nx) as f32).collect();
        let lon: Vec<f32> = (0..nx * ny).map(|k| -100.0 + 0.1 * (k % nx) as f32).collect();
        assert_eq!(latlon_rows(&lat, &lon, nx, ny), None);
    }

    /// THE DEFECT: on the global 0.5 degree maps the noise of the reduction
    /// stood over high ground at every latitude.  High ground at the
    /// equator, at 70 S, and astride the 180 degree seam must each lose
    /// almost all of it; low ground keeps its pressure word for word.
    #[test]
    fn a_global_plane_is_smoothed_at_every_latitude_and_across_the_seam() {
        let (lat, lon, nx, ny) = global_half_degree();
        let mut values = Vec::with_capacity(nx * ny);
        let mut terrain = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                values.push(101_000.0 + ring_noise(i, j));
                let (la, lo) = (lat[j * nx + i], lon[j * nx + i]);
                let plateau = (la.abs() < 10.0 && (60.0..100.0).contains(&lo))
                    || la < -65.0
                    || (la.abs() < 10.0 && lo.abs() > 165.0);
                terrain.push(if plateau { 3_000.0 } else { 0.0 });
            }
        }
        let before = values.clone();
        let changed = display_mslp(&mut values, &terrain, &lat, &lon, nx, ny);
        assert!(changed > 0);
        for k in 0..nx * ny {
            if terrain[k] == 0.0 {
                assert_eq!(values[k].to_bits(), before[k].to_bits(), "low ground moved at {k}");
            }
        }
        let raw = 600.0;
        // the equatorial plateau (70-90 E, 4 S-4 N, away from its edges)
        let cols: Vec<usize> = (500..540).collect();
        assert!(amplitude(&values, nx, 172..188, &cols) < 0.05 * raw);
        // 70-80 S, all the way round
        let cols: Vec<usize> = (0..nx).collect();
        assert!(amplitude(&values, nx, 20..40, &cols) < 0.05 * raw);
        // astride the seam: the last and first columns, 4 S-4 N
        let cols: Vec<usize> = (0..12).chain(nx - 12..nx).collect();
        assert!(amplitude(&values, nx, 172..188, &cols) < 0.05 * raw);
    }

    /// Smoothing a global plane does not care where its seam is: rolling the
    /// columns and smoothing gives the smoothed plane rolled.
    #[test]
    fn the_seam_is_not_an_edge() {
        let (lat, lon, nx, ny) = global_half_degree();
        let mut values = Vec::with_capacity(nx * ny);
        let mut terrain = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                values.push(101_000.0 + ring_noise(i, j) + 40.0 * (i as f32 * 0.37).cos());
                terrain.push(if (j / 7 + i / 11) % 3 == 0 { 2_000.0 } else { 0.0 });
            }
        }
        let roll = |plane: &[f32], by: usize| -> Vec<f32> {
            let mut out = vec![0f32; plane.len()];
            for j in 0..ny {
                for i in 0..nx {
                    out[j * nx + (i + by) % nx] = plane[j * nx + i];
                }
            }
            out
        };
        let mut a = values.clone();
        display_mslp(&mut a, &terrain, &lat, &lon, nx, ny);
        let mut b = roll(&values, 300);
        display_mslp(&mut b, &roll(&terrain, 300), &lat, &lon, nx, ny);
        let a = roll(&a, 300);
        for k in 0..nx * ny {
            assert!((a[k] - b[k]).abs() < 0.01, "{k}: {} {}", a[k], b[k]);
        }
    }
}
