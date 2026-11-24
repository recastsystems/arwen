//! `mslp_terrain_check` -- how tangled a frame's map sea-level pressure is
//! over high ground, and what the map's terrain smoothing changed.
//!
//! The renderer's sea-level pressure for a WRF-shaped frame is wrf-core's
//! `slp` (DCOMPUTESEAPRS), then [`rustwx_calc::mslp_smoothing::display_mslp`]
//! over high ground.  This example takes both planes from one frame, exactly
//! as `rw_wrfbatch` takes them, and prints:
//!
//! * `CROSSINGS band edges raw drawn`: 4 hPa isobar crossings per 1,000
//!   neighbour edges, raw reduction and drawn plane, by the higher ground of
//!   the edge's two points (`low` < 300 m, `plain` 300-500 m, `slope`
//!   500-1,500 m, `high` >= 1,500 m, `high-polar` >= 1,500 m poleward of 60
//!   degrees).  A tangle of small closed isobars is many crossings.
//! * `CHANGE band points changed max_abs_hpa`: where the drawn plane differs
//!   from the reduction.  Ground at or below 500 m must read 0 changed: the
//!   lows and highs over low ground and the sea do not move.
//! * `EXTREMA`: the deepest low and the highest high over ground at or below
//!   500 m, in both planes.
//!
//! ```text
//! cargo run --release --example mslp_terrain_check -- WRFOUT [WRFOUT ...]
//! ```
//!
//! `RUSTWX_MSLP_TERRAIN_SMOOTH_KM` is honoured as the renderer honours it
//! (`0` draws the raw reduction).

use std::process::ExitCode;

use rustwx_calc::mslp_smoothing::display_mslp;
use wrf_core::{ComputeOpts, WrfFile, getvar};

const BANDS: [&str; 5] = ["low", "plain", "slope", "high", "high-polar"];

fn band(terrain_m: f32, lat_deg: f32) -> Option<usize> {
    if !terrain_m.is_finite() {
        return None;
    }
    Some(match terrain_m {
        z if z < 300.0 => 0,
        z if z < 500.0 => 1,
        z if z < 1_500.0 => 2,
        _ if lat_deg.abs() >= 60.0 => 4,
        _ => 3,
    })
}

fn crossings(values: &[f32], terrain: &[f32], lat: &[f32], nx: usize, ny: usize) -> ([usize; 5], [usize; 5]) {
    let (mut edges, mut crossed) = ([0usize; 5], [0usize; 5]);
    let isobar = |p: f32| (p / 400.0).floor() as i64;
    let mut visit = |a: usize, b: usize| {
        let (pa, pb) = (values[a], values[b]);
        if !pa.is_finite() || !pb.is_finite() {
            return;
        }
        let (high, at) = if terrain[a] >= terrain[b] { (terrain[a], a) } else { (terrain[b], b) };
        let Some(k) = band(high, lat[at]) else { return };
        edges[k] += 1;
        if isobar(pa) != isobar(pb) {
            crossed[k] += 1;
        }
    };
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            if i + 1 < nx {
                visit(k, k + 1);
            }
            if j + 1 < ny {
                visit(k, k + nx);
            }
        }
    }
    (edges, crossed)
}

fn per_thousand(crossed: usize, edges: usize) -> f64 {
    if edges == 0 { 0.0 } else { 1_000.0 * crossed as f64 / edges as f64 }
}

fn extrema(values: &[f32], terrain: &[f32]) -> (f32, f32) {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for (&p, &z) in values.iter().zip(terrain) {
        if p.is_finite() && z.is_finite() && z <= 500.0 {
            low = low.min(p);
            high = high.max(p);
        }
    }
    (low / 100.0, high / 100.0)
}

fn check(path: &str) -> Result<(), String> {
    let file = WrfFile::open(path).map_err(|e| format!("{path}: {e}"))?;
    let opts = ComputeOpts { units: Some("Pa".into()), ..ComputeOpts::default() };
    let slp = getvar(&file, "slp", Some(0), &opts).map_err(|e| format!("{path}: slp: {e}"))?;
    let (ny, nx) = match slp.shape.as_slice() {
        [ny, nx] | [1, ny, nx] => (*ny, *nx),
        other => return Err(format!("{path}: slp shape {other:?}")),
    };
    let raw: Vec<f32> = slp.data.iter().map(|&v| if v.is_finite() { v as f32 } else { f32::NAN }).collect();
    let terrain: Vec<f32> = file.terrain(0).map_err(|e| e.to_string())?.iter().map(|&z| z as f32).collect();
    let lat: Vec<f32> = file.xlat(0).map_err(|e| e.to_string())?.iter().map(|&v| v as f32).collect();
    let lon: Vec<f32> = file.xlong(0).map_err(|e| e.to_string())?.iter().map(|&v| v as f32).collect();
    let mut drawn = raw.clone();
    let changed = display_mslp(&mut drawn, &terrain, &lat, &lon, nx, ny);
    println!("FRAME\t{path}\t{nx}x{ny}\tchanged {changed}");
    let (edges, raw_x) = crossings(&raw, &terrain, &lat, nx, ny);
    let (_, drawn_x) = crossings(&drawn, &terrain, &lat, nx, ny);
    for k in 0..BANDS.len() {
        println!(
            "CROSSINGS\t{}\t{}\t{:.1}\t{:.1}",
            BANDS[k],
            edges[k],
            per_thousand(raw_x[k], edges[k]),
            per_thousand(drawn_x[k], edges[k])
        );
    }
    let mut points = [0usize; 5];
    let mut moved = [0usize; 5];
    let mut largest = [0f32; 5];
    for idx in 0..raw.len() {
        let Some(k) = band(terrain[idx], lat[idx]) else { continue };
        points[k] += 1;
        if raw[idx].is_finite() && drawn[idx].to_bits() != raw[idx].to_bits() {
            moved[k] += 1;
            largest[k] = largest[k].max((drawn[idx] - raw[idx]).abs() / 100.0);
        }
    }
    for k in 0..BANDS.len() {
        println!("CHANGE\t{}\t{}\t{}\t{:.2}", BANDS[k], points[k], moved[k], largest[k]);
    }
    let (raw_low, raw_high) = extrema(&raw, &terrain);
    let (drawn_low, drawn_high) = extrema(&drawn, &terrain);
    println!("EXTREMA\tlow-ground min/max hPa\traw {raw_low:.2}/{raw_high:.2}\tdrawn {drawn_low:.2}/{drawn_high:.2}");
    Ok(())
}

fn main() -> ExitCode {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: mslp_terrain_check WRFOUT [WRFOUT ...]");
        return ExitCode::from(2);
    }
    for path in &paths {
        if let Err(err) = check(path) {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
