// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use std::env;
use std::fs;
use std::path::Path;

/// True if `feature` is listed in comma-separated `CARGO_CFG_TARGET_FEATURE`
fn has_feature(list: &str, feature: &str) -> bool {
    list.split(',').any(|f| f == feature)
}

fn main() {
    // Target triple features supplied by `cargo` (`--print cfg`)
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let feats = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    // w8 == 8-bits, w16 == 16-bits, w32 == 32-bits, w64 == 64-bits.
    //
    // ==> for u8, u16, u32/f32, u64/f64 lane counts
    //
    // These become constants after the build process
    // i.e., `W8` has the right number of lanes for U8, etc.

    // Allow override via environment variable
    // Format: SIMD_LANES_OVERRIDE="64,32,16,8"
    let override_lanes = env::var("SIMD_LANES_OVERRIDE").ok();

    let (w8, w16, w32, w64) = if let Some(val) = override_lanes {
        let parts: Vec<_> = val.split(',').map(|s| s.trim().parse::<usize>()).collect();
        if parts.len() == 4 && parts.iter().all(|r| r.is_ok()) {
            let vals: Vec<usize> = parts.into_iter().map(|r| r.unwrap()).collect();
            println!("cargo:warning=SIMD_LANES_OVERRIDE applied: {:?}", vals);
            (vals[0], vals[1], vals[2], vals[3])
        } else {
            panic!("Invalid SIMD_LANES_OVERRIDE. Expected 4 comma-separated integers, e.g., \"64,32,16,8\"");
        }
    } else {
        match arch.as_str() {
            // x86 / x86_64
            "x86_64" | "x86" => {
                if has_feature(&feats, "avx512f") {
                    (64, 32, 16, 8)
                }
                // 512-bit
                else if has_feature(&feats, "avx2") {
                    (32, 16, 8, 4)
                }
                // 256-bit
                else if has_feature(&feats, "sse2") {
                    (16, 8, 4, 2)
                }
                // 128-bit
                else {
                    (8, 4, 2, 1)
                } // scalar/soft
            }

            // 64-bit ARM
            // All aarch64 CPUs have NEON (128-bit) by spec; if it was
            // explicitly disabled via `-C target-feature=-neon`, fall back.
            "aarch64" => {
                if has_feature(&feats, "neon") {
                    (16, 8, 4, 2)
                } else {
                    (8, 4, 2, 1)
                }
            }

            // wasm32 with or without SIMD
            "wasm32" => {
                if has_feature(&feats, "simd128") {
                    (16, 8, 4, 2)
                } else {
                    (8, 4, 2, 1)
                }
            }

            // anything else
            _ => (8, 4, 2, 1),
        }
    };

    // Writes to a consts file, and reads at run-time for a dep-free lazy static
    let out_path = Path::new(&env::var("OUT_DIR").unwrap()).join("simd_lanes.rs");

    // The below widths are for each type that has that width, so the larger ones
    // e.g., W64 are actually smaller. I.e., W64 is for i64, etc., but less i64's
    // fit than W8 (U8), etc.
    fs::write(
        &out_path,
        format!(
            "
/// Auto-generated SIMD lane widths from build.rs

/// SIMD lane count for 8-bit elements (u8, i8).
/// Determined at build time based on target architecture capabilities,
/// or overridden via `SIMD_LANES_OVERRIDE`.
#[allow(non_upper_case_globals)]
pub const W8: usize = {w8};

/// SIMD lane count for 16-bit elements (u16, i16).
/// Determined at build time based on target architecture capabilities,
/// or overridden via `SIMD_LANES_OVERRIDE`.
#[allow(non_upper_case_globals)]
pub const W16: usize = {w16};

/// SIMD lane count for 32-bit elements (u32, i32, f32).
/// Determined at build time based on target architecture capabilities,
/// or overridden via `SIMD_LANES_OVERRIDE`.
#[allow(non_upper_case_globals)]
pub const W32: usize = {w32};

/// SIMD lane count for 64-bit elements (u64, i64, f64).
/// Determined at build time based on target architecture capabilities,
/// or overridden via `SIMD_LANES_OVERRIDE`.
#[allow(non_upper_case_globals)]
pub const W64: usize = {w64};
"
        ),
    )
    .unwrap();

    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo:rerun-if-env-changed=SIMD_LANES_OVERRIDE");
}
