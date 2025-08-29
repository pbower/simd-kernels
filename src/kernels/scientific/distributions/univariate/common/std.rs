// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, Vec64};

/// Dense kernel helper
///
/// ### Null handling
/// - Null mask appearing in the dense path means a mask was supplied
/// to the kernel function, with a null_count of `0`. This can reflect
/// a scenario where one knew there was no nulls for a whole vector, or
/// the supplied window, and therefore supplied `0` to ensure that the
/// dense path was used for the kernel.
/// - Any `NaN` or `inf` values generated in the kernel function
/// are kept verbatim, without `nulling` them in the (optional)
/// mask, given that:
/// 1. These values can represent additional data signal.
/// 2. Handling them requires additional CPU cycles on the hot path.
///
/// Therefore, one can treat them further if needed.
#[inline(always)]
pub fn dense_univariate_kernel_f64_std<FScalar>(
    x: &[f64],
    has_mask: bool,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);

    for &xi in x {
        out.push(scalar_body(xi));
    }

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// Null-aware masked kernel helper.
///
/// ### Null handling
/// - Input mask is required and propagates nulls accordingly.
/// - Any `NaN` or `inf` values generated in the kernel function
/// are kept verbatim, without `nulling` them in the (optional)
/// mask, given that:
/// 1. These values can represent additional data signal.
/// 2. Handling them requires additional CPU cycles on the hot path.
/// Therefore, one can treat them further if desired.
#[inline(always)]
pub fn masked_univariate_kernel_f64_std<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    let mut out_mask = mask.clone();

    process_scalar_masked_f64_std(x, mask, 0, &mut out, &mut out_mask, &scalar_body);

    (out, out_mask)
}

/// Processes the scalar section for masked kernels.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_f64_std<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    start: usize,
    out: &mut Vec64<f64>,
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out.push(scalar_body(xi));
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// Dense kernel helper for u64->f64 kernels.
///
/// # Parameters
/// - `x`:      input `u64` data
/// - `has_mask`: whether to return an “all‐true” mask
/// - `simd_body`: given each chunk as `Simd<f64, N>`, produce `Simd<f64, N>`
/// - `scalar_body`: given each scalar `u64`, produce its `f64` result
#[inline(always)]
pub fn dense_univariate_kernel_u64_std<FScalar>(
    x: &[u64],
    has_mask: bool,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);

    for &ki in x {
        out.push(scalar_body(ki));
    }

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// Null‐aware masked kernel helper for u64->f64 kernels.
///
/// Propagates the input mask and leaves any “invalid” (null) lanes as NULL;
/// lanes that are in the mask but produce a non‐finite result in the scalar
/// or SIMD body also become NULL.
#[inline(always)]
pub fn masked_univariate_kernel_u64_std<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    let mut out_mask = mask.clone();

    process_scalar_masked_u64_std(x, mask, 0, &mut out, &mut out_mask, &scalar_body);

    (out, out_mask)
}

/// Processes the scalar section for masked `u64->f64` kernels.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_u64_std<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    start: usize,
    out: &mut Vec64<f64>,
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out.push(scalar_body(xi));
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}
