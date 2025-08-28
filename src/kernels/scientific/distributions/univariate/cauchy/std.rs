//! # **Cauchy Distribution Scalar Implementation**
//!
//! Scalar implementations using standard library transcendental functions.
//! PDF uses efficient rational function, CDF/quantile use atan/tan functions.

// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray, Vec64};
use std::f64::consts::PI;

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::constants::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// Cauchy PDF (vectorised, SIMD where available), null-aware and Arrow-compliant.
/// f(x; location, scale) = (1/π)·[scale / ((x − location)² + scale²)]
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn cauchy_pdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_pdf: invalid location or scale".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_scale = 1.0 / scale;
    let inv_ps = INV_PI * inv_scale;

    // scalar fallback
    let scalar_body = move |xi: f64| {
        let z = (xi - location) * inv_scale;
        inv_ps / (1.0 + z * z)
    };

    // Dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_pdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Cauchy CDF (scalar), null-aware and Arrow-compliant.
/// F(x; location, scale) = ½ + (1/π)·atan((x − location)/scale)
#[inline(always)]
pub fn cauchy_cdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Validate parameters
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_cdf: invalid location or scale".into(),
        ));
    }
    let len = x.len();
    // Empty input → output
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }
    let inv_scale = 1.0 / scale;

    // Fast dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &xi in x {
            let z = (xi - location) * inv_scale;
            out.push(0.5 + INV_PI * z.atan());
        }
        // clone 'None' when no bitmask supplied
        // otherwise all true for "null_count: 0"
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_cdf: null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(len);
    let mut out_mask = Bitmask::new_set_all(len, true);

    for idx in 0..len {
        if !mask.get(idx) {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = x[idx];
            let z = (xi - location) * inv_scale;
            let v = 0.5 + INV_PI * z.atan();
            out.push(v);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Cauchy quantile (inverse CDF): Q(p; location, scale) = location + scale · tan[π·(p - ½)]
#[inline(always)]
pub fn cauchy_quantile_std(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Validate parameters
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_quantile: invalid location or scale".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Fast dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &pi in p {
            if pi == 0.0 {
                out.push(f64::NEG_INFINITY);
            } else if pi == 1.0 {
                out.push(f64::INFINITY);
            } else if pi > 0.0 && pi < 1.0 {
                // Robust tails
                let t = pi.min(1.0 - pi);
                if t <= 1e-8 {
                    // asymptotic: tan(π(p-1/2)) = -cot(πp) ~ -1/(πp) for p→0
                    let s = if pi < 0.5 { -1.0 } else { 1.0 };
                    let tail = s / (std::f64::consts::PI * t);
                    out.push(location + scale * tail);
                } else {
                    // interior: use FMA to reduce cancellation
                    let angle = pi.mul_add(std::f64::consts::PI, -0.5 * std::f64::consts::PI);
                    out.push(location + scale * angle.tan());
                }
            } else {
                out.push(f64::NAN);
            }
        }
        return Ok(FloatArray::from_vec64(out, None));
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_quantile: null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(len);

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            // preserve input null
            out.push(f64::NAN);
        } else {
            let pi = p[idx];

            // honour -inf/inf at the endpoints; outside (0,1) → NaN
            if pi == 0.0 {
                out.push(f64::NEG_INFINITY);
            } else if pi == 1.0 {
                out.push(f64::INFINITY);
            } else if pi > 0.0 && pi < 1.0 {
                // robust tails
                let t = pi.min(1.0 - pi);
                if t <= 1e-8 {
                    // tan(π(p−½)) = −cot(πp) ≈ −1/(πp) for p→0
                    let s = if pi < 0.5 { -1.0 } else { 1.0 };
                    let tail = s / (PI * t);
                    out.push(location + scale * tail);
                } else {
                    // interior: fused multiply-add for better precision
                    let angle = pi.mul_add(PI, -0.5 * PI);
                    out.push(location + scale * angle.tan());
                }
            } else {
                // produce NaN but mark valid
                out.push(f64::NAN);
            }
        }
    }

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
