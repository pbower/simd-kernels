// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Normal Distribution Scalar Implementations** - *Foundation for Statistical Computing*
//!
//! Scalar (non-SIMD) implementations of normal distribution functions providing the computational 
//! foundation for statistical analysis.
use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::shared::scalar::normal_quantile_scalar;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::erf::erf;
use crate::utils::has_nulls;

/// Normal PDF (vectorised, SIMD where available), null-aware and Arrow-compliant.
/// Propagates input nulls and sets output null for any non-finite result.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
/// Normal PDF (vectorised, SIMD where available), null-aware.
/// Propagates input nulls and sets null on any non-finite output.
/// Normal PDF (vectorised, SIMD where available), null-aware and Arrow-compliant.
/// Propagates input nulls and sets output null for any non-finite result.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn normal_pdf_std(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if std <= 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(KernelError::InvalidArguments(
            "normal_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_sigma = 1.0 / std;
    let norm = inv_sigma / SQRT_2PI;

    let scalar_body = |xi: f64| {
        let z = (xi - mean) * inv_sigma;
        norm * (-0.5 * z * z).exp()
    };

    // Dense path
    if !has_nulls(null_count, null_mask) {
        let (out, out_mask) = dense_univariate_kernel_f64_std(x, null_mask.is_some(), scalar_body);

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked kernel path
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    let (out, out_mask) = masked_univariate_kernel_f64_std(x, mask, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Normal CDF (vectorised, SIMD where available).
/// Propagates input nulls and sets output null for any non-finite result.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn normal_cdf_std(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if std <= 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(KernelError::InvalidArguments(
            "normal_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_sigma = 1.0 / std;

    let scalar_body = |xi| {
        let z = (xi - mean) * inv_sigma / SQRT_2;
        0.5 * (1.0 + erf(z))
    };

    if !has_nulls(null_count, null_mask) {
        let (out, out_mask) = dense_univariate_kernel_f64_std(x, null_mask.is_some(), scalar_body);

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked kernel path
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    let (out, out_mask) = masked_univariate_kernel_f64_std(x, mask, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
/// https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
#[inline(always)]
pub fn normal_quantile_std(
    p: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if std <= 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(KernelError::InvalidArguments(
            "normal_quantile: invalid parameters".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Stable left-tail helper: Φ^{-1}(p) for tiny p (standard normal),
    // using asymptotic expansion + 2 Halley refinements with a Mills-ratio
    // CDF approximation to avoid catastrophic cancellation.
    #[inline(always)]
    fn inv_norm_left_tail_tiny(p: f64) -> f64 {
        debug_assert!(p > 0.0 && p < 0.5);
        const LN_4PI: f64 = 2.531_024_246_969_290_7_f64;
        // Initial asymptotic guess
        let u = -2.0 * p.ln();
        let t = u.sqrt();
        let lu = u.ln();
        let mut z = -(t - (lu + LN_4PI) / (2.0 * t));

        // Two Halley steps with tail-safe CDF approx:
        // Φ(z) (z<<0) ≈ φ(z) * (1/x - 1/x^3 + 3/x^5 - 15/x^7 + 105/x^9), x=-z>0
        for _ in 0..2 {
            let pdf = (-0.5 * z * z).exp() / SQRT_2PI; // φ(z) (safe here)
            let x = -z;
            let inv = 1.0 / x;
            let inv2 = inv * inv;
            let series = inv - inv * inv2 + 3.0 * inv * inv2 * inv2
                - 15.0 * inv * inv2 * inv2 * inv2
                + 105.0 * inv * inv2 * inv2 * inv2 * inv2;
            let cdf_approx = pdf * series;
            let f = cdf_approx - p; // Φ(z) - p (approx)
            let fp = pdf; // φ(z)
            let r = f / fp; // Newton ratio
            // Halley: z_{n+1} = z - r / (1 - 0.5 * r * f''/f')
            // here f''/f' = -z
            let denom = 1.0 + 0.5 * r * z;
            z -= r / denom;
        }
        z
    }

    // Right tail via symmetry.
    #[inline(always)]
    fn inv_norm_right_tail_tiny(q: f64) -> f64 {
        // q = 1 - p is tiny → Φ^{-1}(p) = -Φ^{-1}(q)
        -inv_norm_left_tail_tiny(q)
    }

    // Cutoffs where we switch to the asymptotic path.
    // (Acklam is excellent, but we guard the extreme tails to avoid any NaN/rounding issues.)
    const TINY: f64 = 1e-280;

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &pi in p {
            let z = if !pi.is_finite() || pi < 0.0 || pi > 1.0 {
                f64::NAN
            } else if pi == 0.0 {
                f64::NEG_INFINITY
            } else if pi == 1.0 {
                f64::INFINITY
            } else if pi < TINY {
                inv_norm_left_tail_tiny(pi)
            } else if (1.0 - pi) < TINY {
                inv_norm_right_tail_tiny(1.0 - pi)
            } else {
                // Regular high-accuracy path (Acklam)
                normal_quantile_scalar(pi, 0.0, 1.0)
            };
            out.push(mean + std * z);
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(len);

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
        } else {
            let pi = unsafe { *p.get_unchecked(idx) };
            let z = if !pi.is_finite() || pi < 0.0 || pi > 1.0 {
                f64::NAN
            } else if pi == 0.0 {
                f64::NEG_INFINITY
            } else if pi == 1.0 {
                f64::INFINITY
            } else if pi < TINY {
                inv_norm_left_tail_tiny(pi)
            } else if (1.0 - pi) < TINY {
                inv_norm_right_tail_tiny(1.0 - pi)
            } else {
                normal_quantile_scalar(pi, 0.0, 1.0)
            };
            out.push(mean + std * z);
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
