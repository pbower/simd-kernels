// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Student's t-Distribution Scalar Implementations** - *Small-Sample Statistical Foundation*
//!
//! Scalar implementations of Student's t-distribution functions providing the computational
//! foundation for statistical inference with unknown population variance.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Student‐t PDF
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn student_t_pdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) argument checks
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_pdf: invalid df".into(),
        ));
    }
    // 2) precompute the normalising constant
    let coeff =
        (ln_gamma((df + 1.0) * 0.5) - ln_gamma(df * 0.5) - 0.5 * (df * std::f64::consts::PI).ln())
            .exp();
    // 3) empty‐input fast path
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // scalar fallback per‐element
    let scalar_body =
        move |xi: f64| -> f64 { coeff * (1.0 + xi * xi / df).powf(-(df + 1.0) * 0.5) };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null‐aware masked path
    let mask_ref = null_mask.expect("student_t_pdf: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Student-t CDF
/// - Algorithm AS 3 (Hill, 1970) / Algorithm 395 (G. W. Hill, 1970)
/// - https://www.jstor.org/stable/2346841
///
/// Scalar implementation uses the incomplete beta via continued fraction.
/// Returns error if df < 1.
#[inline(always)]
pub fn student_t_cdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if df < 1.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_cdf: invalid df".into(),
        ));
    }

    let len = x.len();
    let mut out = Vec64::with_capacity(len);

    if !has_nulls(null_count, null_mask) {
        // Dense (non-null) fast path
        let half_df = 0.5 * df;
        let t = df;
        let a = half_df;
        let b = 0.5;
        for &xi in x {
            let xb = t / (t + xi * xi);
            let ib = incomplete_beta(a, b, xb);
            let sign = if xi >= 0.0 { 1.0 } else { -1.0 };
            let res = 0.5 + sign * 0.5 * (1.0 - ib);
            out.push(res);
        }
        Ok(FloatArray::from_vec64(out, null_mask.cloned()))
    } else {
        // Null-aware path, propagate mask and set NaN for nulls
        let mask = null_mask.expect("non-dense null path requires mask");
        let half_df = 0.5 * df;
        let t = df;
        let a = half_df;
        let b = 0.5;

        for idx in 0..len {
            if !unsafe { mask.get_unchecked(idx) } {
                out.push(f64::NAN);
            } else {
                let xi = unsafe { *x.get_unchecked(idx) };
                let xb = t / (t + xi * xi);
                let ib = incomplete_beta(a, b, xb);
                let sign = if xi >= 0.0 { 1.0 } else { -1.0 };
                let res = 0.5 + sign * 0.5 * (1.0 - ib);
                out.push(res);
            }
        }
        Ok(FloatArray {
            data: out.into(),
            null_mask: Some(mask.clone()),
        })
    }
}

/// Student-t quantile (inverse CDF), using the AS 241 algorithm
/// Reference: Michael J. Wichura, Algorithm AS 241: The Percentage Points of the Normal Distribution
#[inline(always)]
pub fn student_t_quantile_std(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if df < 1.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_quantile: invalid df".into(),
        ));
    }
    let len = p.len();
    let mut out = Vec64::with_capacity(len);

    // Closure for per-probability quantile computation
    let compute_quantile = |prob: f64| -> f64 {
        // 0 becomes -inf, 1 becomes inf
        // outside of these, is NaN
        if !(prob >= 0.0 && prob <= 1.0) || !prob.is_finite() {
            f64::NAN
        } else if prob == 0.0 {
            f64::NEG_INFINITY
        } else if prob == 1.0 {
            f64::INFINITY
        } else if df > 1e7 {
            normal_quantile_scalar(prob, 0.0, 1.0)
        } else {
            let a = 0.5 * df;
            let b = 0.5;
            let x = if prob < 0.5 {
                incomplete_beta_inv(a, b, 2.0 * prob)
            } else {
                incomplete_beta_inv(a, b, 2.0 * (1.0 - prob))
            };
            let t = ((df * (1.0 - x)) / x).sqrt();
            if prob < 0.5 { -t } else { t }
        }
    };

    if !has_nulls(null_count, null_mask) {
        for &prob in p {
            out.push(compute_quantile(prob));
        }
        Ok(FloatArray::from_vec64(out, null_mask.cloned()))
    } else {
        let mask = null_mask.expect("null path requires a mask");
        for idx in 0..len {
            if !unsafe { mask.get_unchecked(idx) } {
                out.push(f64::NAN);
            } else {
                let prob = unsafe { *p.get_unchecked(idx) };
                out.push(compute_quantile(prob));
            }
        }
        Ok(FloatArray {
            data: out.into(),
            null_mask: Some(mask.clone()),
        })
    }
}
