// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Uniform Distribution Scalar Implementations** - *Perfect Simplicity and Precision*
//!
//! Scalar implementations of uniform distribution functions.

use minarrow::{Bitmask, FloatArray};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// Uniform PDF: f(x|a,b) = 1/(b-a) for x in [a, b], 0 otherwise.
#[inline(always)]
pub fn uniform_pdf_std(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter checks
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_pdf: a must be < b and finite".into(),
        ));
    }
    // 2) empty‐input fast path
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_width = 1.0 / (b - a);

    let scalar_body = move |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= a && xi <= b {
            inv_width
        } else {
            0.0
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("uniform_pdf: null_count > 0 requires null_mask");

    // Scalar fallback - alignment check failed
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Uniform CDF: F(x|a,b) = 0 if x < a, (x-a)/(b-a) if x in [a,b], 1 if x > b.
#[inline(always)]
pub fn uniform_cdf_std(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_cdf: a must be < b and both finite".into(),
        ));
    }
    // empty input
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_width = 1.0 / (b - a);

    let scalar_body = |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi < a {
            0.0
        } else if xi > b {
            1.0
        } else {
            (xi - a) * inv_width
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("uniform_cdf: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Continuous-uniform quantile
#[inline(always)]
pub fn uniform_quantile_std(
    p: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_quantile: a must be < b and finite".into(),
        ));
    }
    // empty input
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let width = b - a;

    // scalar & SIMD bodies
    let scalar_body = move |pi: f64| -> f64 {
        if (0.0..=1.0).contains(&pi) && pi.is_finite() {
            a + pi * width
        } else {
            f64::NAN
        }
    };

    // dense path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // null-aware path
    let mask_ref = null_mask.expect("uniform_quantile: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
