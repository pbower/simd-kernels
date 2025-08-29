// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! Standard (scalar) implementation of Gumbel distribution functions.
//!
//! This module provides scalar implementations of the Gumbel (Type I extreme value)
//! distribution's probability density function (PDF), cumulative distribution function (CDF),
//! and quantile function. These implementations serve as the fallback when SIMD is not
//! available and are optimised for scalar computation patterns.

use minarrow::{Bitmask, FloatArray};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Gumbel (type I extreme value)
#[inline(always)]
pub fn gumbel_pdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_pdf: invalid location or scale".into(),
        ));
    }
    // 2) Empty input
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_b = 1.0 / scale;

    // scalar fallback and for the tail
    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        inv_b * (-(z + (-z).exp())).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask_out) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask_out,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_pdf: null_count > 0 requires null_mask");

    let (data, mask_out) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(mask_out),
    })
}

/// CDF of the Gumbel (type I extreme) distribution:
/// F(x; μ, β) = exp(−exp(−z)),   z = (x − μ)/β
#[inline(always)]
pub fn gumbel_cdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_cdf: invalid location or scale".into(),
        ));
    }
    // Empty input
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_b = 1.0 / scale;

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        (-(-z).exp()).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask_out) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask_out,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_cdf: null_count > 0 requires null_mask");

    let (data, mask_out) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(mask_out),
    })
}

/// Gumbel quantile (inverse CDF), null-aware and SIMD-accelerated.
/// Q(p; μ, β) = μ − β · ln(−ln(p))
#[inline(always)]
pub fn gumbel_quantile_std(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter validation
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_quantile: invalid location or scale".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let scalar_body = |pi: f64| -> f64 { location - scale * (-pi.ln()).ln() };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask_out) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask_out,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_quantile: null_count > 0 requires null_mask");

    let (data, mask_out) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(mask_out),
    })
}
