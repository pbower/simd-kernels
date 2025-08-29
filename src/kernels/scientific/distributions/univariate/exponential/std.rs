// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Exponential PDF: f(x|λ) = λ·exp(-λ·x) for x ≥ 0, 0 otherwise.
/// Returns error if λ ≤ 0 or non-finite.
#[inline(always)]
pub fn exponential_pdf_std(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter checks
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_pdf: λ must be positive and finite".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    // scalar fallback body
    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            lambda * (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    // 2) Dense (no nulls) path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware path
    let mask_ref = null_mask.expect("exponential_pdf: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Exponential CDF: F(x|λ) = 1 – exp(–λ·x) for x ≥ 0, 0 otherwise.
/// Error if λ ≤ 0 or non‐finite.
#[inline(always)]
pub fn exponential_cdf_std(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter checks
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_cdf: λ must be positive and finite".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            1.0 - (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    // 2) Dense (no nulls) path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware masked path
    let mask_ref = null_mask.expect("exponential_cdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Exponential quantile (inverse CDF): Q(p|λ) = –ln(1–p)/λ, pure math
#[inline(always)]
pub fn exponential_quantile_std(
    p: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter checks
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_quantile: λ must be positive and finite".into(),
        ));
    }
    // 2) empty
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let scalar_body = |pi: f64| -((1.0 - pi).ln()) / lambda;

    // 3) dense path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask_out) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask_out,
        });
    }

    // 4) null-aware path
    let mask_ref = null_mask.expect("exponential_quantile: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
