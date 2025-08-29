// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// Laplace (double-exponential) distribution PDF, null-aware and SIMD-accelerated.
/// f(x; μ, b) = (1 / (2b)) · exp(−|x − μ| / b)
#[inline(always)]
pub fn laplace_pdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_pdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_b = 1.0 / scale;
    let norm = 0.5 * inv_b; // 1/(2b)

    let scalar_body = |xi: f64| {
        let z = (xi - location).abs() * inv_b;
        norm * (-z).exp()
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
    let mask_ref = null_mask.expect("laplace_pdf: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Laplace (double-exponential) distribution CDF, null-aware and SIMD-accelerated.
/// F(x; μ, b) = 0.5·exp((x−μ)/b)          if x < μ
///           = 1 − 0.5·exp(−(x−μ)/b)     if x ≥ μ
#[inline(always)]
pub fn laplace_cdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_cdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_b = 1.0 / scale;

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
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
    let mask_ref = null_mask.expect("laplace_cdf: null_count > 0 requires null_mask");

    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Laplace quantile (inverse CDF), null-aware and SIMD-accelerated.
/// Q(p; μ, b) = μ + b·ln(2p)           if p < 0.5
///           = μ − b·ln(2(1−p))       if p ≥ 0.5
#[inline(always)]
pub fn laplace_quantile_std(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_quantile: invalid location or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let scalar_body = move |pi: f64| -> f64 {
        if pi > 0.0 && pi < 1.0 && pi.is_finite() {
            if pi < 0.5 {
                location + scale * (2.0 * pi).ln()
            } else {
                location - scale * (2.0 * (1.0 - pi)).ln()
            }
        } else if pi == 0.0 {
            f64::NEG_INFINITY
        } else if pi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("laplace_quantile: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
