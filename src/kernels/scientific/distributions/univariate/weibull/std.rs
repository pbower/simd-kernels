// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Weibull Distribution Scalar Implementations** - *Reliability and Survival Analysis Foundation*
//!
//! Scalar implementations of Weibull distribution functions optimised for reliability engineering
//! and survival analysis applications. These implementations emphasise numerical stability across
//! the wide parameter ranges encountered in practical applications.

use minarrow::{Bitmask, FloatArray};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};

use crate::utils::has_nulls;

#[inline(always)]
pub fn weibull_pdf_std(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_pdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_scale = 1.0 / scale;
    let coeff = shape * inv_scale; // k/λ

    // scalar & SIMD kernels
    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            let t = xi * inv_scale; // x/λ
            let t_pow_k = t.powf(shape); // (x/λ)^k
            coeff * t.powf(shape - 1.0) * (-t_pow_k).exp()
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_pdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Weibull CDF: F(x; k, λ) = 1 − exp[−(x/λ)^k]  for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_cdf_std(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_cdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_scale = 1.0 / scale;

    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            1.0 - (-(xi * inv_scale).powf(shape)).exp()
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        // covers all true null mask plus no mask supplied case
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_cdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Weibull quantile (inverse CDF):
/// Q(p; k, λ) = λ · [−ln(1−p)]^(1/k),  p ∈ (0,1)
#[inline(always)]
pub fn weibull_quantile_std(
    p: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_quantile: invalid shape or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let inv_k = 1.0 / shape;

    let scalar_body = move |pi: f64| -> f64 {
        // 0.0 returns 0 , 1.0 returns inf
        // outside these ranges returns NaN
        if 0.0 < pi && pi < 1.0 {
            scale * (-(1.0 - pi).ln()).powf(inv_k)
        } else if pi == 0.0 {
            0.0
        } else if pi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some(); // create all-true mask if caller supplied one
        let (data, mask) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_quantile: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        // propagate the (possibly pre-existing) mask; we don’t add extra nulls
        null_mask: Some(out_mask),
    })
}
