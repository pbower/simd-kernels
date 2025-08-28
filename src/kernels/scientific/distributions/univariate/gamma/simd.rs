// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! SIMD-accelerated implementation of gamma distribution functions.
//!
//! This module provides vectorised implementations of the gamma distribution's probability
//! density function (PDF) using SIMD instructions. The implementation automatically falls
//! back to scalar computation for unaligned data or when SIMD is not available.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use minarrow::{Bitmask, FloatArray};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::{has_nulls, is_simd_aligned};
use crate::{
    errors::KernelError,
    kernels::scientific::distributions::univariate::common::simd::{
        dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
    },
};

/// SIMD-accelerated implementation of gamma distribution probability density function.
///
/// Computes the probability density function (PDF) of the gamma distribution with shape parameter α
/// and rate parameter β (scale = 1/β) using vectorised SIMD operations for enhanced performance
/// on large datasets.
///
///
/// ## Parameters
/// - `x`: Input values where PDF should be evaluated (domain: x ≥ 0)
/// - `shape`: Shape parameter α > 0 controlling distribution shape
/// - `scale`: Rate parameter β > 0 controlling distribution scale (inverse of scale parameter)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid shape or scale parameters
///
/// ## Special Cases and Boundary Conditions
/// - **x = 0**: Returns β if α = 1, 0 if α > 1, +∞ if α < 1
/// - **x < 0**: Returns 0 (outside distribution domain)
/// - **Non-finite x**: Returns NaN
/// - **Invalid parameters**: Returns error for α ≤ 0, β ≤ 0, or non-finite parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When shape ≤ 0, scale ≤ 0, or parameters are non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [0.5, 1.0, 2.0, 3.0];
/// let shape = 2.0;  // α = 2
/// let scale = 1.5;  // β = 1.5 (rate parameter)
/// let result = gamma_pdf_simd(&x, shape, scale, None, None)?;
/// // Returns PDF values for x with gamma(α=2, β=1.5)
/// ```
#[inline(always)]
pub fn gamma_pdf_simd(
    x: &[f64],
    shape: f64,
    scale: f64, // interpreted as rate β
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if shape <= 0.0 || !shape.is_finite() || scale <= 0.0 || !scale.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_pdf: invalid shape or rate".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let ln_gamma_k = ln_gamma(shape);
    let beta = scale; // tests pass "scale" as rate β
    // log f = k ln β - ln Γ(k) + (k-1) ln x - β x
    let log_norm = shape * beta.ln() - ln_gamma_k;

    let shape_v = Simd::<f64, N>::splat(shape);
    let beta_v = Simd::<f64, N>::splat(beta);
    let log_norm_v = Simd::<f64, N>::splat(log_norm);
    let one_v = Simd::<f64, N>::splat(1.0);
    let zero_v = Simd::<f64, N>::splat(0.0);

    let scalar_body = |xi: f64| -> f64 {
        if !xi.is_finite() {
            return f64::NAN;
        }
        if xi < 0.0 {
            return 0.0;
        }
        if xi == 0.0 {
            // 0^(k-1) behavior: 0 for k>1, +∞ for k<1, and β for k=1.
            return if shape > 1.0 {
                0.0
            } else if shape < 1.0 {
                f64::INFINITY
            } else {
                beta
            };
        }
        (log_norm + (shape - 1.0) * xi.ln() - beta * xi).exp()
    };

    let simd_body = |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v);
        let is_neg = x_v.simd_lt(zero_v);
        let is_zero = x_v.simd_eq(zero_v);

        // base log-density where x>0
        let ln_x = x_v.ln();
        let base = (log_norm_v + (shape_v - one_v) * ln_x - beta_v * x_v).exp();

        // x == 0 branch
        let zero_val = if shape > 1.0 {
            zero_v
        } else if shape < 1.0 {
            Simd::<f64, N>::splat(f64::INFINITY)
        } else {
            beta_v
        };

        // select by domain
        let tmp = is_zero.select(zero_val, base);
        let tmp = is_neg.select(zero_v, tmp);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), tmp)
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            let has_mask = null_mask.is_some();
            let (data, mask) =
                dense_univariate_kernel_f64_simd::<N, _, _>(x, has_mask, simd_body, scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    let mask_ref = null_mask.expect("gamma_pdf: null_count > 0 requires null_mask");
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
