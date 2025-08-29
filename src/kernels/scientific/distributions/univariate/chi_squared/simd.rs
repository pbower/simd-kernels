// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Chi-Squared Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of chi-squared distribution functions
//! utilising vectorised operations for bulk computations on 64-byte aligned data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the chi-squared distribution that process multiple values simultaneously
//! using CPU vector instructions. The implementations automatically fall back to
//! scalar versions when data alignment requirements are not met.
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// SIMD-accelerated implementation of chi-squared distribution probability density function.
///
/// Processes multiple PDF evaluations simultaneously using vectorised logarithmic
/// computations for great performance on large datasets with 64-byte memory alignment.
///
/// ## Mathematical Definition
/// ```text
/// f(x; k) = 1/(2^(k/2) · Γ(k/2)) · x^(k/2-1) · e^(-x/2)
/// ```
/// where k is the degrees of freedom parameter.
///
/// ## Parameters
/// - `x`: Input values to evaluate (requires 64-byte alignment for SIMD path)
/// - `df`: Degrees of freedom k (must be positive and finite)
/// - `null_mask`: Optional bitmask for null value handling
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing PDF values, or error for invalid parameters.
#[inline(always)]
pub fn chi_square_pdf_simd(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter check
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_pdf: invalid df".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let k2 = 0.5 * df;
    let log_norm = -k2 * std::f64::consts::LN_2 - ln_gamma(k2);

    // pre‐splat for SIMD

    let k2m1_v = Simd::<f64, N>::splat(k2 - 1.0);
    let log_norm_v = Simd::<f64, N>::splat(log_norm);
    let zero_v = Simd::<f64, N>::splat(0.0);
    let neg_half_v = Simd::<f64, N>::splat(-0.5);

    // scalar fallback body
    let scalar_body = move |xi: f64| {
        if xi < 0.0 {
            0.0
        } else if xi == 0.0 && k2 < 1.0 {
            f64::INFINITY // Special case for df < 2
        } else if xi == 0.0 && k2 == 1.0 {
            log_norm.exp() // Special case for df = 2, returns 0.5
        } else if xi == 0.0 {
            0.0 // General case for df > 2
        } else if xi.is_infinite() {
            0.0 // For infinity
        } else if xi.is_nan() {
            f64::NAN // Preserve NaN for null propagation
        } else {
            (log_norm + (k2 - 1.0) * xi.ln() - 0.5 * xi).exp()
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_negative = x_v.simd_lt(zero_v);
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
        let is_infinite = x_v.is_infinite();
        let is_zero = x_v.simd_eq(zero_v);

        // For finite positive values, compute normally
        let lnx = x_v.ln();
        let expv = neg_half_v * x_v;
        let y = (log_norm_v + k2m1_v * lnx + expv).exp();

        // Special case for x=0: depends on df
        // When x=0 and df=2 (k2=1), result should be exp(log_norm) = 0.5
        // When x=0 and df<2 (k2<1), result should be INFINITY
        // When x=0 and df>2 (k2>1), result should be 0
        let zero_result = if k2 == 1.0 {
            Simd::<f64, N>::splat(log_norm.exp())
        } else if k2 < 1.0 {
            Simd::<f64, N>::splat(f64::INFINITY)
        } else {
            zero_v
        };

        // Handle special cases - preserve NaN for null handling
        let nan_v = Simd::<f64, N>::splat(f64::NAN);
        let result = is_nan.select(
            nan_v,
            is_negative.select(
                zero_v,
                is_infinite.select(zero_v, is_zero.select(zero_result, y)),
            ),
        );

        result
    };

    // 2) Dense (no nulls) path
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(x) {
            let has_mask = null_mask.is_some();
            let (data, mask) =
                dense_univariate_kernel_f64_simd::<N, _, _>(x, has_mask, simd_body, scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }

        // Scalar fallback (used when SIMD disabled OR alignment check failed)
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware path
    let mask_ref = null_mask.expect("chi_square_pdf: null_count > 0 requires null_mask");

    // Check if arrays are SIMD 64-byte aligned
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }

    // Scalar fallback (used when SIMD disabled OR alignment check failed)
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
