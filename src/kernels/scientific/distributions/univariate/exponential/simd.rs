// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Exponential Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of exponential distribution functions
//! utilising vectorised transcendental operations for bulk computations on aligned data.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use minarrow::{Bitmask, FloatArray};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::{
    errors::KernelError,
    kernels::scientific::distributions::univariate::common::simd::{
        dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
    },
};

use crate::utils::{has_nulls, is_simd_aligned};

/// SIMD-accelerated implementation of exponential distribution probability density function.
///
/// Processes multiple PDF evaluations simultaneously using vectorised exponential
/// operations for great performance on large datasets with 64-byte memory alignment.
///
/// ## Mathematical Definition
/// ```text
/// f(x; λ) = λ·exp(-λx)  for x ≥ 0
///         = 0           otherwise
/// ```
/// where λ is the rate parameter (λ > 0).
///
/// ## Parameters
/// - `x`: Input values to evaluate (requires 64-byte alignment for SIMD path)
/// - `lambda`: Rate parameter λ (must be positive and finite)
/// - `null_mask`: Optional bitmask for null value handling
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing PDF values, or error for invalid parameters.
#[inline(always)]
pub fn exponential_pdf_simd(
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

    const N: usize = W64;
    // pre‐splat for SIMD
    let lambda_v = Simd::<f64, N>::splat(lambda);
    let zero_v = Simd::<f64, N>::splat(0.0);

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

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
        let valid = x_v.simd_ge(zero_v);
        let pdf = lambda_v * (-lambda_v * x_v).exp();
        let result = valid.select(pdf, zero_v);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
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

        // Scalar fallback - alignment check failed
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware path
    let mask_ref = null_mask.expect("exponential_pdf: null_count > 0 requires null_mask");

    // Check if arrays are SIMD 64-byte aligned
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }

    // Scalar fallback - alignment check failed
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// SIMD-accelerated implementation of exponential distribution cumulative distribution function.
///
/// Processes multiple CDF evaluations simultaneously using vectorised exponential
/// operations for great performance on large datasets with 64-byte memory alignment.
///
/// ## Mathematical Definition
/// ```text
/// F(x; λ) = 1 - exp(-λx)  for x ≥ 0
///         = 0              otherwise
/// ```
/// where λ is the rate parameter (λ > 0).
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing CDF values, or error for invalid parameters.
#[inline(always)]
pub fn exponential_cdf_simd(
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

    const N: usize = W64;

    // SIMD splats
    let lambda_v = Simd::<f64, N>::splat(lambda);
    let zero_v = Simd::<f64, N>::splat(0.0);
    let one_v = Simd::<f64, N>::splat(1.0);

    // scalar fallback
    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            1.0 - (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
        let valid = x_v.simd_ge(zero_v);
        let y = one_v - (-lambda_v * x_v).exp();
        let result = valid.select(y, zero_v);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
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
        // Scalar fallback - alignment check failed
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware masked path
    let mask_ref = null_mask.expect("exponential_cdf: null_count > 0 requires null_mask");

    // Check if arrays are 64-byte aligned for SIMD
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }

    // Scalar fallback - alignment check failed
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// SIMD-accelerated implementation of exponential distribution quantile function.
///
/// Processes multiple quantile evaluations simultaneously using vectorised logarithmic
/// operations for great performance on large datasets with 64-byte memory alignment.
///
/// ## Mathematical Definition
/// ```text
/// Q(p; λ) = -ln(1 - p) / λ
/// ```
/// where λ is the rate parameter (λ > 0) and p ∈ [0, 1].
///
/// ## Parameters
/// - `p`: Probability values to invert (requires 64-byte alignment for SIMD path)
/// - `lambda`: Rate parameter λ (must be positive and finite)
/// - `null_mask`: Optional bitmask for null value handling
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing quantile values, or error for invalid parameters.
#[inline(always)]
pub fn exponential_quantile_simd(
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

    const N: usize = W64;

    let lambda_v = Simd::<f64, N>::splat(lambda);
    let scalar_body = |pi: f64| -((1.0 - pi).ln()) / lambda;
    let simd_body = move |p_v: Simd<f64, N>| -((Simd::splat(1.0) - p_v).ln()) / lambda_v;

    // 3) dense path
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(p) {
            let has_mask = null_mask.is_some();
            let (data, mask_out) = dense_univariate_kernel_f64_simd::<N, _, _>(
                p,
                has_mask,
                // SIMD body: pure math (ln(0)→inf, ln(neg)→NaN, ln(1)→0)
                simd_body,
                scalar_body,
            );
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask_out,
            });
        }

        // Scalar fallback - alignment check failed
        let has_mask = null_mask.is_some();
        let (data, mask_out) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask_out,
        });
    }

    // 4) null-aware path
    let mask_ref = null_mask.expect("exponential_quantile: null_count > 0 requires null_mask");

    // Check if arrays are SIMD 64-byte aligned
    if is_simd_aligned(p) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(p, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }

    // Scalar fallback - alignment check failed
    let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
