// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Logistic Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of logistic distribution functions
//! utilising vectorised operations for bulk computations on S-shaped statistical data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the logistic distribution, characterised by its symmetric S-shaped (sigmoid)
//! curve. The implementations automatically fall back to scalar versions when data
//! alignment requirements are not met.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::f64::{INFINITY, NEG_INFINITY};

use std::simd::{
    Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};

use crate::utils::{has_nulls, is_simd_aligned};

/// SIMD-accelerated implementation of logistic distribution probability density function.
///
/// Computes the probability density function (PDF) of the logistic distribution
/// using vectorised SIMD operations for enhanced performance on S-shaped
/// (sigmoid-related) statistical computations.
///
/// ## Applications
/// Commonly used in:
/// - Logistic regression and classification models
/// - Growth curve modelling (S-curves)
/// - Neural network activation functions
/// - Survival analysis and reliability engineering
///
/// ## Parameters
/// - `x`: Input values where PDF should be evaluated (domain: all real numbers)
/// - `location`: Location parameter μ (mean and median of distribution)
/// - `scale`: Scale parameter s > 0 (controls distribution spread)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid scale or location parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
/// let location = 0.0;  // μ = 0 (centred)
/// let scale = 1.0;     // s = 1 (standard scale)
/// let result = logistic_pdf_simd(&x, location, scale, None, None)?;
/// // Returns PDF values for standard logistic distribution
/// ```
#[inline(always)]
pub fn logistic_pdf_simd(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if scale <= 0.0 || !scale.is_finite() || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "logistic_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;

    // SIMD splats

    let loc_v = Simd::<f64, N>::splat(location);
    let scale_v = Simd::<f64, N>::splat(scale);
    let one_v = Simd::<f64, N>::splat(1.0);

    // Scalar fallback
    let scalar_body = move |xi: f64| -> f64 {
        let z = (xi - location) / scale;
        let ez = (-z).exp();
        ez / (scale * (1.0 + ez).powi(2))
    };

    // SIMD body
    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) / scale_v;
        let ez = (-z).exp();
        ez / (scale_v * (one_v + ez) * (one_v + ez))
    };

    // Dense fast path (no nulls)
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

    // Null-aware masked path
    let mask_ref = null_mask.expect("logistic_pdf: null_count > 0 requires null_mask");

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

/// SIMD-accelerated implementation of logistic distribution cumulative distribution function.
///
/// Computes the cumulative distribution function (CDF) of the logistic distribution
/// using vectorised SIMD operations for enhanced performance on S-shaped
/// (sigmoid) probability calculations.
///
/// ## Parameters
/// - `x`: Input values where CDF should be evaluated (domain: all real numbers)
/// - `location`: Location parameter μ (mean and median of distribution)
/// - `scale`: Scale parameter s > 0 (controls distribution spread)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed CDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid scale or location parameters
///
/// ## Special Cases and Boundary Conditions
/// - **x = μ**: CDF equals 0.5 (median point)
/// - **x → -∞**: CDF approaches 0
/// - **x → +∞**: CDF approaches 1
/// - **S-shaped curve**: Characteristic sigmoid shape with inflection at x = μ
/// - **Invalid parameters**: Returns error for s ≤ 0 or non-finite parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Use cases
/// The logistic CDF represents the probability of observing a value less than or
/// equal to x, essential for:
/// - Probability calculations in logistic regression
/// - Classification thresholds and decision boundaries
/// - Growth curve analysis (cumulative adoption, etc.)
/// - Risk assessment with S-shaped probability functions
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
/// let location = 0.0;  // μ = 0 (centred)
/// let scale = 1.0;     // s = 1 (standard scale)
/// let result = logistic_cdf_simd(&x, location, scale, None, None)?;
/// // Returns cumulative probabilities for standard logistic
/// ```
#[inline(always)]
pub fn logistic_cdf_simd(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if scale <= 0.0 || !scale.is_finite() || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "logistic_cdf: invalid parameters".into(),
        ));
    }
    // Empty input
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let inv_s = 1.0 / scale;

    // SIMD splats

    let loc_v = Simd::<f64, N>::splat(location);
    let inv_s_v = Simd::<f64, N>::splat(inv_s);
    let one_v = Simd::<f64, N>::splat(1.0);

    // Scalar fallback
    let scalar_body = move |xi: f64| -> f64 {
        let z = (xi - location) * inv_s;
        1.0 / (1.0 + (-z).exp())
    };

    // SIMD body
    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) * inv_s_v;
        one_v / (one_v + (-z).exp())
    };

    // Dense fast path (no nulls)
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

    // Null-aware masked path
    let mask_ref = null_mask.expect("null_count > 0 requires null_mask");

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

/// SIMD-accelerated implementation of logistic distribution quantile function.
///
/// Computes the quantile function (inverse CDF) of the logistic distribution
/// using vectorised SIMD operations for enhanced performance on inverse
/// sigmoid (logit) calculations.
///
/// ## Parameters
/// - `p`: Input probability values where quantiles should be evaluated (domain: p ∈ [0, 1])
/// - `location`: Location parameter μ (mean and median of distribution)
/// - `scale`: Scale parameter s > 0 (controls distribution spread)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed quantile values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid location or scale parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Use cases
/// Quantile calculations are essential for:
/// - **Confidence intervals**: Statistical bounds for logistic regression parameters
/// - **Classification thresholds**: Decision boundaries in machine learning models
/// - **Risk assessment**: Probability-to-outcome transformations
/// - **Simulation**: Inverse transform sampling for random number generation
///
/// ## Example Usage
/// ```rust,ignore
/// let p = [0.01, 0.25, 0.5, 0.75, 0.99];  // probability values
/// let location = 0.0;                      // μ = 0 (centred)
/// let scale = 1.0;                         // s = 1 (standard scale)
/// let result = logistic_quantile_simd(&p, location, scale, None, None)?;
/// // Returns quantile values for standard logistic distribution
/// ```
#[inline(always)]
pub fn logistic_quantile_simd(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if !location.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(KernelError::InvalidArguments(
            "logistic_quantile: invalid location or scale".into(),
        ));
    }
    // Empty input
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Constants
    const N: usize = W64;

    let loc_v = Simd::<f64, N>::splat(location);
    let scale_v = Simd::<f64, N>::splat(scale);
    let one_v = Simd::<f64, N>::splat(1.0);

    // Scalar fallback
    let scalar_body = move |pi: f64| -> f64 {
        // Q(p | μ, s) = μ + s · ln( p / (1-p) ) only for 0 < p < 1  and  p finite
        if pi > 0.0 && pi < 1.0 && pi.is_finite() {
            location + scale * (pi.ln() - (1.0 - pi).ln())
        } else if pi == 0.0 {
            NEG_INFINITY
        } else if pi == 1.0 {
            INFINITY
        } else {
            f64::NAN
        }
    };

    // SIMD body
    let simd_body = move |p_v: Simd<f64, N>| {
        let zero_v = Simd::<f64, N>::splat(0.0);
        let ninf_v = Simd::<f64, N>::splat(NEG_INFINITY);
        let pinf_v = Simd::<f64, N>::splat(INFINITY);
        let nan_v = Simd::<f64, N>::splat(f64::NAN);

        // Handle edge cases
        let is_zero = p_v.simd_eq(zero_v);
        let is_one = p_v.simd_eq(one_v);
        let is_valid = p_v.simd_gt(zero_v) & p_v.simd_lt(one_v) & p_v.is_finite();

        // Q(p | μ, s) = μ + s · ln( p / (1-p) ) for 0 < p < 1
        let qv = loc_v + scale_v * (p_v.ln() - (one_v - p_v).ln());

        // Select appropriate result: -inf for p=0, +inf for p=1, NaN for invalid, qv for valid
        is_zero.select(ninf_v, is_one.select(pinf_v, is_valid.select(qv, nan_v)))
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(p) {
            let has_mask = null_mask.is_some();
            let (data, mask) =
                dense_univariate_kernel_f64_simd::<N, _, _>(p, has_mask, simd_body, scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
        // Scalar fallback - alignment check failed
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(p, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("null_count > 0 requires null_mask");

    // Check if arrays are SIMD 64-byte aligned
    if is_simd_aligned(p) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd(p, mask_ref, simd_body, scalar_body);
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
