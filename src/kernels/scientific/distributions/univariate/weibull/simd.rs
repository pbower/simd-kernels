// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Weibull Distribution SIMD Implementations** - *Vectorised Reliability Engineering*
//!
//! High-performance SIMD-accelerated implementations of Weibull distribution function.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// **Weibull Distribution Probability Density Function** - *SIMD-Accelerated Reliability PDF*
///
/// Computes the probability density function of the Weibull distribution using vectorised SIMD operations
/// where possible, with automatic scalar fallback for optimal performance in reliability engineering applications.
///
/// ## Mathematical Definition
///
/// The Weibull probability density function is defined as:
///
/// ```text
/// f(x|k,λ) = {
///   (k/λ) × (x/λ)^(k-1) × exp(-(x/λ)^k)   if x ≥ 0
///   0                                     if x < 0
/// }
/// ```
///
/// Where:
/// - `x` ∈ [0,+∞): random variable (input values, non-negative)
/// - `k` > 0: shape parameter (determines distribution shape)
/// - `λ` > 0: scale parameter (characteristic life)
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where PDF is evaluated (x ≥ 0 for non-zero density)
/// * `shape` - Shape parameter (k), must be positive and finite
/// * `scale` - Scale parameter (λ), must be positive and finite  
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn weibull_pdf_simd(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_pdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let inv_scale = 1.0 / scale;
    let coeff = shape * inv_scale; // k/λ

    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            let t = xi * inv_scale;
            let t_pow_k = t.powf(shape);
            coeff * t.powf(shape - 1.0) * (-t_pow_k).exp()
        }
    };

    let simd_body = {
        let shape_v = Simd::<f64, N>::splat(shape);
        let inv_s_v = Simd::<f64, N>::splat(inv_scale);
        let coeff_v = Simd::<f64, N>::splat(coeff);
        let one_v = Simd::<f64, N>::splat(1.0);
        let zero_v = Simd::<f64, N>::splat(0.0);
        move |x_v: Simd<f64, N>| {
            let is_nan = x_v.simd_ne(x_v);
            let positive = x_v.simd_ge(zero_v);
            let t = x_v * inv_s_v;
            let ln_t = t.ln();
            let t_pow_k = (shape_v * ln_t).exp();
            let t_pow_km1 = ((shape_v - one_v) * ln_t).exp();
            let pdf_pos = coeff_v * t_pow_km1 * (-t_pow_k).exp();
            is_nan.select(Simd::splat(f64::NAN), positive.select(pdf_pos, zero_v))
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            let (data, mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
                x,
                null_mask.is_some(),
                simd_body,
                scalar_body,
            );
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        } else {
            let (data, mask) = dense_univariate_kernel_f64_std(x, null_mask.is_some(), scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
    }

    let mask_ref = null_mask.expect("weibull_pdf: null_count > 0 requires null_mask");
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    } else {
        let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    }
}

/// **Weibull Distribution Cumulative Distribution Function** - *SIMD-Accelerated Reliability CDF*
///
/// Computes the cumulative distribution function of the Weibull distribution using vectorised SIMD operations
/// where possible, providing high-precision probability computation for reliability and survival analysis.
///
/// ## Mathematical Definition
///
/// The Weibull cumulative distribution function is defined as:
///
/// ```text
/// F(x|k,λ) = {
///   1 - exp(-(x/λ)^k)   if x ≥ 0
///   0                   if x < 0
/// }
/// ```
///
/// Where:
/// - `x` ∈ [0,+∞): quantile value (input values, non-negative)
/// - `k` > 0: shape parameter (determines distribution shape)
/// - `λ` > 0: scale parameter (characteristic life)
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where CDF is evaluated
/// * `shape` - Shape parameter (k), must be positive and finite
/// * `scale` - Scale parameter (λ), must be positive and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with CDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn weibull_cdf_simd(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_cdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let inv_scale = 1.0 / scale;

    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            1.0 - (-(xi * inv_scale).powf(shape)).exp()
        }
    };

    let simd_body = {
        let shape_v = Simd::<f64, N>::splat(shape);
        let inv_s_v = Simd::<f64, N>::splat(inv_scale);
        let one_v = Simd::<f64, N>::splat(1.0);
        let zero_v = Simd::<f64, N>::splat(0.0);
        move |x_v: Simd<f64, N>| {
            let positive = x_v.simd_ge(zero_v);
            let t = x_v * inv_s_v;
            let t_pow = (shape_v * t.ln()).exp();
            let cdf_pos = one_v - (-t_pow).exp();
            positive.select(cdf_pos, zero_v)
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            let (data, mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
                x,
                null_mask.is_some(),
                simd_body,
                scalar_body,
            );
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        } else {
            let (data, mask) = dense_univariate_kernel_f64_std(x, null_mask.is_some(), scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
    }

    let mask_ref = null_mask.expect("weibull_cdf: null_count > 0 requires null_mask");
    if is_simd_aligned(x) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(x, mask_ref, simd_body, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    } else {
        let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    }
}

/// **Weibull Distribution Quantile Function** - *SIMD-Accelerated Reliability Quantile*
///
/// Computes the quantile function (inverse CDF) of the Weibull distribution using vectorised SIMD operations
/// where possible, providing efficient random variate generation and percentile computation for reliability analysis.
///
/// ## Mathematical Definition
///
/// The Weibull quantile function is defined as:
///
/// ```text
/// Q(p|k,λ) = {
///   λ × [-ln(1-p)]^(1/k)   if 0 < p < 1
///   0                      if p = 0
///   +∞                     if p = 1
///   NaN                    otherwise
/// }
/// ```
///
/// Where:
/// - `p` ∈ [0,1]: probability value (input values)
/// - `k` > 0: shape parameter (determines distribution shape)
/// - `λ` > 0: scale parameter (characteristic life)
///
/// ## Parameters
///
/// * `p` - Input data slice of `f64` probability values where quantile function is evaluated
/// * `shape` - Shape parameter (k), must be positive and finite
/// * `scale` - Scale parameter (λ), must be positive and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with quantile values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn weibull_quantile_simd(
    p: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_quantile: invalid shape or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let inv_k = 1.0 / shape;

    let scalar_body = move |pi: f64| -> f64 {
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

    let simd_body = {
        let invk_v = Simd::<f64, N>::splat(inv_k);
        let scale_v = Simd::<f64, N>::splat(scale);
        let zero_v = Simd::<f64, N>::splat(0.0);
        let one_v = Simd::<f64, N>::splat(1.0);
        let nan_v = Simd::<f64, N>::splat(f64::NAN);
        move |p_v: Simd<f64, N>| {
            let is_zero = p_v.simd_eq(zero_v);
            let is_one = p_v.simd_eq(one_v);
            let is_valid = p_v.simd_gt(zero_v) & p_v.simd_lt(one_v) & p_v.is_finite();
            let t = -(one_v - p_v).ln();
            let q_v = scale_v * (invk_v * t.ln()).exp();
            let inf_v = Simd::<f64, N>::splat(f64::INFINITY);
            is_zero.select(zero_v, is_one.select(inf_v, is_valid.select(q_v, nan_v)))
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(p) {
            let (data, mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
                p,
                null_mask.is_some(),
                simd_body,
                scalar_body,
            );
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        } else {
            let (data, mask) = dense_univariate_kernel_f64_std(p, null_mask.is_some(), scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
    }

    let mask_ref = null_mask.expect("weibull_quantile: null_count > 0 requires null_mask");
    if is_simd_aligned(p) {
        let (data, out_mask) =
            masked_univariate_kernel_f64_simd::<N, _, _>(p, mask_ref, simd_body, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    } else {
        let (data, out_mask) = masked_univariate_kernel_f64_std(p, mask_ref, scalar_body);
        Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        })
    }
}
