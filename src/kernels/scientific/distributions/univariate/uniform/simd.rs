// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Uniform Distribution SIMD Implementations**
//!
//! High-performance SIMD-accelerated implementations of uniform distribution functions optimised
//! for Monte Carlo simulation, random sampling, and statistical modelling applications requiring
//! flat probability distributions over bounded intervals.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::univariate::common::simd::masked_univariate_kernel_f64_simd;
use crate::kernels::scientific::distributions::univariate::common::{
    simd::dense_univariate_kernel_f64_simd,
    std::{dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std},
};
use crate::utils::has_nulls;

/// **Uniform Distribution Probability Density Function**
///
/// Computes the probability density function of the continuous uniform distribution using vectorised
/// SIMD operations where possible, with scalar fallback for compatibility.
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where PDF is evaluated
/// * `a` - Lower bound of uniform distribution, must be finite
/// * `b` - Upper bound of uniform distribution, must be finite and b > a
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn uniform_pdf_simd(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_pdf: a must be < b and finite".into(),
        ));
    }
    // empty‐input fast path
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    let inv_width = 1.0 / (b - a);

    // SIMD body
    let simd_body = move |x_v: Simd<f64, N>| -> Simd<f64, N> {
        let a_v = Simd::<f64, N>::splat(a);
        let b_v = Simd::<f64, N>::splat(b);
        let inv_v = Simd::<f64, N>::splat(inv_width);
        let zero = Simd::<f64, N>::splat(0.0);
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
        let in_range = x_v.simd_ge(a_v) & x_v.simd_le(b_v);
        let result = in_range.select(inv_v, zero);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
    };

    // scalar fallback
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
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(x) {
            // even if a mask was supplied with null_count == 0, return an all‐true mask
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
    let mask_ref = null_mask.expect("uniform_pdf: null_count > 0 requires null_mask");

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

/// **Uniform Distribution Cumulative Distribution Function** - *SIMD-Accelerated Flat CDF*
///
/// Computes the cumulative distribution function of the continuous uniform distribution using vectorised
/// SIMD operations where possible, providing high-precision probability computation for bounded intervals.
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where CDF is evaluated
/// * `a` - Lower bound of uniform distribution, must be finite
/// * `b` - Upper bound of uniform distribution, must be finite and b > a
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with CDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn uniform_cdf_simd(
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

    const N: usize = W64;
    let inv_width = 1.0 / (b - a);

    // scalar fallback (one lane)
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

    // SIMD body (N lanes)

    let simd_body = {
        let a_v = Simd::<f64, N>::splat(a);
        let b_v = Simd::<f64, N>::splat(b);
        let inv_v = Simd::<f64, N>::splat(inv_width);
        let one = Simd::<f64, N>::splat(1.0);
        let zero = Simd::<f64, N>::splat(0.0);
        move |x_v: Simd<f64, N>| -> Simd<f64, N> {
            let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
            let below = x_v.simd_lt(a_v);
            let above = x_v.simd_gt(b_v);
            let linear = (x_v - a_v) * inv_v;
            let result = below.select(zero, above.select(one, linear));
            is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(x) {
            // if a mask was passed with `null_count==0`, still return an all-true mask
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
    let mask_ref = null_mask.expect("uniform_cdf: null_count > 0 requires null_mask");

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

/// **Uniform Distribution Quantile Function** - *SIMD-Accelerated Inverse CDF*
///
/// Computes the quantile function (inverse CDF) of the continuous uniform distribution using vectorised
/// SIMD operations where possible, providing efficient random variate generation and percentile computation.
///
/// ## Parameters
///
/// * `p` - Input data slice of `f64` probability values where quantile function is evaluated
/// * `a` - Lower bound of uniform distribution, must be finite
/// * `b` - Upper bound of uniform distribution, must be finite and b > a
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with quantile values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn uniform_quantile_simd(
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

    const N: usize = W64;
    let width = b - a;

    // scalar & SIMD bodies
    let scalar_body = move |pi: f64| -> f64 {
        if (0.0..=1.0).contains(&pi) && pi.is_finite() {
            a + pi * width
        } else {
            f64::NAN
        }
    };

    let simd_body = {
        let a_v = Simd::<f64, N>::splat(a);
        let width_v = Simd::<f64, N>::splat(width);
        let zero = Simd::<f64, N>::splat(0.0);
        let one = Simd::<f64, N>::splat(1.0);
        move |p_v: Simd<f64, N>| {
            let valid = p_v.simd_ge(zero) & p_v.simd_le(one) & p_v.is_finite();
            let q = a_v + p_v * width_v;
            valid.select(q, Simd::splat(f64::NAN))
        }
    };

    // dense path (no nulls)
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(p) {
            // includes 'all-true' mask or no mask
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

    // null-aware path
    let mask_ref = null_mask.expect("uniform_quantile: null_count > 0 requires null_mask");

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
