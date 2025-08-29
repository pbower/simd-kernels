// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Normal Distribution SIMD Implementations** - *Vectorised Statistical Foundation*
//!
//! High-performance SIMD-accelerated implementations of normal distribution functions leveraging
//! modern CPU vector instructions for maximum throughput on large statistical datasets.
//!
//! ## SIMD Vector Lane Configuration
//! - **Lane width**: configured by `W64` constant (8×f64 on AVX-512, 4×f64 on AVX2)
//! - **Memory alignment**: requires 64-byte aligned input arrays for optimal performance
//! - **Automatic fallback**: falls back to scalar implementation when alignment fails
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{Simd, StdFloat};

use minarrow::{Bitmask, FloatArray};

use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use minarrow::enums::error::KernelError;

use crate::kernels::scientific::erf::{erf, erf_simd};
use crate::utils::has_nulls;

/// **Normal Distribution Probability Density Function** - *SIMD-Accelerated Gaussian PDF*
///
/// Computes the probability density function of the normal distribution using vectorised SIMD operations
/// where possible, with scalar fallback for compatibility.
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where PDF is evaluated
/// * `mean` - Normal distribution mean (μ), must be finite
/// * `std` - Normal distribution standard deviation (σ), must be positive and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn normal_pdf_simd(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if std <= 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(KernelError::InvalidArguments(
            "normal_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Constants
    let inv_sigma = 1.0 / std;
    let norm = inv_sigma / SQRT_2PI;
    let mean_v = Simd::<f64, N>::splat(mean);
    let inv_sigma_v = Simd::<f64, N>::splat(inv_sigma);
    let norm_v = Simd::<f64, N>::splat(norm);

    let simd_body = |x_v: Simd<f64, N>| {
        let z = (x_v - mean_v) * inv_sigma_v;
        let exp = (-(z * z) * Simd::<f64, N>::splat(0.5)).exp();
        norm_v * exp
    };

    let scalar_body = |xi: f64| {
        let z = (xi - mean) * inv_sigma;
        norm * (-0.5 * z * z).exp()
    };

    // Dense path
    if !has_nulls(null_count, null_mask) {
        const N: usize = W64;

        let (out, out_mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
            x,
            null_mask.is_some(),
            simd_body,
            scalar_body,
        );

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked kernel path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    const N: usize = W64;

    let (out, out_mask) =
        masked_univariate_kernel_f64_simd::<N, _, _>(x, mask, simd_body, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// **Normal Distribution Cumulative Distribution Function** - *SIMD-Accelerated Gaussian CDF*
///
/// Computes the cumulative distribution function of the normal distribution using vectorised error function
/// evaluation with SIMD acceleration where possible, providing high-precision probability computation.
///
/// ## Parameters
///
/// * `x` - Input data slice of `f64` values where CDF is evaluated
/// * `mean` - Normal distribution mean (μ), must be finite
/// * `std` - Normal distribution standard deviation (σ), must be positive and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with CDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn normal_cdf_simd(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if std <= 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(KernelError::InvalidArguments(
            "normal_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Constants
    const N: usize = W64;

    let inv_sigma = 1.0 / std;
    let mean_v = Simd::<f64, N>::splat(mean);
    let inv_sigma_v = Simd::<f64, N>::splat(inv_sigma);
    let sqrt2_v = Simd::<f64, N>::splat(SQRT_2);

    let simd_body = |x_v| {
        let z = (x_v - mean_v) * inv_sigma_v / sqrt2_v;
        let erf_v = erf_simd::<N>(z);
        Simd::<f64, N>::splat(0.5) * (Simd::<f64, N>::splat(1.0) + erf_v)
    };

    // Fallback
    let scalar_body = |xi| {
        let z = (xi - mean) * inv_sigma / SQRT_2;
        0.5 * (1.0 + erf(z))
    };

    if !has_nulls(null_count, null_mask) {
        let (out, out_mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
            x,
            null_mask.is_some(),
            simd_body,
            scalar_body,
        );

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked kernel path
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    let (out, out_mask) =
        masked_univariate_kernel_f64_simd::<N, _, _>(x, mask, simd_body, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}
