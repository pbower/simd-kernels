// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Student's t-Distribution SIMD Implementations** - *Vectorised Heavy-Tailed Statistical Inference*
//!
//! High-performance SIMD-accelerated implementations of Student's t-distribution pdf function.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{Simd, StdFloat};

use minarrow::{Bitmask, FloatArray};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::{has_nulls, is_simd_aligned};

/// **Student's t-Distribution Probability Density Function** - *SIMD-Accelerated Heavy-Tailed PDF*
/// 
/// Computes the probability density function of Student's t-distribution using vectorised SIMD operations
/// where possible, with automatic scalar fallback for compatibility across all architectures.
/// 
/// ## Parameters
/// 
/// * `x` - Input data slice of `f64` values where PDF is evaluated
/// * `df` - Degrees of freedom (ν), must be positive and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
/// 
/// ## Returns
/// 
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PDF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
#[inline(always)]
pub fn student_t_pdf_simd(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // argument checks
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_pdf: invalid df".into(),
        ));
    }
    // precompute the normalizing constant
    let coeff =
        (ln_gamma((df + 1.0) * 0.5) - ln_gamma(df * 0.5) - 0.5 * (df * std::f64::consts::PI).ln())
            .exp();
    // empty‐input fast path
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;
    // scalar fallback per‐element
    let scalar_body =
        move |xi: f64| -> f64 { coeff * (1.0 + xi * xi / df).powf(-(df + 1.0) * 0.5) };

    // SIMD body: works on N‐lane Simd<f64>

    let simd_body = move |x_v: Simd<f64, N>| -> Simd<f64, N> {
        let df_v = Simd::<f64, N>::splat(df);
        let coeff_v = Simd::<f64, N>::splat(coeff);
        let exp_v = Simd::<f64, N>::splat(-(df + 1.0) * 0.5);
        let one_v = Simd::<f64, N>::splat(1.0);
        let t = one_v + (x_v * x_v) / df_v;
        // coeff * exp(ln(t) * exp_v)  == coeff * t.powf(exp_v)
        coeff_v * (t.ln() * exp_v).exp()
    };

    // Dense fast path
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

    // Null‐aware masked path
    let mask_ref = null_mask.expect("student_t_pdf: null_count > 0 requires null_mask");

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
