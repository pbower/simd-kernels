// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::erf::erf;
use crate::utils::has_nulls;

#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};

/// Lognormal PDF: f(x|μ,σ) = 1/(xσ√2π) * exp(-½[(ln(x)-μ)/σ]^2)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_pdf_std(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let denom = sdlog * SQRT_2PI;

    let scalar_body = |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog;
            (-0.5 * z * z).exp() / (denom * xi)
        } else {
            0.0
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

    // Null-aware path
    let mask_ref = null_mask.expect("lognormal_pdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Lognormal CDF: F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_cdf_std(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Constants
    let sqrt2 = SQRT_2;

    // scalar fallback
    let scalar_body = move |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog / sqrt2;
            0.5 * (1.0 + erf(z))
        } else {
            0.0
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null‐aware path
    let mask_ref = null_mask.expect("lognormal_cdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Lognormal quantile: Q(p|μ,σ) = exp(μ + σ * Φ⁻¹(p)), p in (0,1)
#[inline(always)]
pub fn lognormal_quantile_std(
    p: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_quantile: invalid parameters".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Dense fast path: no nulls
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &pi in p {
            let nq = normal_quantile_scalar(pi, 0.0, 1.0);
            out.push((meanlog + sdlog * nq).exp());
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Propagate input mask nulls as is; retain any new `NaN` or `inf`
    // values verbatim
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(len);

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            continue;
        }
        let pi = p[idx];
        let nq = normal_quantile_scalar(pi, 0.0, 1.0);
        let q = (meanlog + sdlog * nq).exp();
        out.push(q);
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
