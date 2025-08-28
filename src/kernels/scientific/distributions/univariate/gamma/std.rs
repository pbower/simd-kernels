// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! Standard (scalar) implementation of gamma distribution functions.
//!
//! This module provides scalar implementations of the gamma distribution's probability
//! density function (PDF), cumulative distribution function (CDF), and quantile function
//! (inverse CDF). These implementations serve as the fallback when SIMD is not available
//! or for data that is not properly aligned for vectorised computation.
//!
//! ## Implementation Notes
//!
//! ### Rate Parameterisation
//! All functions use the rate parameterisation internally where β = 1/θ (rate = 1/scale).
//!
//! ### Numerical Stability
//! - **PDF**: Uses log-space arithmetic to prevent overflow for large shape parameters
//! - **CDF**: Employs continued fraction expansion of the incomplete gamma function
//! - **Quantile**: Uses region-specific inversion algorithms for optimal accuracy
//!
//! ### Special Case Handling
//! - Zero and negative inputs are handled explicitly
//! - Boundary cases (p=0, p=1) return appropriate limit values
//! - Non-finite inputs and invalid parameters produce NaN or errors as appropriate

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::scalar::*;

#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn gamma_pdf_std(
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

    let beta = scale;
    let ln_gamma_k = ln_gamma(shape);
    let log_norm = shape * beta.ln() - ln_gamma_k;

    let scalar_body = |xi: f64| -> f64 {
        if !xi.is_finite() {
            return f64::NAN;
        }
        if xi < 0.0 {
            return 0.0;
        }
        if xi == 0.0 {
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

    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    let mask_ref = null_mask.expect("gamma_pdf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Gamma CDF: F(x) = γ(k, x/θ) / Γ(k)
/// Gamma CDF with rate β: F(x) = P(k, β x) = γ(k, β x)/Γ(k)
#[inline(always)]
pub fn gamma_cdf_std(
    x: &[f64],
    shape: f64,
    scale: f64, // interpreted as rate β
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if shape <= 0.0 || !shape.is_finite() || scale <= 0.0 || !scale.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_cdf: invalid shape or rate".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }
    let beta = scale;

    let len = x.len();
    let mut out = Vec64::with_capacity(len);

    let eval = |xi: f64| -> f64 {
        if !xi.is_finite() {
            return f64::NAN;
        }
        if xi <= 0.0 {
            return 0.0;
        }
        reg_lower_gamma(shape, beta * xi)
    };

    if !has_nulls(null_count, null_mask) {
        for &xi in x {
            out.push(eval(xi));
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("Null mask must be present if null_count > 0");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            out.push(eval(x[i]));
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

/// Gamma quantile (inverse CDF): x such that CDF(x) = p
/// Gamma quantile (inverse CDF): x such that CDF(x) = p
///
/// IMPORTANT: `scale` is the **rate** β (>0), for consistency with `gamma_pdf`/`gamma_cdf`.
#[inline(always)]
pub fn gamma_quantile_std(
    p: &[f64],
    shape: f64, // k > 0
    scale: f64, // rate β > 0  (NOT θ)
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if shape <= 0.0 || !shape.is_finite() || scale <= 0.0 || !scale.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_quantile: invalid shape or rate".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let beta = scale;

    // Use upper-tail inversion when (1-p) is tiny (large-x regime).
    // Central & left regions go through the lower-tail inverter.
    const RIGHT_TAIL_SWITCH: f64 = 1e-10;

    let mut out = Vec64::with_capacity(p.len());
    let eval = |prob: f64| -> f64 {
        if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
            return f64::NAN;
        }
        if prob == 0.0 {
            return 0.0;
        }
        if prob == 1.0 {
            return f64::INFINITY;
        }

        let q = 1.0 - prob;
        let z = if q <= RIGHT_TAIL_SWITCH {
            // robust for extreme right tail
            inv_reg_upper_gamma(shape, q)
        } else {
            // central / left
            inv_reg_lower_gamma(shape, prob)
        };
        z / beta
    };

    if !has_nulls(null_count, null_mask) {
        for &pi in p {
            out.push(eval(pi));
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("Null mask must be present if null_count > 0");
    for i in 0..p.len() {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            out.push(eval(p[i]));
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
