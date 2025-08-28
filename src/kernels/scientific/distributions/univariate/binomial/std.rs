// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Binomial Distribution Scalar Implementation**
//!
//! Scalar implementations of binomial distribution functions using
//! traditional mathematical algorithms optimised for accuracy and numerical stability.
//!
//! ## Overview
//! This module provides the scalar (non-SIMD) reference implementations for binomial
//! distribution calculations. These implementations serve as both fallback implementations 
//! when SIMD is unavailable.

use minarrow::{Bitmask, FloatArray, vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::{
    errors::KernelError,
    kernels::scientific::distributions::univariate::common::std::{
        dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
    },
    utils::has_nulls,
};

/// Scalar implementation of binomial distribution probability mass function.
///
/// Computes the PMF using logarithmic computation for numerical stability:
/// log(PMF) = log_gamma(n+1) - log_gamma(k+1) - log_gamma(n-k+1) + k×log(p) + (n-k)×log(1-p)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_pmf_std(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if !(0.0 <= p && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "binomial_pmf: invalid p".into(),
        ));
    }
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let ln_choose_n = ln_gamma((n as f64) + 1.0);

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki <= n {
            (ln_choose_n - ln_gamma((ki as f64) + 1.0) - ln_gamma(((n - ki) as f64) + 1.0)
                + (ki as f64) * p.ln()
                + ((n - ki) as f64) * (1.0 - p).ln())
            .exp()
        } else {
            0.0
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

        let (out, out_mask) =
            dense_univariate_kernel_f64_std(&k_f64, null_mask.is_some(), scalar_body);

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    let (out, out_mask) = masked_univariate_kernel_f64_std(&k_f64, mask, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Scalar implementation of binomial distribution cumulative distribution function.
///
/// Computes the CDF using the relationship with the regularised incomplete beta function:
/// P(X ≤ k) = I₁₋ₚ(n-k, k+1) where I is the regularised incomplete beta function.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_cdf_std(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if !(p >= 0.0 && p <= 1.0) || n > 1_000_000_000 {
        return Err(KernelError::InvalidArguments(
            "binomial_cdf: invalid p or n".into(),
        ));
    }
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let n_f = n as f64;

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki >= n {
            1.0
        } else if p == 0.0 {
            1.0
        } else if p == 1.0 {
            if ki >= n { 1.0 } else { 0.0 }
        } else {
            1.0 - incomplete_beta((ki + 1) as f64, n_f - ki as f64, p)
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

        let (out, out_mask) =
            dense_univariate_kernel_f64_std(&k_f64, null_mask.is_some(), scalar_body);

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked path using masked_univariate_kernel
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    let (out, out_mask) = masked_univariate_kernel_f64_std(&k_f64, mask, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Scalar implementation of binomial distribution quantile function (inverse CDF).
///
/// Computes the quantile function using a dual-strategy approach optimised for
/// different parameter ranges to balance accuracy and computational efficiency.
#[inline(always)]
pub fn binomial_quantile_std(
    p: &[f64],
    n: u64,
    p_: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Arg checks
    if !(0.0 < p_ && p_ < 1.0) || !p_.is_finite() {
        return Err(KernelError::InvalidArguments(
            "binomial_quantile: invalid p_".into(),
        ));
    }
    if n == 0 {
        return Ok(FloatArray::from_vec64(vec64![0.0; p.len()], None));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // Small-n pre-computed CDF
    let small_n = n <= 100;
    let small_cdf: Option<Vec<f64>> = if small_n {
        // pre-compute cumulative probabilities C(0..n)
        let mut cdf = vec![0.0; (n + 1) as usize];
        let mut prob = (1.0 - p_).powf(n as f64);
        cdf[0] = prob;
        for k in 1..=n as usize {
            prob *= ((n - k as u64 + 1) as f64) * p_ / (k as f64 * (1.0 - p_));
            cdf[k] = cdf[k - 1] + prob;
        }
        Some(cdf)
    } else {
        None
    };

    // Scalar evaluator
    let scalar_body = |pi: f64| -> f64 {
        if !(pi >= 0.0 && pi <= 1.0) || !pi.is_finite() {
            return f64::NAN;
        }
        if pi == 0.0 {
            return -1.0;
        }
        if pi == 1.0 {
            return n as f64;
        }
        if small_n {
            let cdf = small_cdf.as_ref().unwrap();
            let mut k = 0_usize;
            while k < cdf.len() && cdf[k] < pi {
                k += 1;
            }
            k as f64
        } else {
            binomial_quantile_cornish_fisher(pi, n, p_)
        }
    };

    if !has_nulls(null_count, null_mask) {
        // dense path (no nulls)
        let (out, out_mask) = dense_univariate_kernel_f64_std(p, null_mask.is_some(), scalar_body);
        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let (out, out_mask) = masked_univariate_kernel_f64_std(p, mask, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}
