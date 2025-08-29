// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Negative Binomial Scalar Implementations** - *CPU-Optimised Discrete Statistics*
//!
//! Scalar (non-SIMD) implementations of negative binomial distribution functions optimised for
//! numerical stability and computational efficiency. These implementations serve as the foundation
//! for SIMD acceleration and provide reliable fallback behaviour.

use std::f64;

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std, masked_univariate_kernel_u64_std,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray, Vec64};

/// Negative Binomial PMF (Pascal distribution, number of failures before r-th success)
/// PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k, for k=0,1,...
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn neg_binomial_pmf_std(
    k: &[u64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) Parameter checks
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_pmf: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    // 2) Empty input
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let log_p = p.ln();
    let log1mp = (1.0 - p).ln();
    let r_f64 = r as f64;

    let scalar_body = move |ki: u64| {
        let lf = ln_choose(ki + r - 1, ki);
        (lf + r_f64 * log_p + (ki as f64) * log1mp).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        // if a mask was passed (with null_count==0), we still want to return an all-true mask
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_u64_std(k, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null‐aware masked path
    let mask_ref = null_mask.expect("neg_binomial_pmf: null_count > 0 requires null_mask");
    let (data, out_mask) = masked_univariate_kernel_u64_std(k, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Negative Binomial CDF (sum of PMFs to k): F(X ≤ k) = I_p(r, k+1)
#[inline(always)]
pub fn neg_binomial_cdf_std(
    k: &[u64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_cdf: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    let r_f64 = r as f64;
    let len = k.len();

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &ki in k {
            out.push(incomplete_beta(r_f64, (ki as f64) + 1.0, p));
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path
    let mut out = Vec64::with_capacity(len);
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            out.push(f64::NAN);
        } else {
            let ki = k[idx];
            let cdf = incomplete_beta(r_f64, (ki as f64) + 1.0, p);
            out.push(cdf);
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

/// Negative Binomial Quantile (inverse CDF): returns minimal k such that CDF(k) >= q
#[inline(always)]
pub fn neg_binomial_quantile_std(
    q: &[f64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_quantile: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    let r_f64 = r as f64;
    let len = q.len();

    // Quantile per probability entry
    let compute_quantile = |qi: f64| -> f64 {
        if qi > 0.0 && qi < 1.0 {
            let x = incomplete_beta_inv(r_f64, 1.0, qi);
            let mut k = x.floor().max(0.0) as u64;
            while incomplete_beta(r_f64, (k as f64) + 1.0, p) < qi {
                k += 1;
            }
            k as f64

        // 0 becomes -1, 1 becomes infinite
        // outside of these is NaN
        } else if qi == 0.0 {
            -1.0
        } else if qi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // Dense path
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &qi in q {
            out.push(compute_quantile(qi));
        }
        // either `None` for null mask, or "all true" i.e., null count: 0
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path
    let mut out = Vec64::with_capacity(len);
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            // push sentinel value
            out.push(f64::NAN);
            continue;
        }
        let qi = q[idx];
        out.push(compute_quantile(qi));
    }
    // propagate input null mask
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
