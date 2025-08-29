// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Binomial Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of binomial distribution functions
//! utilising vectorised operations for bulk computations on 64-byte aligned data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! that process multiple binomial distribution evaluations simultaneously using
//! CPU vector instructions. The implementations automatically fall back to scalar
//! versions when data alignment requirements are not met.

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};
use std::simd::{Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use crate::kernels::scientific::distributions::shared::scalar::{ln_gamma_simd, *};
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd, masked_univariate_kernel_f64_simd,
};
use crate::utils::has_nulls;

/// SIMD-accelerated implementation of binomial distribution probability mass function.
///
/// Processes multiple PMF evaluations simultaneously using vectorised logarithmic
/// computations for improved performance on large datasets with 64-byte memory alignment.
#[inline(always)]
pub fn binomial_pmf_simd(
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

    const N: usize = W64;
    let ln_choose_n = ln_gamma((n as f64) + 1.0);

    let ln_choose_n_v = Simd::<f64, N>::splat(ln_choose_n);
    let ln_p_v = Simd::<f64, N>::splat(p.max(f64::MIN_POSITIVE).ln());
    let ln_q_v = Simd::<f64, N>::splat((1.0 - p).max(f64::MIN_POSITIVE).ln());
    let n_f = Simd::<f64, N>::splat(n as f64);
    let n_i = Simd::<u64, N>::splat(n);

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

    let simd_body = |k_v: Simd<f64, N>| {
        let k_u = k_v.cast::<u64>();
        let valid = k_u.simd_le(n_i);

        let ln_g_k1 = ln_gamma_simd(k_v + Simd::splat(1.0));
        let ln_g_nk1 = ln_gamma_simd(n_f - k_v + Simd::splat(1.0));

        let ln_pmf = ln_choose_n_v - ln_g_k1 - ln_g_nk1 + k_v * ln_p_v + (n_f - k_v) * ln_q_v;
        let pmf = ln_pmf.exp();

        valid.select(pmf, Simd::splat(0.0))
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

        let (out, out_mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
            &k_f64,
            null_mask.is_some(),
            simd_body,
            scalar_body,
        );

        return Ok(FloatArray {
            data: out.into(),
            null_mask: out_mask,
        });
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    let (out, out_mask) =
        masked_univariate_kernel_f64_simd::<N, _, _>(&k_f64, mask, simd_body, scalar_body);

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Partially SIMD-accelerated implementation of binomial distribution cumulative distribution function.
/// There is little benefit here except in null mask handling.
#[inline(always)]
pub fn binomial_cdf_simd(
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

    const N: usize = W64;
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

    let simd_body = |k_v: Simd<f64, N>| {
        let k_u = k_v.cast::<u64>();
        let n_simd = Simd::<u64, N>::splat(n);
        let one_simd = Simd::<f64, N>::splat(1.0);
        let zero_simd = Simd::<f64, N>::splat(0.0);

        // Handle simple edge cases with SIMD
        let ki_ge_n = k_u.simd_ge(n_simd);

        // Edge case results
        let mut result = one_simd; // Default for ki >= n and p == 0

        if p == 1.0 {
            // p=1: return 1.0 if ki >= n, else 0.0
            result = ki_ge_n.select(one_simd, zero_simd);
        } else if p != 0.0 {
            // Need incomplete_beta for ki < n
            let mut out_arr = [0.0f64; N];
            let k_arr = k_v.to_array();
            for i in 0..N {
                let ki = k_arr[i] as u64;
                out_arr[i] = if ki >= n {
                    1.0
                } else {
                    1.0 - incomplete_beta((ki + 1) as f64, n_f - ki as f64, p)
                };
            }
            return Simd::<f64, N>::from_array(out_arr);
        }

        result
    };

    // Use SIMD for edge cases, scalar fallback for incomplete_beta
    if !has_nulls(null_count, null_mask) {
        let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();
        let (data, mask) = dense_univariate_kernel_f64_simd::<N, _, _>(
            &k_f64,
            null_mask.is_some(),
            simd_body,
            scalar_body,
        );
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();
    let (data, out_mask) =
        masked_univariate_kernel_f64_simd::<N, _, _>(&k_f64, mask, simd_body, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
