// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Poisson Distribution SIMD Implementations** - *Vectorised Discrete Event Processing*
//!
//! High-performance SIMD-accelerated implementations of Poisson distribution PMF function.
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::Mask;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Simd, StdFloat, num::SimdUint};

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_u64_simd, masked_univariate_kernel_u64_simd,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std, masked_univariate_kernel_u64_std,
};
use crate::utils::{
    bitmask_to_simd_mask, has_nulls, simd_mask_to_bitmask, write_global_bitmask_block,
};
use crate::{errors::KernelError, utils::is_simd_aligned};

/// **Poisson Distribution Probability Mass Function** - *SIMD-Accelerated Discrete Event PMF*
/// 
/// Computes the probability mass function of the Poisson distribution using vectorised SIMD operations
/// where possible, with automatic scalar fallback for optimal performance in discrete counting processes
/// and event modelling applications.
/// 
/// ## Mathematical Definition
/// 
/// The Poisson distribution PMF is defined as:
/// 
/// ```text
/// P(X=k|λ) = e^(-λ) × λ^k / k!
/// ```
/// 
/// Where:
/// - `X = k` ∈ ℕ₀: observed event counts (input values, non-negative integers)
/// - `λ` ≥ 0: event rate parameter (average number of events per interval)
/// - `k!`: factorial of k
/// 
/// ## Parameters
/// 
/// * `k` - Input data slice of `u64` event counts where PMF is evaluated
/// * `lambda` - Event rate parameter (λ), must be non-negative and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
/// 
/// ## Returns
/// 
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PMF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
pub fn poisson_pmf_simd(
    k: &[u64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // Allow λ == 0 (degenerate); forbid negatives / non-finite
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "poisson_pmf: λ must be non-negative and finite".into(),
        ));
    }
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    const N: usize = W64;

    // Degenerate λ == 0 → PMF(k) = 1_{k==0}
    if lambda == 0.0 {
        let mut out = Vec64::with_capacity(k.len());
        if !has_nulls(null_count, null_mask) {
            if is_simd_aligned(k) {
                let zero_u = Simd::<u64, N>::splat(0);
                let one_f = Simd::<f64, N>::splat(1.0);
                let zero_f = Simd::<f64, N>::splat(0.0);
                let mut i = 0;
                while i + N <= k.len() {
                    let ku = Simd::<u64, N>::from_slice(&k[i..i + N]);
                    let is_zero = ku.simd_eq(zero_u);
                    let vals = is_zero.select(one_f, zero_f);
                    out.extend_from_slice(vals.as_array());
                    i += N;
                }
                for &ki in &k[i..] {
                    out.push(if ki == 0 { 1.0 } else { 0.0 });
                }
            } else {
                for &ki in k {
                    out.push(if ki == 0 { 1.0 } else { 0.0 });
                }
            }
            return Ok(FloatArray::from_vec64(out, None));
        }

        // masked path
        let mask = null_mask.expect("poisson_pmf: null_count > 0 requires null_mask");
        let mut out_mask = mask.clone();
        if is_simd_aligned(k) {
            let mask_bytes = mask.as_bytes();
            let zero_u = Simd::<u64, N>::splat(0);
            let one_f = Simd::<f64, N>::splat(1.0);
            let zero_f = Simd::<f64, N>::splat(0.0);
            let mut i = 0;
            while i + N <= k.len() {
                let lane_mask: Mask<i64, N> =
                    bitmask_to_simd_mask::<N, i64>(mask_bytes, i, k.len());
                let mut tmp = [0u64; N];
                for j in 0..N {
                    tmp[j] = if unsafe { lane_mask.test_unchecked(j) } {
                        k[i + j]
                    } else {
                        1
                    }; // any nonzero ok
                }
                let ku = Simd::<u64, N>::from_array(tmp);
                let is_zero = ku.simd_eq(zero_u);
                let vals = is_zero.select(one_f, zero_f);
                let res = lane_mask.select(vals, Simd::<f64, N>::splat(f64::NAN));
                out.extend_from_slice(res.as_array());

                let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
                write_global_bitmask_block(&mut out_mask, &bits, i, N);
                i += N;
            }
            for idx in i..k.len() {
                if !unsafe { mask.get_unchecked(idx) } {
                    out.push(f64::NAN);
                    unsafe { out_mask.set_unchecked(idx, false) }
                } else {
                    out.push(if k[idx] == 0 { 1.0 } else { 0.0 });
                    unsafe { out_mask.set_unchecked(idx, true) }
                }
            }
        } else {
            for idx in 0..k.len() {
                if !mask.get(idx) {
                    out.push(f64::NAN);
                    unsafe { out_mask.set_unchecked(idx, false) }
                } else {
                    out.push(if k[idx] == 0 { 1.0 } else { 0.0 });
                    unsafe { out_mask.set_unchecked(idx, true) }
                }
            }
        }
        return Ok(FloatArray {
            data: out.into(),
            null_mask: Some(out_mask),
        });
    }

    // Regular path (λ > 0)
    let log_lambda = lambda.ln();
    let neg_lambda = -lambda;

    let scalar_body = move |ki: u64| -> f64 {
        let kf = ki as f64;
        (neg_lambda + kf * log_lambda - ln_gamma_plus1(kf)).exp()
    };

    let simd_body = move |k_u: Simd<u64, N>| -> Simd<f64, N> {
        let kf = k_u.cast::<f64>();
        let mut arr = [0.0f64; N];
        for j in 0..N {
            arr[j] = ln_gamma_plus1(kf[j]);
        }
        let ln_kfact = Simd::<f64, N>::from_array(arr);
        let log_pmf = Simd::splat(neg_lambda) + kf * Simd::splat(log_lambda) - ln_kfact;
        log_pmf.exp()
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(k) {
            let has_mask = null_mask.is_some();
            let (data, mask) =
                dense_univariate_kernel_u64_simd::<N, _, _>(k, has_mask, simd_body, scalar_body);
            return Ok(FloatArray {
                data: data.into(),
                null_mask: mask,
            });
        }
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_u64_std(k, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    let mask_ref = null_mask.expect("poisson_pmf: null_count > 0 requires null_mask");
    if is_simd_aligned(k) {
        let (data, out_mask) =
            masked_univariate_kernel_u64_simd::<N, _, _>(k, mask_ref, simd_body, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: Some(out_mask),
        });
    }
    let (data, out_mask) = masked_univariate_kernel_u64_std(k, mask_ref, scalar_body);
    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}
