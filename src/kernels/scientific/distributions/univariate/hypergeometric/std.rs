// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std, masked_univariate_kernel_u64_std,
};
use crate::utils::has_nulls;

/// Hypergeometric PMF: P(X = k)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn hypergeometric_pmf_std(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // parameter checks
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_pmf: invalid parameters".into(),
        ));
    }
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    // common pre-computations
    let ln_denom = ln_choose(population, draws);
    let min_k = success.min(draws);

    // scalar body (u64 → f64)
    let scalar_body = |ki: u64| -> f64 {
        if ki <= min_k && draws >= ki && draws - ki <= population - success {
            (ln_choose(success, ki) + ln_choose(population - success, draws - ki) - ln_denom).exp()
        } else {
            0.0
        }
    };

    // choose dense vs. masked
    if !has_nulls(null_count, null_mask) {
        // dense path
        let (data, mask) = dense_univariate_kernel_u64_std(k, null_mask.is_some(), scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // masked path
    let in_mask = null_mask.expect("hypergeometric_pmf: null_count > 0 requires null_mask");
    let (data, mask_out) = masked_univariate_kernel_u64_std(k, in_mask, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(mask_out),
    })
}

/// Hypergeometric CDF: F(k) = ∑_{i=0}^k PMF(i)
#[inline(always)]
pub fn hypergeometric_cdf_std(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_cdf: invalid parameters".into(),
        ));
    }
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }
    let min_k = draws.min(success);
    let len = k.len();

    if !has_nulls(null_count, null_mask) {
        let ln_denom = ln_choose(population, draws);
        let mut out = Vec64::with_capacity(len);
        for &ki in k {
            let kmax = ki.min(min_k);
            let mut sum = 0.0;
            for s in 0..=kmax {
                if draws >= s && draws - s <= population - success {
                    sum += (ln_choose(success, s) + ln_choose(population - success, draws - s)
                        - ln_denom)
                        .exp();
                }
            }
            out.push(sum);
        }

        // Clone `None` for null mask, or "all true" where null_count was `0`
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let ln_denom = ln_choose(population, draws);
    let mut out = Vec64::with_capacity(len);

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            // Sentinel
            // We could leave trash in there for better performance in future
            // given it's masked anyway.
            out.push(f64::NAN);
            continue;
        }
        let ki = k[idx];
        let kmax = ki.min(min_k);
        let mut sum = 0.0;
        for s in 0..=kmax {
            if draws >= s && draws - s <= population - success {
                sum += (ln_choose(success, s) + ln_choose(population - success, draws - s)
                    - ln_denom)
                    .exp();
            }
        }
        out.push(sum);
    }
    // we propagate null mask
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

/// Hypergeometric quantile: Q(p) = smallest k such that CDF(k) ≥ p
#[inline(always)]
pub fn hypergeometric_quantile_std(
    p: &[f64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_quantile: invalid parameters".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let n = draws;
    let kmax = draws.min(success);
    let len = p.len();
    let ln_denom = ln_choose(population, draws);

    let compute_quantile = |pi: f64| -> f64 {
        // early exit
        // 0.0 returns -1, 1.0 returns kmax, and outside of that is NaN
        if !pi.is_finite() || pi < 0.0 || pi > 1.0 + 1e-14 {
            return f64::NAN;
        }
        if pi == 0.0 {
            return -1.0;
        }
        if pi >= 1.0 - 1e-14 {
            // Handle values very close to 1.0
            return kmax as f64;
        }

        // main logic
        let mut cdf = 0.0;
        let mut qk = kmax; // default to kmax if no k satisfies condition
        for k in 0..=kmax {
            if n >= k && n - k <= population - success {
                let pmf = (ln_choose(success, k) + ln_choose(population - success, n - k)
                    - ln_denom)
                    .exp();
                cdf += pmf;
                if cdf >= pi {
                    qk = k;
                    break;
                }
            }
        }
        qk as f64
    };

    let mut out = Vec64::with_capacity(len);

    if !has_nulls(null_count, null_mask) {
        for &pi in p {
            out.push(compute_quantile(pi));
        }
        // Clone `None` for null mask, or "all true" where null_count was `0`
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            // Push sentinel
            out.push(f64::NAN);
        } else {
            out.push(compute_quantile(p[idx]));
        }
    }

    // Propagate input mask
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
