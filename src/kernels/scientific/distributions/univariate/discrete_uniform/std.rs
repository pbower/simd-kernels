// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
use crate::utils::has_nulls;

/// Discrete uniform PMF (SciPy randint semantics: support = {low, …, high-1}).
/// P(X=k) = 1/(high−low) for k ∈ [low, high), else 0.
#[inline(always)]
pub fn discrete_uniform_pmf_std(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_pmf: require low < high (upper exclusive)".into(),
        ));
    }

    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_pmf: range overflow".into())
    })?; // span = N = high - low  (N >= 1)
    let p = 1.0 / (span as f64);

    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(k.len());
        for &ki in k {
            out.push(if (low..high).contains(&ki) { p } else { 0.0 });
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("discrete_uniform_pmf: null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(k.len());
    for i in 0..k.len() {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            let ki = k[i];
            out.push(if (low..high).contains(&ki) { p } else { 0.0 });
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

/// Discrete uniform CDF (lower tail, inclusive).
/// F(k) = 0 for k < low; F(k) = 1 for k ≥ high-1; else F(k) = (k − low + 1)/(high − low).
#[inline(always)]
pub fn discrete_uniform_cdf_std(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_cdf: require low < high (upper exclusive)".into(),
        ));
    }

    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_cdf: range overflow".into())
    })?; // N = high - low
    let inv_n = 1.0 / (span as f64);
    if k.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(k.len());
        for &ki in k {
            let v = if ki < low {
                0.0
            } else if ki >= high - 1 {
                1.0
            } else {
                ((ki - low + 1) as f64) * inv_n
            };
            out.push(v);
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("discrete_uniform_cdf: null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(k.len());
    for i in 0..k.len() {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            let ki = k[i];
            let v = if ki < low {
                0.0
            } else if ki >= high - 1 {
                1.0
            } else {
                ((ki - low + 1) as f64) * inv_n
            };
            out.push(v);
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

/// Discrete uniform quantile (lower tail) matching SciPy randint.ppf.
/// Support = {low,…,high-1}, N = high−low.
/// Q(p) = low + clamp(ceil(p*N)−1, −1, N−1).
/// p is clamped to [0,1]; NaN → NaN (valid).
#[inline(always)]
pub fn discrete_uniform_quantile_std(
    p: &[f64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_quantile: require low < high (upper exclusive)".into(),
        ));
    }

    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_quantile: range overflow".into())
    })?; // N = high - low ≥ 1
    let n_f = span as f64;
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(p.len());
        for &pi in p {
            if pi.is_nan() {
                out.push(f64::NAN);
                continue;
            }
            let pc = pi.clamp(0.0, 1.0);
            let t = (pc * n_f).ceil(); // t ∈ [0, N]
            let mut idx = (t as i64).saturating_sub(1); // ∈ [-1, N-1]
            let n_minus1 = (span as i64).saturating_sub(1);
            if idx < -1 {
                idx = -1;
            }
            if idx > n_minus1 {
                idx = n_minus1;
            }
            let k = (low as i64).saturating_add(idx); // low-1 … high-1
            out.push(k as f64);
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    let mask = null_mask.expect("discrete_uniform_quantile: null_count > 0 requires null_mask");
    let mut out = Vec64::with_capacity(p.len());
    for i in 0..p.len() {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            let pi = p[i];
            if pi.is_nan() {
                out.push(f64::NAN);
                continue;
            }
            let pc = pi.clamp(0.0, 1.0);
            let t = (pc * n_f).ceil();
            let mut idx = (t as i64).saturating_sub(1);
            let n_minus1 = (span as i64).saturating_sub(1);
            if idx < -1 {
                idx = -1;
            }
            if idx > n_minus1 {
                idx = n_minus1;
            }
            let k = (low as i64).saturating_add(idx);
            out.push(k as f64);
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
