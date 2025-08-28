// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::errors::KernelError;
use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, masked_univariate_kernel_f64_std,
};
use crate::utils::has_nulls;

/// Chi-square PDF: f(x; k) = 1/(2^{k/2} Γ(k/2)) x^{k/2-1} e^{-x/2}
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn chi_square_pdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    // 1) parameter check
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_pdf: invalid df".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let k2 = 0.5 * df;
    let log_norm = -k2 * std::f64::consts::LN_2 - ln_gamma(k2);

    let scalar_body = move |xi: f64| {
        if xi < 0.0 {
            0.0
        } else if xi == 0.0 && k2 < 1.0 {
            f64::INFINITY // Special case for df < 2
        } else if xi == 0.0 && k2 == 1.0 {
            log_norm.exp() // Special case for df = 2, returns 0.5
        } else if xi == 0.0 {
            0.0 // General case for df > 2
        } else if xi.is_infinite() {
            0.0 // For infinity
        } else if xi.is_nan() {
            f64::NAN // Preserve NaN for null propagation
        } else {
            (log_norm + (k2 - 1.0) * xi.ln() - 0.5 * xi).exp()
        }
    };

    // 2) Dense (no nulls) path
    if !has_nulls(null_count, null_mask) {
        let has_mask = null_mask.is_some();
        let (data, mask) = dense_univariate_kernel_f64_std(x, has_mask, scalar_body);
        return Ok(FloatArray {
            data: data.into(),
            null_mask: mask,
        });
    }

    // 3) Null‐aware path
    let mask_ref = null_mask.expect("chi_square_pdf: null_count > 0 requires null_mask");

    // Scalar fallback
    let (data, out_mask) = masked_univariate_kernel_f64_std(x, mask_ref, scalar_body);

    Ok(FloatArray {
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

/// Chi-square CDF: F(x; k) = P(k/2, x/2)
#[inline(always)]
pub fn chi_square_cdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_cdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let len = x.len();
    let k2 = 0.5 * df;

    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(len);
        for &xi in x {
            let res = if xi < 0.0 {
                0.0
            } else if !xi.is_finite() {
                1.0 // CDF(∞) = 1.0
            } else {
                reg_lower_gamma(k2, 0.5 * xi)
            };
            out.push(res);
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path
    let mask = null_mask.expect("null path requires a mask");
    let mut out = Vec64::with_capacity(len);
    let mut out_mask = Bitmask::new_set_all(len, true);

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            let res = if xi < 0.0 {
                0.0
            } else if !xi.is_finite() {
                1.0 // CDF(∞) = 1.0
            } else {
                reg_lower_gamma(k2, 0.5 * xi)
            };
            out.push(res);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }

    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Chi-square quantile function (inverse CDF).
///
/// For each `p` in `p`, returns the value `x` such that `P(X ≤ x) = p` for
/// a chi-square distribution with `df` degrees of freedom.
/// Returns error for invalid input probabilities outside [0,1] or NaN.
pub fn chi_square_quantile_std(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_quantile: invalid df".into(),
        ));
    }
    if p.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }

    let a = 0.5 * df;
    let ln_gamma_ap1 = ln_gamma(a + 1.0);
    const SMALL_P: f64 = 1e-200;

    let solve_one = |prob: f64| -> f64 {
        if !(prob >= 0.0 && prob <= 1.0) || !prob.is_finite() {
            return f64::NAN;
        }
        if prob == 0.0 {
            return 0.0;
        }
        if prob == 1.0 {
            return f64::INFINITY;
        }
        // tiny p: asymptotic (already matched to SciPy)
        if prob < SMALL_P {
            let ln_x_half = (prob.ln() + ln_gamma_ap1) / a;
            return 2.0 * ln_x_half.exp();
        }

        // Special case for df=1: chi-squared(1) = N(0,1)^2
        // P(χ² ≤ x) = P(|Z| ≤ √x) = 2*Φ(√x) - 1 = p
        // So Φ(√x) = (1+p)/2, √x = Φ⁻¹((1+p)/2), x = [Φ⁻¹((1+p)/2)]²
        let mut x0 = if df == 1.0 && prob < 0.1 {
            let z = inv_std_normal(0.5 * (1.0 + prob));
            (z * z).max(1e-100)
        } else {
            // Wilson–Hilferty initial guess
            let nu = df;
            let z = inv_std_normal(prob);
            let c = 2.0 / (9.0 * nu);
            let base = 1.0 - (2.0 / (9.0 * nu)) + z * c.sqrt();
            (nu * base * base * base).max(0.0)
        };

        if !x0.is_finite() || x0 == 0.0 {
            // Fallback to previous inversion if seed is degenerate
            x0 = 2.0 * inv_reg_lower_gamma(a, prob).max(1e-100);
        }

        if prob > 0.999 {
            chi2_newton_refine_extreme(x0, a, prob)
        } else {
            chi2_newton_refine(x0, a, prob)
        }
    };

    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(p.len());
        for &prob in p {
            out.push(solve_one(prob));
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // masked
    let mask = null_mask.expect("null path requires a mask");
    let mut out = Vec64::with_capacity(p.len());
    for i in 0..p.len() {
        if !unsafe { mask.get_unchecked(i) } {
            out.push(f64::NAN);
        } else {
            out.push(solve_one(p[i]));
        }
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}
