// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **SIMD Arithmetic Kernels Module** - *High-Performance Arithmetic*
//!
//! Inner SIMD-accelerated implementations using `std::simd` for maximum performance on modern hardware.
//! Prefer dispatch.rs for easily handling the general case, otherwise you can use these inner functions
//! directly (e.g., "dense_simd") vs. "maybe masked, maybe simd". 
//!
//! ## Overview
//! - **Portable SIMD**: Uses `std::simd` for cross-platform vectorisation with compile-time lane optimisation
//! - **Null masks**: Dense (no nulls) and masked variants for Arrow-compatible null handling. 
//!   These are uniified in dispatch.rs, and opting out of masking yields no performance penalty.
//! - **Type support**: Integer and floating-point arithmetic with specialised FMA operations
//! - **Safety**: All unsafe operations are bounds-checked or guaranteed by caller invariants
//!
//! ## Architecture Notes
//! - Building blocks for higher-level dispatch layers, or for low-level hot loops
//! where one wants to fully avoid abstraction overhead.
//! - Parallelisation intentionally excluded to allow flexible chunking strategies
//! - Power operations fall back to scalar for integers, use logarithmic computation for floats

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use core::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialEq;

use minarrow::Bitmask;
use num_traits::{One, PrimInt, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub, Zero};

use crate::kernels::bitmask::simd::all_true_mask_simd;
use crate::operators::ArithmeticOperator;
use crate::utils::simd_mask;

/// SIMD integer arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Panics on division/remainder by zero (consistent with scalar behaviour).
#[inline(always)]
pub fn int_dense_body_simd<T, const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) where
    T: Copy + One + PrimInt + ToPrimitive + Zero + SimdElement + WrappingMul,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Add<Output = Simd<T, LANES>>
        + Sub<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Rem<Output = Simd<T, LANES>>,
{
    let n = lhs.len();
    let mut vectorisable = n / LANES * LANES;
    let mut i = 0;
    while i < vectorisable {
        let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
        let r = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => a / b, // Panics if divisor is zero
            ArithmeticOperator::Remainder => a % b, // Panics if divisor is zero
            ArithmeticOperator::Power => {
                vectorisable = 0;
                break;
            }
        };
        r.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    // Scalar tail
    for idx in vectorisable..n {
        out[idx] = match op {
            ArithmeticOperator::Add => lhs[idx] + rhs[idx],
            ArithmeticOperator::Subtract => lhs[idx] - rhs[idx],
            ArithmeticOperator::Multiply => lhs[idx] * rhs[idx],
            ArithmeticOperator::Divide => lhs[idx] / rhs[idx], // Panics if divisor is zero
            ArithmeticOperator::Remainder => lhs[idx] % rhs[idx], // Panics if divisor is zero
            ArithmeticOperator::Power => {
                let mut acc = T::one();
                let exp = rhs[idx].to_u32().unwrap_or(0);
                for _ in 0..exp {
                    acc = acc.wrapping_mul(&lhs[idx]);
                }
                acc
            }
        };
    }
}

/// SIMD integer arithmetic kernel with null mask support.
/// Division/remainder by zero produces null results (mask=false) rather than panicking.
#[inline(always)]
pub fn int_masked_body_simd<T, const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy
        + PrimInt
        + ToPrimitive
        + Zero
        + One
        + SimdElement
        + PartialEq
        + WrappingAdd
        + WrappingMul
        + WrappingSub,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Add<Output = Simd<T, LANES>>
        + SimdPartialEq<Mask = Mask<<T as SimdElement>::Mask, LANES>>
        + Sub<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Rem<Output = Simd<T, LANES>>,
{
    let n = lhs.len();
    let dense = all_true_mask_simd(mask);

    /* If dense, we unfortunately need to near-replicate the dense implementation
    as that dedicated function panics on `div/0` as it needs to stay mask-free,
    to support varied workloads. This one works on the same dense principles,
    but substitutes the null mask when any div/0 issues occur. */

    if dense {
        // This block replaces the int_dense_body_simd call and handles masking for div/rem
        let vectorisable = n / LANES * LANES;
        let mut i = 0;
        while i < vectorisable {
            let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
            let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);

            let (r, valid): (Simd<T, LANES>, Mask<<T as SimdElement>::Mask, LANES>) = match op {
                ArithmeticOperator::Add => (a + b, Mask::splat(true)),
                ArithmeticOperator::Subtract => (a - b, Mask::splat(true)),
                ArithmeticOperator::Multiply => (a * b, Mask::splat(true)),
                ArithmeticOperator::Power => {
                    let mut tmp = [T::zero(); LANES];
                    for l in 0..LANES {
                        tmp[l] = a[l].pow(b[l].to_u32().unwrap_or(0));
                    }
                    (Simd::<T, LANES>::from_array(tmp), Mask::splat(true))
                }
                ArithmeticOperator::Divide | ArithmeticOperator::Remainder => {
                    let div_zero = b.simd_eq(Simd::splat(T::zero()));
                    let valid = !div_zero;
                    let safe_b = div_zero.select(Simd::splat(T::one()), b);
                    let r = match op {
                        ArithmeticOperator::Divide => a / safe_b,
                        ArithmeticOperator::Remainder => a % safe_b,
                        _ => unreachable!(),
                    };
                    let r = div_zero.select(Simd::splat(T::zero()), r);
                    (r, valid)
                }
            };
            r.copy_to_slice(&mut out[i..i + LANES]);
            // Write the out_mask based on the op
            let valid_bits = valid.to_bitmask();
            for l in 0..LANES {
                let idx = i + l;
                if idx < n {
                    unsafe {
                        out_mask.set_unchecked(idx, ((valid_bits >> l) & 1) == 1);
                    }
                }
            }
            i += LANES;
        }
        // Scalar tail
        for idx in vectorisable..n {
            match op {
                ArithmeticOperator::Add => {
                    out[idx] = lhs[idx].wrapping_add(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Subtract => {
                    out[idx] = lhs[idx].wrapping_sub(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Multiply => {
                    out[idx] = lhs[idx].wrapping_mul(&rhs[idx]);
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Power => {
                    out[idx] = lhs[idx].pow(rhs[idx].to_u32().unwrap_or(0));
                    unsafe {
                        out_mask.set_unchecked(idx, true);
                    }
                }
                ArithmeticOperator::Divide | ArithmeticOperator::Remainder => {
                    if rhs[idx] == T::zero() {
                        out[idx] = T::zero();
                        unsafe {
                            out_mask.set_unchecked(idx, false);
                        }
                    } else {
                        out[idx] = match op {
                            ArithmeticOperator::Divide => lhs[idx] / rhs[idx],
                            ArithmeticOperator::Remainder => lhs[idx] % rhs[idx],
                            _ => unreachable!(),
                        };
                        unsafe {
                            out_mask.set_unchecked(idx, true);
                        }
                    }
                }
            }
        }
        return;
    }

    let mut i = 0;
    while i + LANES <= n {
        let a = Simd::<T, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<T, LANES>::from_slice(&rhs[i..i + LANES]);
        let m_src: Mask<_, LANES> = simd_mask::<_, LANES>(mask, i, n); // validity mask

        // divisor-is-zero mask
        let div_zero: Mask<_, LANES> = b.simd_eq(Simd::splat(T::zero()));

        // ── compute result ───────────────────────────────────────────────────
        let res = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => {
                let safe_b = div_zero.select(Simd::splat(T::one()), b); // 0 → 1
                let q = a / safe_b;
                div_zero.select(Simd::splat(T::zero()), q) // restore 0
            }
            ArithmeticOperator::Remainder => {
                let safe_b = div_zero.select(Simd::splat(T::one()), b);
                let r = a % safe_b;
                div_zero.select(Simd::splat(T::zero()), r)
            }
            ArithmeticOperator::Power => {
                // scalar per-lane power
                let mut tmp = [T::zero(); LANES];
                for l in 0..LANES {
                    tmp[l] = a[l].pow(b[l].to_u32().unwrap_or(0));
                }
                Simd::<T, LANES>::from_array(tmp)
            }
        };

        // apply source validity mask, write results
        let selected = m_src.select(res, Simd::splat(T::zero()));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        // write out-mask bits: combine source mask with div-by-zero validity
        let final_mask = match op {
            ArithmeticOperator::Divide | ArithmeticOperator::Remainder => {
                // For div/rem: valid iff source is valid AND not dividing by zero
                m_src & !div_zero
            }
            _ => m_src,
        };
        let mbits = final_mask.to_bitmask();
        for l in 0..LANES {
            let idx = i + l;
            if idx < n {
                unsafe { out_mask.set_unchecked(idx, ((mbits >> l) & 1) == 1) };
            }
        }
        i += LANES;
    }

    // scalar tail
    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            let (result, final_valid) = match op {
                ArithmeticOperator::Add => (lhs[j].wrapping_add(&rhs[j]), true),
                ArithmeticOperator::Subtract => (lhs[j].wrapping_sub(&rhs[j]), true),
                ArithmeticOperator::Multiply => (lhs[j].wrapping_mul(&rhs[j]), true),
                ArithmeticOperator::Divide => {
                    if rhs[j] == T::zero() {
                        (T::zero(), false) // division by zero -> invalid
                    } else {
                        (lhs[j] / rhs[j], true)
                    }
                }
                ArithmeticOperator::Remainder => {
                    if rhs[j] == T::zero() {
                        (T::zero(), false) // remainder by zero -> invalid
                    } else {
                        (lhs[j] % rhs[j], true)
                    }
                }
                ArithmeticOperator::Power => (lhs[j].pow(rhs[j].to_u32().unwrap_or(0)), true),
            };
            out[j] = result;
            unsafe { out_mask.set_unchecked(j, final_valid) };
        } else {
            out[j] = T::zero();
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f32 arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no exceptions.
/// Power operations use scalar fallback with logarithmic computation.
#[inline(always)]
pub fn float_masked_body_f32_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f32],
    rhs: &[f32],
    mask: &Bitmask,
    out: &mut [f32],
    out_mask: &mut Bitmask,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    type M = <f32 as SimdElement>::Mask;

    let n = lhs.len();
    let mut i = 0;
    let dense = all_true_mask_simd(mask);

    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let m: Mask<M, LANES> = if dense {
            Mask::splat(true)
        } else {
            simd_mask::<M, LANES>(mask, i, n)
        };

        let res = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => a / b,
            ArithmeticOperator::Remainder => a % b,
            ArithmeticOperator::Power => (b * a.ln()).exp(),
        };

        let selected = m.select(res, Simd::<f32, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        let mbits = m.to_bitmask();
        for l in 0..LANES {
            let idx = i + l;
            if idx < n {
                unsafe { out_mask.set_unchecked(idx, ((mbits >> l) & 1) == 1) };
            }
        }
        i += LANES;
    }

    // Tail often caused by `n % LANES =! 0`; uses scalar fallback
    for j in i..n {
        let valid = dense || unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = match op {
                ArithmeticOperator::Add => lhs[j] + rhs[j],
                ArithmeticOperator::Subtract => lhs[j] - rhs[j],
                ArithmeticOperator::Multiply => lhs[j] * rhs[j],
                ArithmeticOperator::Divide => lhs[j] / rhs[j],
                ArithmeticOperator::Remainder => lhs[j] % rhs[j],
                ArithmeticOperator::Power => (rhs[j] * lhs[j].ln()).exp(),
            };
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f64 arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no exceptions.
/// Power operations use scalar fallback with logarithmic computation.
#[inline(always)]
pub fn float_masked_body_f64_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f64],
    rhs: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    type M = <f64 as SimdElement>::Mask;

    let n = lhs.len();
    let dense = all_true_mask_simd(mask);

    if dense {
        // hot
        float_dense_body_f64_simd::<LANES>(op, lhs, rhs, out);
        out_mask.fill(true);
        return;
    }

    let mut i = 0;
    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let m: Mask<M, LANES> = simd_mask::<M, LANES>(mask, i, n);

        let res = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => a / b,
            ArithmeticOperator::Remainder => a % b,
            ArithmeticOperator::Power => (b * a.ln()).exp(),
        };

        let selected = m.select(res, Simd::<f64, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        let mbits = m.to_bitmask();
        for l in 0..LANES {
            let idx = i + l;
            if idx < n {
                unsafe { out_mask.set_unchecked(idx, ((mbits >> l) & 1) == 1) };
            }
        }
        i += LANES;
    }

    // Tail often caused by `n % LANES =! 0`; uses scalar fallback
    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = match op {
                ArithmeticOperator::Add => lhs[j] + rhs[j],
                ArithmeticOperator::Subtract => lhs[j] - rhs[j],
                ArithmeticOperator::Multiply => lhs[j] * rhs[j],
                ArithmeticOperator::Divide => lhs[j] / rhs[j],
                ArithmeticOperator::Remainder => lhs[j] % rhs[j],
                ArithmeticOperator::Power => (rhs[j] * lhs[j].ln()).exp(),
            };
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f32 arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Division by zero produces Inf/NaN following IEEE 754 semantics.
#[inline(always)]
pub fn float_dense_body_f32_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = lhs.len();
    let mut i = 0;
    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let res = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => a / b,
            ArithmeticOperator::Remainder => a % b,
            ArithmeticOperator::Power => (b * a.ln()).exp(),
        };
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    // Tail often caused by `n % LANES =! 0`; uses scalar fallback
    for j in i..n {
        out[j] = match op {
            ArithmeticOperator::Add => lhs[j] + rhs[j],
            ArithmeticOperator::Subtract => lhs[j] - rhs[j],
            ArithmeticOperator::Multiply => lhs[j] * rhs[j],
            ArithmeticOperator::Divide => lhs[j] / rhs[j],
            ArithmeticOperator::Remainder => lhs[j] % rhs[j],
            ArithmeticOperator::Power => (rhs[j] * lhs[j].ln()).exp(),
        };
    }
}

/// SIMD f64 arithmetic kernel for dense arrays (no nulls).
/// Vectorised operations with scalar fallback for power operations and array tails.
/// Division by zero produces Inf/NaN following IEEE 754 semantics.
#[inline(always)]
pub fn float_dense_body_f64_simd<const LANES: usize>(
    op: ArithmeticOperator,
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [f64],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = lhs.len();
    let mut i = 0;
    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let res = match op {
            ArithmeticOperator::Add => a + b,
            ArithmeticOperator::Subtract => a - b,
            ArithmeticOperator::Multiply => a * b,
            ArithmeticOperator::Divide => a / b,
            ArithmeticOperator::Remainder => a % b,
            ArithmeticOperator::Power => (b * a.ln()).exp(),
        };
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    // Tail often caused by `n % LANES =! 0`; uses scalar fallback
    for j in i..n {
        out[j] = match op {
            ArithmeticOperator::Add => lhs[j] + rhs[j],
            ArithmeticOperator::Subtract => lhs[j] - rhs[j],
            ArithmeticOperator::Multiply => lhs[j] * rhs[j],
            ArithmeticOperator::Divide => lhs[j] / rhs[j],
            ArithmeticOperator::Remainder => lhs[j] % rhs[j],
            ArithmeticOperator::Power => (rhs[j] * lhs[j].ln()).exp(),
        };
    }
}

/// SIMD f32 fused multiply-add kernel with null mask support.
/// Hardware-accelerated `a.mul_add(b, c)` with proper null propagation.
#[inline(always)]
pub fn fma_masked_body_f32_simd<const LANES: usize>(
    lhs: &[f32],
    rhs: &[f32],
    acc: &[f32],
    mask: &Bitmask,
    out: &mut [f32],
    out_mask: &mut minarrow::Bitmask,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    use core::simd::{Mask, Simd};

    let n = lhs.len();
    let mut i = 0;
    let dense = all_true_mask_simd(mask);

    if dense {
        fma_dense_body_f32_simd::<LANES>(lhs, rhs, acc, out);
        out_mask.fill(true);
        return;
    }

    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f32, LANES>::from_slice(&acc[i..i + LANES]);
        let m: Mask<i32, LANES> = simd_mask::<i32, LANES>(mask, i, n);

        let res = a.mul_add(b, c);

        let selected = m.select(res, Simd::<f32, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        let mbits = m.to_bitmask();
        for l in 0..LANES {
            let idx = i + l;
            if idx < n {
                unsafe { out_mask.set_unchecked(idx, ((mbits >> l) & 1) == 1) };
            }
        }
        i += LANES;
    }

    // Scalar tail

    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = lhs[j].mul_add(rhs[j], acc[j]);
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f64 fused multiply-add kernel with null mask support.
/// Hardware-accelerated `a.mul_add(b, c)` with proper null propagation.
#[inline(always)]
pub fn fma_masked_body_f64_simd<const LANES: usize>(
    lhs: &[f64],
    rhs: &[f64],
    acc: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut minarrow::Bitmask,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    use core::simd::{Mask, Simd};

    let n = lhs.len();
    let mut i = 0;
    let dense = all_true_mask_simd(mask);

    if dense {
        // Hot
        fma_dense_body_f64_simd::<LANES>(lhs, rhs, acc, out);
        out_mask.fill(true);
        return;
    }

    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f64, LANES>::from_slice(&acc[i..i + LANES]);
        let m: Mask<i64, LANES> = simd_mask::<i64, LANES>(mask, i, n);

        let res = a.mul_add(b, c);

        let selected = m.select(res, Simd::<f64, LANES>::splat(0.0));
        selected.copy_to_slice(&mut out[i..i + LANES]);

        let mbits = m.to_bitmask();
        for l in 0..LANES {
            let idx = i + l;
            if idx < n {
                unsafe { out_mask.set_unchecked(idx, ((mbits >> l) & 1) == 1) };
            }
        }
        i += LANES;
    }

    // Scalar tail

    for j in i..n {
        let valid = unsafe { mask.get_unchecked(j) };
        if valid {
            out[j] = lhs[j].mul_add(rhs[j], acc[j]);
            unsafe { out_mask.set_unchecked(j, true) };
        } else {
            out[j] = 0.0;
            unsafe { out_mask.set_unchecked(j, false) };
        }
    }
}

/// SIMD f32 fused multiply-add kernel for dense arrays (no nulls).
/// Hardware-accelerated `a.mul_add(b, c)` with vectorised and scalar tail processing.
#[inline(always)]
pub fn fma_dense_body_f32_simd<const LANES: usize>(
    lhs: &[f32],
    rhs: &[f32],
    acc: &[f32],
    out: &mut [f32],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    use core::simd::Simd;

    let n = lhs.len();
    let mut i = 0;

    while i + LANES <= n {
        let a = Simd::<f32, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f32, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f32, LANES>::from_slice(&acc[i..i + LANES]);
        let res = a.mul_add(b, c);
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    for j in i..n {
        out[j] = lhs[j].mul_add(rhs[j], acc[j]);
    }
}

/// SIMD f64 fused multiply-add kernel for dense arrays (no nulls).
/// Hardware-accelerated `a.mul_add(b, c)` with vectorised and scalar tail processing.
#[inline(always)]
pub fn fma_dense_body_f64_simd<const LANES: usize>(
    lhs: &[f64],
    rhs: &[f64],
    acc: &[f64],
    out: &mut [f64],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    use core::simd::Simd;

    let n = lhs.len();
    let mut i = 0;

    while i + LANES <= n {
        let a = Simd::<f64, LANES>::from_slice(&lhs[i..i + LANES]);
        let b = Simd::<f64, LANES>::from_slice(&rhs[i..i + LANES]);
        let c = Simd::<f64, LANES>::from_slice(&acc[i..i + LANES]);
        let res = a.mul_add(b, c);
        res.copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }

    // Tail uses scalar fallback
    for j in i..n {
        out[j] = lhs[j].mul_add(rhs[j], acc[j]);
    }
}
