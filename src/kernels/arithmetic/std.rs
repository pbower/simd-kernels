// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Standard Arithmetic Kernels Module** - *Scalar Fallback / Non-SIMD Implementations*
//!
//! Portable scalar implementations of arithmetic operations for compatibility and unaligned data.
//!
//! Prefer dispatch.rs for easily handling the general case, otherwise you can use these inner functions
//! directly (e.g., "dense_std") vs. "maybe masked, maybe std". 
//! 
//! ## Overview
//! - **Scalar loops**: Standard element-wise operations without vectorisation
//! - **Fallback role**: Used when SIMD alignment requirements aren't met or SIMD is disabled
//! - **Full compatibility**: Works on any architecture regardless of SIMD support
//! - **Null-aware**: Supports Arrow-compatible null mask propagation
//!
//! ## Design Notes
//! - Intentionally avoids parallelisation to allow higher-level chunking strategies
//! - Wrapping arithmetic for integers to prevent overflow panics
//! - Division by zero handling: panics for integers, produces Inf/NaN for floats

use crate::operators::ArithmeticOperator;
use minarrow::Bitmask;
use num_traits::{Float, PrimInt, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub};

/// Scalar integer arithmetic kernel for dense arrays (no nulls).
/// Performs element-wise operations using wrapping arithmetic to prevent overflow panics.
/// Panics on division/remainder by zero.
#[inline(always)]
pub fn int_dense_body_std<T: PrimInt + ToPrimitive + WrappingAdd + WrappingSub + WrappingMul>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) {
    let n = lhs.len();
    for i in 0..n {
        out[i] = match op {
            ArithmeticOperator::Add => lhs[i].wrapping_add(&rhs[i]),
            ArithmeticOperator::Subtract => lhs[i].wrapping_sub(&rhs[i]),
            ArithmeticOperator::Multiply => lhs[i].wrapping_mul(&rhs[i]),
            ArithmeticOperator::Divide => {
                if rhs[i] == T::zero() {
                    panic!("Division by zero")
                } else {
                    lhs[i] / rhs[i]
                }
            }
            ArithmeticOperator::Remainder => {
                if rhs[i] == T::zero() {
                    panic!("Remainder by zero")
                } else {
                    lhs[i] % rhs[i]
                }
            }
            ArithmeticOperator::Power => lhs[i].pow(rhs[i].to_u32().unwrap_or(0)),
        };
    }
}

/// Scalar integer arithmetic kernel with null mask support.
/// Handles division by zero gracefully by marking results as null instead of panicking.
/// Invalid inputs (mask=false) and zero division produce null outputs.
#[inline(always)]
pub fn int_masked_body_std<T: PrimInt + ToPrimitive + WrappingAdd + WrappingSub + WrappingMul>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = lhs.len();
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            let (result, final_valid) = match op {
                ArithmeticOperator::Add => (lhs[i].wrapping_add(&rhs[i]), true),
                ArithmeticOperator::Subtract => (lhs[i].wrapping_sub(&rhs[i]), true),
                ArithmeticOperator::Multiply => (lhs[i].wrapping_mul(&rhs[i]), true),
                ArithmeticOperator::Divide => {
                    if rhs[i] == T::zero() {
                        (T::zero(), false) // division by zero -> invalid
                    } else {
                        (lhs[i] / rhs[i], true)
                    }
                }
                ArithmeticOperator::Remainder => {
                    if rhs[i] == T::zero() {
                        (T::zero(), false) // remainder by zero -> invalid
                    } else {
                        (lhs[i] % rhs[i], true)
                    }
                }
                ArithmeticOperator::Power => (lhs[i].pow(rhs[i].to_u32().unwrap_or(0)), true),
            };
            out[i] = result;
            unsafe {
                out_mask.set_unchecked(i, final_valid);
            }
        } else {
            out[i] = T::zero();
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }
    }
}

/// Scalar floating-point arithmetic kernel for dense arrays (no nulls).
/// Division by zero produces Inf/NaN rather than panicking.
/// Power operations use logarithmic exponentiation: `exp(b * ln(a))`.
#[inline(always)]
pub fn float_dense_body_std<T: Float>(op: ArithmeticOperator, lhs: &[T], rhs: &[T], out: &mut [T]) {
    let n = lhs.len();
    for i in 0..n {
        out[i] = match op {
            ArithmeticOperator::Add => lhs[i] + rhs[i],
            ArithmeticOperator::Subtract => lhs[i] - rhs[i],
            ArithmeticOperator::Multiply => lhs[i] * rhs[i],
            ArithmeticOperator::Divide => lhs[i] / rhs[i],
            ArithmeticOperator::Remainder => lhs[i] % rhs[i],
            ArithmeticOperator::Power => (rhs[i] * lhs[i].ln()).exp(),
        };
    }
}

/// Scalar floating-point arithmetic kernel with null mask support.
/// Preserves IEEE 754 semantics: division by zero produces Inf/NaN, no panicking.
/// Invalid inputs (mask=false) produce null outputs with zero values.
#[inline(always)]
pub fn float_masked_body_std<T: Float>(
    op: ArithmeticOperator,
    lhs: &[T],
    rhs: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = lhs.len();
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            out[i] = match op {
                ArithmeticOperator::Add => lhs[i] + rhs[i],
                ArithmeticOperator::Subtract => lhs[i] - rhs[i],
                ArithmeticOperator::Multiply => lhs[i] * rhs[i],
                ArithmeticOperator::Divide => lhs[i] / rhs[i],
                ArithmeticOperator::Remainder => lhs[i] % rhs[i],
                ArithmeticOperator::Power => (rhs[i] * lhs[i].ln()).exp(),
            };
            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            out[i] = T::zero();
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }
    }
}

/// Fused multiply add (a * b + acc) with null mask
#[inline(always)]
pub fn fma_masked_body_std<T: Float>(
    lhs: &[T],
    rhs: &[T],
    acc: &[T],
    mask: &Bitmask,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = lhs.len();
    for i in 0..n {
        let valid = unsafe { mask.get_unchecked(i) };
        if valid {
            out[i] = lhs[i].mul_add(rhs[i], acc[i]);
            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            out[i] = T::zero();
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }
    }
}

/// Dense fused multiply add (a * b + acc)
#[inline(always)]
pub fn fma_dense_body_std<T: Float>(lhs: &[T], rhs: &[T], acc: &[T], out: &mut [T]) {
    let n = lhs.len();
    for i in 0..n {
        out[i] = lhs[i].mul_add(rhs[i], acc[i]);
    }
}
