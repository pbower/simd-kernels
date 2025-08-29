// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Universal Scalar Function Module** - *High-Performance Element-Wise Mathematical Operations*
//!
//! Full suite of vectorised mathematical functions that operate
//! element-wise on arrays of floating-point values. It serves as the computational backbone for
//! mathematical operations across the simd-kernels crate, for both scalar and SIMD-accelerated
//! implementations with opt-in Arrow-compatible null masking.
//!
//! These are the semantic equivalent of *numpy ufuncs* in Python.
//!
//! ## Overview
//!
//! Universal scalar functions are fundamental building blocks for:
//! - **Data Preprocessing**: Normalisations, transformations, and scaling operations
//! - **Scientific Computing**: Mathematical transformations and special function evaluation
//! - **Machine Learning**: Activation functions, feature engineering, and data preparation
//! - **Signal Processing**: Filtering, transforms, and spectral analysis
//! - **Statistics**: Data transformations and statistical preprocessing
//! - **Financial Mathematics**: Risk calculations and price transformations

use crate::kernels::scientific::erf::erf as erf_fn;
use crate::utils::{bitmask_to_simd_mask, simd_mask_to_bitmask, write_global_bitmask_block};
use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};
use std::simd::{LaneCount, SupportedLaneCount};

/// Generates a mapping kernel that returns a FloatArray<f64>,
/// propagating any input nulls (and never touching lanes that were null).
///
/// `$name`  – function name to create  
/// `$expr`  – expression mapping a scalar `f64 -> f64`  
#[macro_export]
macro_rules! impl_vecmap {
    ($name:ident, $expr:expr) => {
        #[inline(always)]
        pub fn $name<const LANES: usize>(
            input: &[f64],
            null_mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Result<FloatArray<f64>, &'static str>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            let len = input.len();
            // fast length‐0 case
            if len == 0 {
                return Ok(FloatArray::from_slice(&[]));
            }
            // decide if we need the null‐aware path
            let has_nulls = match null_count {
                Some(n) => n > 0,
                None => null_mask.is_some(),
            };
            // dense (no nulls) path
            if !has_nulls {
                let mut out = Vec64::with_capacity(len);
                #[cfg(feature = "simd")]
                {
                    // Check if input array is properly aligned for SIMD (cheap runtime check)
                    if is_simd_aligned(input) {
                        use core::simd::Simd;
                        let mut i = 0;
                        while i + LANES <= len {
                            let v = Simd::<f64, LANES>::from_slice(&input[i..i + LANES]);
                            let mut r = Simd::<f64, LANES>::splat(0.0);
                            for lane in 0..LANES {
                                r[lane] = $expr(v[lane]);
                            }
                            out.extend_from_slice(r.as_array());
                            i += LANES;
                        }
                        // scalar tail
                        for &x in &input[i..] {
                            out.push($expr(x));
                        }
                        return Ok(FloatArray::from_vec64(out, None));
                    }
                    // Fall through to scalar path if alignment check failed
                }
                // Scalar fallback - alignment check failed
                #[cfg(not(feature = "simd"))]
                {
                    for &x in input {
                        out.push($expr(x));
                    }
                }
                #[cfg(feature = "simd")]
                {
                    for &x in input {
                        out.push($expr(x));
                    }
                }
                return Ok(FloatArray::from_vec64(out, None));
            }
            // null‐aware path
            let mb = null_mask.expect(concat!(
                stringify!($name),
                ": input mask required when nulls present"
            ));
            let mut out = Vec64::with_capacity(len);
            let mut out_mask = Bitmask::new_set_all(len, true);

            #[cfg(feature = "simd")]
            {
                // Check if input array is properly aligned for SIMD (cheap runtime check)
                if is_simd_aligned(input) {
                    use core::simd::{Mask, Simd};
                    let mask_bytes = mb.as_bytes();
                    let mut i = 0;
                    while i + LANES <= len {
                        // pull in the Arrow validity into a SIMD mask
                        let lane_valid: Mask<i8, LANES> =
                            bitmask_to_simd_mask::<LANES, i8>(mask_bytes, i, len);

                        // Gather inputs (nulls -> NaN)
                        let mut arr = [0.0f64; LANES];
                        for j in 0..LANES {
                            let idx = i + j;
                            arr[j] = if unsafe { lane_valid.test_unchecked(j) } {
                                input[idx]
                            } else {
                                f64::NAN
                            };
                        }
                        let v = Simd::<f64, LANES>::from_array(arr);

                        // Apply your scalar expr in SIMD form
                        let mut r = Simd::<f64, LANES>::splat(0.0);
                        for lane in 0..LANES {
                            r[lane] = $expr(v[lane]);
                        }
                        let r_arr = r.as_array();
                        out.extend_from_slice(r_arr);

                        // write those same validity bits back into our new null‐bitmap
                        let block = simd_mask_to_bitmask::<LANES, i8>(lane_valid, LANES);
                        write_global_bitmask_block(&mut out_mask, &block, i, LANES);

                        i += LANES;
                    }
                    // scalar tail
                    for idx in i..len {
                        if !unsafe { mb.get_unchecked(idx) } {
                            out.push(f64::NAN);
                            unsafe { out_mask.set_unchecked(idx, false) };
                        } else {
                            let y = $expr(input[idx]);
                            out.push(y);
                            unsafe { out_mask.set_unchecked(idx, true) };
                        }
                    }

                    // if every lane stayed valid, drop the mask
                    let null_bitmap = if out_mask.all_set() {
                        None
                    } else {
                        Some(out_mask)
                    };
                    return Ok(FloatArray {
                        data: out.into(),
                        null_mask: null_bitmap,
                    });
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..len {
                    if !unsafe { mb.get_unchecked(idx) } {
                        out.push(f64::NAN);
                        unsafe { out_mask.set_unchecked(idx, false) };
                    } else {
                        let y = $expr(input[idx]);
                        out.push(y);
                        unsafe { out_mask.set_unchecked(idx, true) };
                    }
                }
            }
            #[cfg(feature = "simd")]
            {
                for idx in 0..len {
                    if !unsafe { mb.get_unchecked(idx) } {
                        out.push(f64::NAN);
                        unsafe { out_mask.set_unchecked(idx, false) };
                    } else {
                        let y = $expr(input[idx]);
                        out.push(y);
                        unsafe { out_mask.set_unchecked(idx, true) };
                    }
                }
            }

            // if every lane stayed valid, drop the mask
            let null_bitmap = if out_mask.all_set() {
                None
            } else {
                Some(out_mask)
            };
            Ok(FloatArray {
                data: out.into(),
                null_mask: null_bitmap,
            })
        }
    };
}

impl_vecmap!(abs, |x: f64| x.abs());
impl_vecmap!(neg, |x: f64| -x);
impl_vecmap!(recip, |x: f64| 1.0 / x);
impl_vecmap!(sqrt, |x: f64| x.sqrt());
impl_vecmap!(cbrt, |x: f64| x.cbrt());

impl_vecmap!(exp, |x: f64| x.exp());
impl_vecmap!(exp2, |x: f64| x.exp2());
impl_vecmap!(ln, |x: f64| x.ln());
impl_vecmap!(log2, |x: f64| x.log2());
impl_vecmap!(log10, |x: f64| x.log10());

impl_vecmap!(sin, |x: f64| x.sin());
impl_vecmap!(cos, |x: f64| x.cos());
impl_vecmap!(tan, |x: f64| x.tan());
impl_vecmap!(asin, |x: f64| x.asin());
impl_vecmap!(acos, |x: f64| x.acos());
impl_vecmap!(atan, |x: f64| x.atan());

impl_vecmap!(sinh, |x: f64| x.sinh());
impl_vecmap!(cosh, |x: f64| x.cosh());
impl_vecmap!(tanh, |x: f64| x.tanh());
impl_vecmap!(asinh, |x: f64| x.asinh());
impl_vecmap!(acosh, |x: f64| x.acosh());
impl_vecmap!(atanh, |x: f64| x.atanh());

impl_vecmap!(erf, |x: f64| erf_fn(x));
impl_vecmap!(erfc, |x: f64| erf_fn(x));

impl_vecmap!(ceil, |x: f64| x.ceil());
impl_vecmap!(floor, |x: f64| x.floor());
impl_vecmap!(trunc, |x: f64| x.trunc());
impl_vecmap!(round, |x: f64| x.round());
impl_vecmap!(sign, |x: f64| x.signum());

impl_vecmap!(sigmoid, |x: f64| 1.0 / (1.0 + (-x).exp()));
impl_vecmap!(softplus, |x: f64| (1.0 + x.exp()).ln());
impl_vecmap!(relu, |x: f64| if x > 0.0 { x } else { 0.0 });
impl_vecmap!(gelu, |x: f64| 0.5
    * x
    * (1.0 + erf_fn(x / std::f64::consts::SQRT_2)));
