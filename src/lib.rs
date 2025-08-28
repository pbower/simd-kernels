// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under the Mozilla Public License (MPL) 2.0. 
// See LICENSE for details.

// At the time of writing this unlocks extra std::simd that the developers
// intend on stabilising but haven't yet.
// This includes custom lane management abstractions, and related features.
#![feature(portable_simd)]
#![feature(float_erf)]

// compile with RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features portable_simd

pub mod operators;

pub mod kernels {
    pub mod aggregate;
    pub mod arithmetic;
    pub mod binary;
    pub mod bitmask;
    pub mod comparison;
    pub mod conditional;
    pub mod logical;
    pub mod sort;
    pub mod string;
    pub mod unary;
    pub mod window;
    pub mod scientific {
        #[cfg(feature = "linear_algebra")]
        pub mod blas_lapack;
        #[cfg(feature = "probability_distributions")]
        pub mod distributions;
        #[cfg(feature = "probability_distributions")]
        pub mod erf;
        #[cfg(feature = "fourier_transforms")]
        pub mod fft;
        #[cfg(feature = "linear_algebra")]
        pub mod matrix;
        #[cfg(feature = "universal_functions")]
        pub mod scalar;
        #[cfg(feature = "linear_algebra")]
        pub mod vector;
    }
}

pub mod traits {
    pub mod to_bits;
}

pub mod config;
pub mod errors;

pub mod utils;
