// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under the Mozilla Public License (MPL) 2.0.
// See LICENSE for details.

// At the time of writing this unlocks extra std::simd that the developers
// intend on stabilising but haven't yet.
// This includes custom lane management abstractions, and related features.
#![feature(portable_simd)]
#![feature(float_erf)]

// Link OpenBLAS when linear_algebra feature is enabled.
// This forces the linker to include the OpenBLAS symbols.
#[cfg(feature = "linear_algebra")]
extern crate openblas_src;

// compile with RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features portable_simd

pub mod operators;

// The bitmask, arithmetic and string kernels are contained in the upstream `Minarrow` crate,
// and are available in the namespace.

pub mod kernels {
    pub mod aggregate;
    pub mod binary;
    pub mod comparison;
    pub mod conditional;
    pub mod logical;
    pub mod sort;
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
    pub mod dense_iter;
    pub mod to_bits;
}

pub mod config;

pub mod utils;
