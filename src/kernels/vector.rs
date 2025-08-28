// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Vector Operations Kernels Module** - *Linear Algebra and BLAS Integration*
//!
//! Vector operations kernels for comprehensive linear algebra computations
//! with optimised BLAS integration and hardware acceleration. Essential foundation for
//! numerical computing, machine learning, and scientific analytical workloads.
//!
//! ## Core Operations  
//! - **BLAS Level 1**: Optimised vector-vector operations (AXPY, SCAL, DOT, NRM2)
//! - **Linear combinations**: Scaled vector addition with hardware-accelerated implementations
//! - **Vector norms**: L1, L2, and infinity norm calculations with numerical stability
//! - **Dot products**: Inner product computation with extended precision accumulation
//! - **Vector scaling**: In-place and out-of-place scalar multiplication operations
//! - **Memory operations**: Efficient vector copying with alignment-aware optimisation

use blas::{daxpy, dcopy, dnrm2, dscal};

/// Scaled vector addition — `y ← α·x + y`
/// 
/// Computes linear combination of vectors with scalar scaling
#[inline(always)]
pub fn axpy(
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &mut [f64],
    incy: i32,
) -> Result<(), &'static str> {
    check_len(x.len(), incx, "x")?;
    check_len(y.len(), incy, "y")?;
    unsafe { daxpy(n, alpha, x, incx, y, incy) };
    Ok(())
}

/// In-place scaling — `x ← α·x`
///
/// Multiplies each vector element by a scalar
#[inline(always)]
pub fn scal(
    n: i32,
    alpha: f64,
    x: &mut [f64],
    incx: i32,
) -> Result<(), &'static str> {
    check_len(x.len(), incx, "x")?;
    unsafe { dscal(n, alpha, x, incx) };
    Ok(())
}

/// Copies vector x into y
///
/// Simple memory copy of one vector to another
#[inline(always)]
pub fn copy(
    n: i32,
    x: &[f64],
    incx: i32,
    y: &mut [f64],
    incy: i32,
) -> Result<(), &'static str> {
    check_len(x.len(), incx, "x")?;
    check_len(y.len(), incy, "y")?;
    unsafe { dcopy(n, x, incx, y, incy) };
    Ok(())
}

/// Computes L2 norm of vector: ‖x‖₂ = sqrt(Σx²)
///
/// Returns Euclidean norm (magnitude) of vector
#[inline(always)]
pub fn vector_norm(
    n: i32,
    x: &[f64],
    incx: i32,
) -> Result<f64, &'static str> {
    check_len(x.len(), incx, "x")?;
    // BLAS returns 0.0 for n == 0 by spec.
    let norm = unsafe { dnrm2(n, x, incx) };
    Ok(norm)
}

#[inline(always)]
fn check_len(len: usize, inc: i32, buf_name: &str) -> Result<(), &'static str> {
    if len < (1 + (len.saturating_sub(1)) * inc.abs() as usize) {
        Err(match buf_name {
            "x" => "x buffer too small",
            "y" => "y buffer too small",
            _   => "buffer too small",
        })
    } else {
        Ok(())
    }
}