// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **BLAS/LAPACK Integration Module** - *High-Performance Linear Algebra Kernels*
//!
//! ****************************************************************************************
//! ⚠️ Warning: This module has not been fully tested, and is not ready for production use. 
//! This warning applies to all multivariate kernels in *SIMD-kernels*, which are to be finalised
//! in an upcoming release.
//! ****************************************************************************************
//! 
//! This module provides low-level bindings and optimised kernels for linear algebra operations
//! through integration with industry-standard BLAS and LAPACK libraries. It forms the 
//! computational backbone for numerical linear algebra in the simd-kernels crate.
//!
//! ## Overview
//!
//! The module wraps BLAS and LAPACK industry-standard linear algebra kernels:
//!
//! ### Level 1 BLAS: Vector Operations
//! - Vector scaling, dot products, and norms
//! - Efficient memory access patterns with unit stride optimisation
//!
//! ### Level 2 BLAS: Matrix-Vector Operations  
//! - **GEMV**: General matrix-vector multiplication with transpose support
//! - Triangular system solvers for upper/lower matrices
//! - Symmetric and packed matrix operations
//!
//! ### Level 3 BLAS: Matrix-Matrix Operations
//! - **GEMM**: General matrix-matrix multiplication with blocking optimisation
//! - **SYRK**: Symmetric rank-k updates for covariance computation
//! - **SYR2K**: Symmetric rank-2k updates for advanced statistical operations
//!
//! ### LAPACK Decompositions and Solvers
//! - **LU Decomposition**: General linear systems with pivoting (`DGETRF`)
//! - **QR Decomposition**: Orthogonal factorisation for least squares (`DGEQRF`)
//! - **Cholesky Decomposition**: Symmetric positive definite systems (`DPOTRF`)
//! - **SVD**: Singular value decomposition for dimensionality reduction (`DGESVD`)
//! - **Eigensolvers**: Symmetric eigenvalue problems (`DSYEV`)
//!
//! ## External Dependencies
//!
//! This module requires linking against BLAS and LAPACK.
//! 
//! Supported implementations include:
//! - OpenBLAS - recommended for general uses
//! - Intel MKL - optimal for Intel processors  
//! - ATLAS, Accelerate, or system-provided BLAS/LAPACK
use blas::*;
use lapack::*;

/// GEMV  (y ← α·A·x + β·y)
#[inline(always)]
pub fn gemv(
    m: i32,
    n: i32,
    a: &[f64], // len = lda * n  (column-major)
    lda: i32,
    x: &[f64], // len = 1 + (len_x-1)*incx
    incx: i32,
    y: &mut [f64], // len = 1 + (len_y-1)*incy
    incy: i32,
    alpha: f64,
    beta: f64,
    trans_a: bool, // false = 'N', true = 'T'
) -> Result<(), &'static str> {
    // Dimension sanity
    if m <= 0 || n <= 0 {
        return Err("m and n must be positive");
    }
    if lda < m {
        return Err("lda must be ≥ m");
    }
    if incx == 0 || incy == 0 {
        return Err("incx and incy must be non-zero");
    }

    // Matrix buffer check
    if a.len() < (lda * n) as usize {
        return Err("A too small");
    }

    // Vector buffer checks
    let (len_x, len_y) = if trans_a { (m, n) } else { (n, m) };

    if x.len() < (1 + (len_x - 1) * incx.abs()) as usize {
        return Err("x too small");
    }
    if y.len() < (1 + (len_y - 1) * incy.abs()) as usize {
        return Err("y too small");
    }

    // BLAS call
    unsafe {
        dgemv(
            if trans_a { b'T' } else { b'N' },
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
    Ok(())
}

/// 4 × 4 micro-kernel (C += A·B)
#[inline(always)]
pub fn gemm_4x4_microkernel(
    a: &[f64; 16],     // 4×4 block of A (column-major)
    b: &[f64; 16],     // 4×4 block of B (column-major)
    c: &mut [f64; 16], // 4×4 block of C (column-major, updated in-place)
    alpha: f64,
    beta: f64,
) {
    // m = n = k = 4, lda = ldb = ldc = 4 (column-major, contiguous)
    unsafe {
        dgemm(
            b'N', // trans_a: not transposed
            b'N', // trans_b: not transposed
            4,    // m
            4,    // n
            4,    // k
            alpha, a, 4, b, 4, beta, c, 4,
        );
    }
}

/// 2×2 triangular solve (U · x = b  OR  L · x = b)
#[inline(always)]
pub fn trisolve_2x2(
    upper: bool,
    a: &mut [f64; 4], // 2×2 triangular matrix, overwritten by LAPACK
    b: &mut [f64; 2], // RHS – will contain the solution on return
) -> Result<(), &'static str> {
    unsafe {
        dtrsv(
            if upper { b'U' } else { b'L' }, // UPLO
            b'N',                            // trans = NoTrans
            b'N',                            // diag  = NonUnit
            2,                               // n
            a,                               // A
            2,                               // lda
            b,                               // x
            1,                               // incx
        );
    }
    Ok(())
}


/// Applies Householder reflector(s) to a matrix panel.
#[inline(always)]
pub fn householder_apply(
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n   (column-major)
    lda: i32,
    taus: &mut [f64], // len ≥ n
) -> Result<(), &'static str> {
    // recommended lwork ≥ n*64 for small panels
    let lwork = (n.max(1) * 64) as i32;
    let mut work = vec![0.0_f64; lwork as usize];
    let mut info = 0;

    unsafe {
        dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgeqrf failed")
    }
}

// Blocked GEMM wrapper (C ← αAB + βC)
#[inline(always)]
pub fn blocked_gemm(
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
    trans_a: bool,
    trans_b: bool,
) -> Result<(), &'static str> {
    // Buffer length checks
    let (_rows_a, cols_a) = if trans_a { (k, m) } else { (m, k) };
    let (_rows_b, cols_b) = if trans_b { (n, k) } else { (k, n) };

    if a.len() < (lda * cols_a) as usize {
        return Err("A too small");
    }
    if b.len() < (ldb * cols_b) as usize {
        return Err("B too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C too small");
    }

    unsafe {
        dgemm(
            if trans_a { b'T' } else { b'N' },
            if trans_b { b'T' } else { b'N' },
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
    Ok(())
}

/// CholeskyPanel  (lower-triangular, in-place)
pub fn cholesky_panel(
    n: i32,
    a: &mut [f64], // len ≥ lda*n ; on exit contains L in lower triangle
    lda: i32,
) -> Result<(), &'static str> {
    let mut info = 0;
    unsafe {
        dpotrf(b'L', n, a, lda, &mut info);
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix not positive-definite")
    } else {
        Err("LAPACK dpotrf argument error")
    }
}

/// Performs LU decomposition with partial pivoting on an m×n panel.
#[inline(always)]
pub fn lu_with_piv(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32], // ↓ length ≥ min(m,n)
) -> Result<(), &'static str> {
    use std::cmp::min;

    if a.len() < (lda * n) as usize {
        return Err("A too small for GETRF");
    }
    if ipiv.len() < min(m, n) as usize {
        return Err("ipiv too small for GETRF");
    }

    let mut info = 0;
    unsafe {
        dgetrf(m, n, a, lda, ipiv, &mut info);
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix is singular to machine precision")
    } else {
        Err("LAPACK dgetrf argument error")
    }
}

/// Blocked QR factorisation (full panel) using Householder reflections.
#[inline(always)]
pub fn qr_block(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    taus: &mut [f64],
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A too small for GEQRF");
    }
    if taus.len() < n as usize {
        return Err("taus too small for GEQRF");
    }

    // Workspace query: call with lwork = -1 to get optimal size
    let mut work_query = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgeqrf(
            m,
            n,
            a,
            lda,
            taus,
            &mut work_query,
            -1, // workspace query
            &mut info,
        );
    }
    if info != 0 {
        return Err("GEQRF workspace query failed");
    }

    // Allocate the optimal amount returned in work_query[0]
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgeqrf failed")
    }
}

// Composite Linear Algebra

/// SYRK_Panel   (C ← α A·Aᵀ + β C)
#[inline(always)]
pub fn syrk_panel(
    n: i32, // C is n×n
    k: i32, // inner-dim
    alpha: f64,
    a: &[f64], // len ≥ lda*k
    lda: i32,
    beta: f64,
    c: &mut [f64], // len ≥ ldc*n   (full-storage, col-major)
    ldc: i32,
    trans_a: bool, // false = use A    , true = use Aᵀ
) -> Result<(), &'static str> {
    use blas::dsyrk;

    if a.len() < (lda * k) as usize {
        return Err("A too small for SYRK");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C too small for SYRK");
    }

    unsafe {
        dsyrk(
            b'L', // UPLO: lower triangle
            if trans_a { b'T' } else { b'N' },
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }
    Ok(())
}

/// Computes eigenvalues/vectors of 2×2 symmetric matrix analytically via SIMD.
#[inline(always)]
pub fn symeig2x2(
    a_in: &[f64; 4], // input 2×2 symmetric (column-major)
    eigvals: &mut [f64; 2],
    eigvecs: &mut [f64; 4], // Q (column-major)
) -> Result<(), &'static str> {
    let mut a = *a_in; // make a mutable copy for LAPACK
    let mut info = 0;
    // workspace: lwork ≥ 3*n-1  when jobz='V'
    let mut work = [0.0_f64; 10];
    let len = work.len();
    unsafe {
        dsyev(
            b'V', // jobz = compute eigen-vectors
            b'U', // upper triangle supplied
            2,    // n
            &mut a, 2,       // a, lda
            eigvals, // w
            &mut work, len as i32, &mut info,
        );
    }
    if info != 0 {
        return Err("LAPACK dsyev failed on 2×2");
    }

    // Copy eigen-vectors (returned in A) for the caller
    eigvecs.copy_from_slice(&a);
    Ok(())
}

/// Performs Golub-Kahan bidiagonalisation (first SVD phase).
#[inline(always)]
pub fn bidiag_reduction(
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n
    lda: i32,
    d: &mut [f64],    // len ≥ min(m,n)
    e: &mut [f64],    // len ≥ min(m,n)-1
    tauq: &mut [f64], // len ≥ min(m,n)   (left  reflectors)
    taup: &mut [f64], // len ≥ min(m,n)   (right reflectors)
) -> Result<(), &'static str> {
    use std::cmp::min;

    use lapack::dgebrd;

    let k = min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for GEBRD");
    }
    if d.len() < k as usize {
        return Err("d too small for GEBRD");
    }
    if e.len() < (k - 1).max(0) as usize {
        return Err("e too small for GEBRD");
    }
    if tauq.len() < k as usize {
        return Err("tauq too small");
    }
    if taup.len() < k as usize {
        return Err("taup too small");
    }

    // Query optimal workspace
    let mut work_query = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgebrd(
            m,
            n,
            a,
            lda,
            d,
            e,
            tauq,
            taup,
            &mut work_query,
            -1, // lwork = -1 query
            &mut info,
        );
    }
    if info != 0 {
        return Err("GEBRD workspace query failed");
    }

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgebrd(m, n, a, lda, d, e, tauq, taup, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgebrd failed")
    }
}

/// Completes the SVD of a bidiagonal matrix produced by `bidiag_reduction`.
#[inline(always)]
pub fn svd_qr_iter() {
    todo!()
}


/// Computes the full or economy-size SVD of `A` in one shot.
#[inline(always)]
pub fn svd_block(
    jobu: u8,  // 'A', 'S', or 'N'
    jobvt: u8, // same for Vᵀ
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n   (destroyed)
    lda: i32,
    s: &mut [f64], // len ≥ min(m,n)
    u: &mut [f64], // len ≥ ldu*ucol   (ucol depends on jobu)
    ldu: i32,
    vt: &mut [f64], // len ≥ ldvt*ncol (ncol depends on jobvt)
    ldvt: i32,
) -> Result<(), &'static str> {
    use std::cmp::min;

    use lapack::dgesvd;

    let k = min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for DGESVD");
    }
    if s.len() < k as usize {
        return Err("s too small for DGESVD");
    }

    // Workspace query
    let mut wk = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgesvd(
            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &mut wk, -1, // query
            &mut info,
        );
    }
    if info != 0 {
        return Err("DGESVD workspace query failed");
    }

    let lwork = wk[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgesvd(
            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &mut work, lwork, &mut info,
        );
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("DGESVD failed to converge")
    } else {
        Err("DGESVD argument error")
    }
}

/// Projects data matrix X onto top-k principal components (PCA scores).
#[inline(always)]
pub fn pca_project(
    m: i32, // obs
    f: i32, // features (cols of X, rows of W)
    k: i32, // components
    alpha: f64,
    x: &[f64],
    ldx: i32, // m×f data matrix  (column-major)
    w: &[f64],
    ldw: i32, // f×k weight matrix
    beta: f64,
    y: &mut [f64],
    ldy: i32, // m×k output scores
) -> Result<(), &'static str> {
    use blas::dgemm;

    if x.len() < (ldx * f) as usize {
        return Err("X buffer too small");
    }
    if w.len() < (ldw * k) as usize {
        return Err("W buffer too small");
    }
    if y.len() < (ldy * k) as usize {
        return Err("Y buffer too small");
    }

    unsafe {
        // Y <- α·X·W + β·Y   (no transposes)
        dgemm(b'N', b'N', m, k, f, alpha, x, ldx, w, ldw, beta, y, ldy);
    }
    Ok(())
}

/// Computes streaming covariance matrix using SYRK on buffered tiles.
#[inline(always)]
pub fn cachecov_syrk(
    n_feat: i32,
    obs: i32,
    x: &[f64],
    ldx: i32,
    c: &mut [f64], // packed ‖C‖, len = n(n+1)/2
) -> Result<(), &'static str> {
    use blas::dsyrk;

    // Packed-triangle length check   n(n+1)/2
    let need = (n_feat as usize * (n_feat as usize + 1)) / 2;
    if c.len() < need {
        return Err("C buffer too small");
    }
    if x.len() < (ldx * obs) as usize {
        return Err("X tile too small");
    }

    // dsyrk wants full-matrix C, so we expand a scratch view & repack.
    // For simplicity (and because tiles are usually small) we allocate
    // a dense n×n scratch; copy back the lower triangle afterwards.
    let n = n_feat as usize;
    let mut full = vec![0.0_f64; n * n];

    // Unpack current packed ‖C‖ into dense
    for col in 0..n {
        for row in col..n {
            // lower
            full[row + col * n] = c[(row * (row + 1)) / 2 + col];
        }
    }

    // C ← Xᵀ·X   (β = 1 to accumulate)
    unsafe {
        dsyrk(
            b'L', b'T', n_feat, obs, 1.0, // α
            x, ldx, 1.0, // β  (accumulate)
            &mut full, n_feat,
        );
    }

    // Re-pack lower triangle back into c
    for col in 0..n {
        for row in col..n {
            c[(row * (row + 1)) / 2 + col] = full[row + col * n];
        }
    }
    Ok(())
}

/// LU decomposition with partial pivoting (PA = LU).
/// 
/// Computes `PA = LU` factorisation *in-place*.
#[inline(always)]
pub fn lufactor(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    piv: &mut [i32], // len ≥ min(m,n)
) -> Result<(), &'static str> {
    use lapack::dgetrf;

    let k = std::cmp::min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for GETRF");
    }
    if piv.len() < k as usize {
        return Err("pivot array too small");
    }

    let mut info = 0;
    unsafe {
        dgetrf(m, n, a, lda, piv, &mut info);
    }

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("U is singular")
    } else {
        Err("GETRF argument error")
    }
}

/// Solves **U X = B** where **U** is upper-triangular.
#[inline(always)]
pub fn trisolve_upper(
    n: i32,
    nrhs: i32,
    u: &[f64],
    ldu: i32, // n×n upper-triangular
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if u.len() < (ldu * n) as usize {
        return Err("U buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    for j in 0..nrhs {
        let col = &mut b[(j * ldb) as usize..][..n as usize];
        unsafe {
            dtrsv(
                b'U', // UPLO = upper
                b'N', // trans = NoTrans
                b'N', // DIAG = Non-unit
                n, u, ldu, col, 1,
            );
        }
    }
    Ok(())
}

/// Solves **L · X = B** for X, where **L** is *lower-triangular*.
#[inline(always)]
pub fn trisolve_lower(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32, // n × n, lower-triangular
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    for j in 0..nrhs {
        let col = &mut b[(j * ldb) as usize..][..n as usize];
        unsafe {
            dtrsv(
                b'L', // UPLO  = lower
                b'N', // trans = NoTrans
                b'N', // DIAG  = non-unit
                n, l, ldl, col, 1,
            );
        }
    }
    Ok(())
}

/// Computes inverse of a triangular matrix (A⁻¹) via block solve.
#[inline(always)]
pub fn tri_inverse(n: i32, t: &mut [f64], ldt: i32, upper: bool) -> Result<(), &'static str> {
    if t.len() < (ldt * n) as usize {
        return Err("T buffer too small");
    }

    let mut info = 0;
    unsafe {
        dtrtri(
            if upper { b'U' } else { b'L' },
            b'N', // non-unit diagonal
            n,
            t,
            ldt,
            &mut info,
        );
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("T is singular")
    } else {
        Err("DTRTRI argument error")
    }
}

// Performs Cholesky decomposition: A = LLᵀ for SPD matrices.
#[inline(always)]
pub fn spd_cholesky(n: i32, a: &mut [f64], lda: i32) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }

    let mut info = 0;
    unsafe { dpotrf(b'L', n, a, lda, &mut info) };

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix is not SPD")
    } else {
        Err("DPOTRF argument error")
    }
}


/// Solves **Σ X = B** where **Σ = L Lᵀ** (factor must already exist).
#[inline(always)]
pub fn spd_solve(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32, // lower-triangular factor
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    let mut info = 0;
    unsafe {
        dpotrs(b'L', n, nrhs, l, ldl, b, ldb, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("DPOTRS failed")
    }
}


/// Computes inverse of SPD matrix via Cholesky factorisation.
#[inline(always)]
pub fn spd_inverse(n: i32, l: &mut [f64], ldl: i32) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }

    let mut info = 0;
    unsafe { dpotri(b'L', n, l, ldl, &mut info) };

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix not SPD (dpotri)")
    } else {
        Err("DPOTRI argument error")
    }
}

/// Panel-wise QR decomposition using Householder reflectors.
#[inline(always)]
pub fn qr_panel(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,         // m × n panel, overwritten with R + Householder vectors
    taus: &mut [f64], // len ≥ n, receives τ scalars
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if taus.len() < n as usize {
        return Err("TAU buffer too small");
    }

    // Workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dgeqrf(m, n, a, lda, taus, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DGEQRF work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DGEQRF factorisation failed")
    }
}

/// Forms orthonormal matrix Q from QR factorisation.
#[inline(always)]
pub fn qr_form_q(
    m: i32,
    n: i32,
    k: i32, // number of elementary reflectors (τ.len() ≥ k)
    a: &mut [f64],
    lda: i32,     // on entry: from `qr_panel`; on exit: Q
    taus: &[f64], // τ from `qr_panel`
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if taus.len() < k as usize {
        return Err("TAU buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dorgqr(m, n, k, a, lda, taus, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DORGQR work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dorgqr(m, n, k, a, lda, taus, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DORGQR failed")
    }
}

/// Solves least-squares problem Ax = b using QR decomposition.
#[inline(always)]
pub fn least_squares_qr(
    m: i32,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32, // A (overwritten)
    b: &mut [f64],
    ldb: i32, // B → on exit contains solution X (min-norm)
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgels(
            b'N',
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            &mut work_q,
            lwork,
            &mut info,
            1, // workspace buffer length (for the query, just 1)
        )
    };
    if info != 0 {
        return Err("DGELS work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];
    let len = work.len();
    unsafe {
        dgels(
            b'N', m, n, nrhs, a, lda, b, ldb, &mut work, lwork, &mut info, len,
        )
    };
    if info == 0 {
        Ok(())
    } else {
        Err("DGELS failed")
    }
}


/// Full symmetric eigendecomposition: A = QΛQᵀ.
#[inline(always)]
pub fn symeig_full(
    n: i32,
    a: &mut [f64],
    lda: i32,      // on entry upper-triangle of A; on exit Q (col-major)
    w: &mut [f64], // eigen-values λ (len ≥ n)
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if w.len() < n as usize {
        return Err("W buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dsyev(b'V', b'U', n, a, lda, w, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DSYEV work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dsyev(b'V', b'U', n, a, lda, w, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DSYEV failed")
    }
}

/// Builds Fisher information matrix or XᵀWX using symmetric rank-k update.
#[inline(always)]
pub fn syrk_fisher_info(
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32, // Xᵀ or W½X depending on context
    beta: f64,
    c: &mut [f64],
    ldc: i32, // symmetric C (lower-packed or full col-major lower)
) -> Result<(), &'static str> {
    if a.len() < (lda * k) as usize {
        return Err("A buffer too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C buffer too small");
    }

    unsafe {
        dsyrk(
            b'L', // use lower triangle
            b'N', // C ← α A Aᵀ + β C
            n, k, alpha, a, lda, beta, c, ldc,
        );
    }
    Ok(())
}

/// Performs symmetric rank-2k update: C += AᵀB + BᵀA.
#[inline(always)]
pub fn sym_rank2k_update(
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) -> Result<(), &'static str> {
    if a.len() < (lda * k) as usize {
        return Err("A buffer too small");
    }
    if b.len() < (ldb * k) as usize {
        return Err("B buffer too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C buffer too small");
    }

    unsafe {
        dsyr2k(
            b'L', // lower triangle
            b'T', // C += α (Aᵀ B + Bᵀ A)
            n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    Ok(())
}
