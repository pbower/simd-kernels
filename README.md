# SIMD-Kernels

**High-performance compute kernels for Rust, built on `std::simd`.**

SIMD-Kernels gives you vectorised arithmetic, statistics, scientific functions, and sorting that operate directly on typed slices. Built on Rust's [`std::simd`](https://doc.rust-lang.org/std/simd/) portable SIMD, so you get hardware-accelerated kernels on every platform without writing a single intrinsic. Null-aware by default, feature-gated for minimal footprint, and ready for real-time or HPC workloads.

## Why SIMD-Kernels?

**The problem:** Writing SIMD by hand means maintaining separate code paths for SSE2, AVX2, AVX-512, and NEON. Most libraries either avoid SIMD entirely, rely on auto-vectorisation hints that may or may not fire, or use trait-object dispatch that defeats the optimiser at the call boundary.

**The solution:** SIMD-Kernels is built directly on [`std::simd`](https://doc.rust-lang.org/std/simd/), the portable SIMD module in Rust's standard library. `std::simd` is a major investment area for the Rust project — it provides a single, portable abstraction over hardware vector instructions that the compiler lowers to the best available ISA on each target. You write one kernel, and it runs vectorised on x86, ARM, and WASM without platform-specific intrinsics or conditional compilation.

On top of that foundation, every kernel here operates on concrete typed slices like `&[f64]` or `&[i32]`, with explicit null-mask support following Apache Arrow semantics. No `dyn`, no `Any`, no runtime downcasting.

## Quick Start

```rust
use simd_kernels::kernels::arithmetic::add_f64_dense;

let a = &[1.0, 2.0, 3.0];
let b = &[10.0, 20.0, 30.0];

let result = add_f64_dense(a, b);
// [11.0, 22.0, 33.0]
```

All arithmetic kernels use SIMD internally and support both dense and null-masked variants.

## What's Included

### Core Kernels

| Module | Description |
|--------|-------------|
| `aggregate` | Sum, mean, variance, min/max, count distinct |
| `binary` | Bitwise operations |
| `comparison` | SIMD mask comparisons across all numeric types |
| `conditional` | Lane-parallel if-then-else |
| `logical` | Boolean logic with bitmap kernels |
| `sort` | SIMD radix sort with optional parallel sorting |
| `unary` | Element-wise transforms |
| `vector` | Dot product, norms, weighted stats |
| `window` | Sliding window aggregations |

### Scientific Computing

| Module | Description |
|--------|-------------|
| `scientific/distributions` | 19 univariate families, 60+ functions — see below |
| `scientific/erf` | Error functions |
| `scientific/fft` | Radix-2/4/8 FFT pipelines with SIMD complex arithmetic |
| `scientific/matrix` | Dense matrix kernels |
| `scientific/scalar` | exp, ln, log10, gamma, and friends |
| `scientific/vector` | SIMD vector operations |
| `scientific/blas_lapack` | Optional BLAS/LAPACK bindings |

### Probability Distributions

The distributions module is one of the most substantial pieces of this library. Each of the 19 univariate families provides PDF, CDF, and quantile functions — over 60 kernels in total — all SIMD-accelerated where beneficial and individually validated against SciPy to high precision.

| Family | | | |
|--------|---|---|---|
| Normal | Beta | Gamma | Student's t |
| Exponential | Weibull | Cauchy | Logistic |
| Lognormal | Laplace | Chi-squared | Gumbel |
| Poisson | Binomial | Geometric | Negative Binomial |
| Hypergeometric | Uniform | Discrete Uniform | |

Each family has both a scalar fallback path and a SIMD-vectorised path. The SIMD path is selected automatically when the `simd` feature is enabled.

```rust
use simd_kernels::kernels::scientific::distributions::univariate::normal::*;

let x = &[-2.0, -1.0, 0.0, 1.0, 2.0];
let pdf = normal_pdf(x, 0.0, 1.0, None, None).unwrap();
let cdf = normal_cdf(x, 0.0, 1.0, None, None).unwrap();
```

To put this in context: SciPy's `scipy.stats` has been the gold standard for statistical distributions in Python for over a decade. SIMD-Kernels provides the same family coverage with the same level of numerical rigour, but running natively in Rust with SIMD vectorisation. Every function is tested against SciPy reference outputs across standard domains, tail regions, and known edge cases.

## Null-Mask Handling

Kernels support Apache Arrow-compatible null masks via [Minarrow](https://github.com/peterbow/minarrow):

- Null masks are opt-in. Omitting them routes directly to dense kernel paths.
- Supplying `null_count = 0` skips mask checks identically.
- Null propagation, masking, and early exits are SIMD-accelerated where possible.

This is useful in micro-batching contexts where you know data is clean.

*This crate is not affiliated with Apache Arrow. It implements Arrow-compatible null semantics and builds on Minarrow, which implements a focused subset of the Arrow specification.*

## Feature Flags

Enable only what you need. Sub 2-second compile times with defaults.

| Feature | Description |
|---------|-------------|
| `simd` | SIMD acceleration via `std::simd` (default) |
| `probability_distributions` | PDFs, CDFs, quantiles (default) |
| `fourier_transforms` | FFT operations (default) |
| `universal_functions` | Scalar maths: exp, ln, sin, etc. (default) |
| `linear_algebra` | BLAS/LAPACK via OpenBLAS |
| `parallel_sort` | Parallel sorting via Rayon |
| `simd_sort` | SIMD-accelerated radix sort for integers |
| `fast_hash` | ahash for count distinct and categorical ops |

```toml
[dependencies]
simd-kernels = { version = "0.2", features = ["linear_algebra"] }
```

## Numerical Accuracy

Getting SIMD performance is one thing. Getting it while matching SciPy's numerical output is another.

Every distribution function, special function, and scientific kernel in this library has been validated against SciPy reference outputs across standard domains, tail regions, and known difficult cases. The reference values are hardcoded from a validated x86_64 baseline and embedded directly in the test suite, so you can run the full accuracy checks on your own architecture.

Typical relative error for f64:

| Domain | Relative Error |
|--------|---------------|
| Core functions like `normal_pdf`, `gamma`, `erf` | < 1e-15 |
| Distributions across standard mean ranges | < 1e-14 |
| Heavy-tail and extreme domains | < 1e-12 |
| Boundary cases where SciPy itself becomes unstable | < 1e-10 |

**This library is provided as-is, with no warranties or guarantees of accuracy, correctness, or fitness for any purpose. Any reliance on it in critical, safety-related, or production systems is entirely at the user’s own risk. Users must independently verify all paths and outputs.**

## SIMD Configuration

Because `std::simd` is portable, you don't need to configure anything for correct vectorisation — it works out of the box. The options below are for squeezing out extra performance or testing specific ISA widths.

### Compiling

To let the compiler use your CPU's full instruction set:

```bash
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features simd
```

### Overriding Lane Widths

By default, lane widths are inferred from `CARGO_CFG_TARGET_FEATURE`. For testing or experimentation, you can override them:

```bash
# Format: "W8,W16,W32,W64"
# Example: simulate AVX-512
SIMD_LANES_OVERRIDE="64,32,16,8" \
RUSTFLAGS="-C target-cpu=native" \
cargo +nightly build --features simd
```

### Reference: SIMD Widths by Architecture

| Feature | Register Width | f64 lanes | f32 lanes | i16 lanes |
|---------|---------------|-----------|-----------|-----------|
| SSE2 | 128-bit | 2 | 4 | 8 |
| AVX/AVX2 | 256-bit | 4 | 8 | 16 |
| AVX-512 | 512-bit | 8 | 16 | 32 |
| NEON | 128-bit | 2 | 4 | 8 |
| WASM SIMD128 | 128-bit | 2 | 4 | 8 |

Check what your CPU supports with `lscpu | grep Flags` and look for `avx`, `avx2`, `avx512f`, etc.

## Going Faster

SIMD gives you parallelism within a single thread — wider lanes means more work per cycle. This library deliberately stops there. Thread-level parallelism is use-case specific, and the orchestration overhead of getting it wrong can dwarf the gains.

The intended pattern is to pair SIMD-Kernels with a threading layer of your choice. [Rayon](https://github.com/rayon-rs/rayon) is the natural fit for batch workloads. For streaming or engine contexts, a work-stealing scheduler can distribute slice-level kernel calls across cores. Either way, the kernels themselves stay single-threaded, predictable, and cache-friendly.

## Contributing

Contributions welcome:

1. **Kernel coverage** - Add kernels for missing primitives
2. **Numerical accuracy** - Validate against SciPy, write regression tests
3. **SIMD optimisation** - Improve vectorisation coverage and throughput

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Mozilla Public License (MPL) 2.0.

This strikes a balance between open ecosystem contribution and enterprise needs. If you have commercial requirements not covered by MPL-2.0, please reach out directly.
