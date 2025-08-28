# SIMD-Kernels – *Lightning Fast, Arrow-Compatible Compute Kernels*

## Intro

_Welcome to `SIMD-Kernels`._

`SIMD-Kernels` is a modern library of compute kernels built on top of [`std::simd`](https://doc.rust-lang.org/std/simd/) for high-performance analytics and scientific computing in Rust.  
It implements the core arithmetic, statistical, logical, and scientific operations required for data systems —  
accelerated with SIMD, aligned for cache efficiency, and compatible with the Apache Arrow model.

The kernels it implements form the computational core of the analytics stack, and it integrates cleanly with the [`minarrow`](https://github.com/lightning-lib/minarrow) columnar runtime.

## Design Focus

- **SIMD-first execution** – Built directly on `std::simd`  
- **Null-aware semantics** – Full support for Arrow-style null masks  
- **Typed, minimal interfaces** – No dynamic typing, no downcasting  
- **Production-grade kernels** – Matches SciPy and BLAS/LAPACK numerics  
- **Configurable footprint** – Feature-gated compilation, opt-in performance semantics.

## Why I built SIMD-Kernels

- **Most compute kernels today are generic, boxed, or opaque**. This creates unnecessary branches, cache misses, and dynamic lookup overhead in performance-critical systems.
- **Rust deserves a low-level kernel library** — typed, SIMD-native, and engineered for real-time and HPC workloads.
- `SIMD-kernels` focuses on ergonomics, throughput, and correctness—designed to **maximise hardware utilisation**, whilst remaining easy to use for modern data breadth.

---

## Key Features

### SIMD by Default

- All operations are vectorised using `std::simd` with auto-vectorising fallback for scalar lanes.
- Native CPU features are respected using `RUSTFLAGS="-C target-cpu=native"`.
- 64-byte aligned buffers via `Vec64`, with alignment checks for SIMD correctness.

### Null-Aware Execution

- Every kernel is null-mask aware and consistent with Apache-Arrow null semantics.
- Null propagation, masking, and early exits are SIMD-accelerated where possible.

### Full Numeric Coverage

- **Arithmetic**: All numeric types (`i8`–`u64`, `f32`/`f64`) with overflow handling.
- **Statistics**: Mean, variance, standard deviation, z-score normalisation.
- **Probability Distributions**: 19 PDFs, CDFs, and quantiles with <1e-12 error bounds.
- **Scientific Functions**: `erf`, `gamma`, FFT, matrix/vector ops.
- **Linear Algebra**: Optional BLAS/LAPACK backend integration.
- **FFT**: Blocked radix-2/4/8 pipelines with SIMD and complex arithmetic.

### Feature Flags

Modular by design. Enable only what you need:

- `linear_algebra` – BLAS/LAPACK via system libraries  
- `probability_distributions` – PDFs, CDFs, quantiles  
- `fourier_transforms` – FFT operations  
- `universal_functions` – Scalar maths: exp, ln, sin, etc.

```toml
[features]
default = []
linear_algebra = ["blas-src"]
probability_distributions = []
fourier_transforms = []
```

### Repo Structure

simd-kernels is divided into tightly scoped submodules:

```markdown
simd-kernels/
├── kernels/
│   ├── arithmetic/       # SIMD + null-safe arithmetic
│   ├── aggregate/        # Sum, mean, variance, etc.
│   ├── comparison/       # SIMD comparisons
│   ├── logical/          # Boolean logic (AND, OR, XOR)
│   ├── conditional/      # if-then-else kernels
│   ├── string/           # String processing
│   ├── window/           # Sliding window kernels
│   ├── binary/           # Bitwise ops
│   ├── sort/             # Parallel SIMD sort kernels
│   ├── scientific/       # Special functions + FFT + matrix
│   │   ├── distributions/ # PDFs, CDFs, quantiles
│   │   ├── erf/           # Error functions
│   │   ├── fft/           # FFT pipelines
│   │   ├── matrix/        # Dense matrix kernels
│   │   ├── vector/        # SIMD vector ops
│   │   └── blas_lapack/   # External LAPACK bindings
├── traits/               # Kernel traits + marker traits
├── config/               # Compile-time feature flags
├── errors/               # KernelError definitions
└── utils/                # Internal helpers (alignment, dispatch)
```

### Example: SIMD Arithmetic with Nulls

```rust
use simd_kernels::kernels::arithmetic::add_f64_dense;
use minarrow::{FloatArray, arr_f64};

let a = arr_f64![1.0, 2.0, 3.0];
let b = arr_f64![10.0, 20.0, 30.0];

let result = add_f64_dense(&a, &b).unwrap();
assert_eq!(result.values(), &[11.0, 22.0, 33.0]);
```

All arithmetic kernels use SIMD internally and support both dense and null-masked variants.
This is particularly effective for fused multiply-add (FMA) kernels.

### Null mask handling

SIMD-kernels supports **Apache-Arrow** compatible null-masks via **Minarrow**.

- Null masks are opt-in
- leaving them out skips checks by going straight to dense kernel versions. 
- Additionally, one can supply a `null_count = 0` to skip it similarly. 

This is useful in micro-batching contexts. 

If you use **Minarrow**, you get this easily and with zero-copy semantics from very low-overhead types.  

*Note: this crate is not affiliated with Apache Arrow, however it implements Arrow-compatible null-semantics, and builds on *Minarrow*, which implements a focused subset of the Apache Arrow specification.*

### SIMD Kernel Coverage

| Operation               | SIMD Support                       |
| ----------------------- | ---------------------------------- |
| `+ - * / %`             | ✅ All numeric types                |
| `< <= == != >= >`       | ✅ SIMD mask comparisons            |
| `is_nan`, `is_null`     | ✅ SIMD + bitmap logic              |
| `exp`, `ln`, `log10`    | ✅ SIMD ufuncs                      |
| `normal_pdf`            | ⚠️ Yes, but only where it makes sense. **21 univariate families (60+ functions!) are implemented, and tested against SciPy**. *Roughly half of these are SIMD accelerated*.|
| `fft8_radix`            | ✅ DIT radix-8 via SIMD complex ops |
| `matmul`, `dot`, `axpy` | ✅ (optional via `linear_algebra`)  |
| `if_else`               | ✅ SIMD-lane conditional            |
| `sum`, `mean`, `stdev`  | ✅ SIMD + null-aware                |
| `regex_match`           | ✅ via regex crate                  |
| `sort`                  | ✅ SIMD radix sort                  |
| |    |

## Accuracy Targets

Most statistical and scientific functions achieve relative error < 1e-15 for f64 compared to SciPy on standard domains.

Typical accuracy in the integration test suite is:

< 1e-15 for core functions (e.g., normal_pdf, gamma, erf)
< 1e-14 for distributions across mean ranges
< 1e-12 in certain heavy-tail or extreme domains
< 1e-10 in certain boundary cases, where SciPy itself becomes numerically unstable

Each implementation is tested against reference outputs from SciPy, hardcoded from a valid baseline x86_64 platform.
These values are embedded in the test suite, so you can run full accuracy tests on your own architecture *(and expect minor floating point tolerances)*.

That said, no accuracy guarantees are made. This library is new, and while SciPy has benefited from over a decade of numerical tuning and user feedback, simd-kernels is still maturing. If you require strict numerical guarantees, you must perform your own validations on all critical paths.

We make no guarantees regarding numerical accuracy. If you rely on this library in critical contexts, you must perform your own validation. 
Use is at your own risk.

## Performance Model
* SIMD auto-vectorisation with std::simd::Simd<T, LANES>
* Fall back to scalar loop for non-aligned or low-lane inputs
* Automatic 64-byte aligned inputs via **Minarrow's** `Vec64`, which is the standard
Vec with a custom 64-byte allocator (*demonstrated in that repo's benchmarks to be practically as fast as the standard Vec*).
* Null-masked iteration is lane-parallel when possible, and bitmask kernels are already highly efficient through word-based kernels.

## Compiling
Make sure SIMD compiles correctly with:
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features simd

## Overriding SIMD Lane Widths
By default, simd-kernels uses conservative architecture-specific lane widths inferred from CARGO_CFG_TARGET_FEATURE.
However, you may override these lane counts at build time to experiment or test alternate configurations. 

Set the environment variable SIMD_LANES_OVERRIDE before compiling:
```bash
# Format: "W8,W16,W32,W64"
# For example, simulate AVX-512:
SIMD_LANES_OVERRIDE="64,32,16,8" \
RUSTFLAGS="-C target-cpu=native" \
cargo +nightly build --features simd
```

This will override the automatically detected SIMD widths with:

* `W8 = 64` (e.g. u8, i8)
* `W16 = 32` (e.g. u16, i16)
* `W32 = 16` (e.g. f32, i32)
* `W64 = 8` (e.g. f64, i64)

### Finding out what your machine supports
An easy way is to check what SIMD lanes your CPU actually supports is `lscpu | grep -i width` in bash, then look for the 'Flags' section. Then, the flags will show text e.g., *avx, avx2, avx512f* etc. This is how to then interpret that, so you can set the flags accordingly:

| Feature      | Register Width | Lane Count (f64) | Lane Count (f32) | Lane Count (i16) |
| ------------ | -------------- | ---------------- | ---------------- | ---------------- |
| SSE2         | 128 bits       | 2                | 4                | 8                |
| AVX          | 256 bits       | 4                | 8                | 16               |
| AVX-512      | 512 bits       | 8                | 16               | 32               |
| NEON         | 128 bits       | 2                | 4                | 8                |
| WASM SIMD128 | 128 bits       | 2                | 4                | 8                |

For e.g., W64 should be set to '2' if you are on SSE2 with a consumer laptop.

## Going faster
Obviously, the more lanes, the more parallel your computations will be within the same 
thread. However this library purposely excludes thread-parallel computations given
that it is use case specific, with millisecond-level orchestration overhead. Combining 
this library with `Rayon` will make data fast. `Minarrow` supports this pattern natively.

## Target Use Cases

| Use Case                        | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| Extreme low-latency computation | Very low abstraction overhead, direct kernels     |
| Engine Kernel Layer             | High-throughput compute for execution engines     |
| Statistical Pipelines           | SIMD evaluation of distributions + aggregates     |
| Signal Processing               | FFTs, filters, and transforms                     |
| Vectorised Scientific Computing | Accurate special functions                        |
| Columnar DBMS                   | Null-aware SIMD kernels for query pipelines       |
| Embedded Systems                | Compile-time feature gating for footprint control |

## Philosophy

**Flexible** – Every kernel is statically typed and callable with no dynamic dispatch. Numerical kernels are all slice-compatible, so they support diverse entry
contexts.

**Fast** – Always use SIMD lanes where possible, fallback only when required.
Guaranteed 64-byte alignment when using `Minarrow`'s `Vec64`, `IntegerArray` or `FloatArray` types.

**Composable** – Minimal deps, fast builds, clean layering.

**Feature-rich** – Proper mask propagation and bitmap handling, even on
univariate distributions. Or, opt out completely for standard float NaN semantics.

**Compatible** – When using via **minarrow**, you get FFI compatible buffers, `.to_apache_arrow()` and `.to_polars()`.


## Contributing

We welcome contributions in the following areas:

* Kernel Coverage – Add kernels for missing primitives
* Numerical Accuracy – Validate against SciPy, write regression tests
* SIMD Optimisation – Improve SIMD coverage, improve speed

See CONTRIBUTING.md for guidance.

## Benchmarks

Coming soon.

## License

Licensed under the Mozilla Public License (MPL) 2.0

This license is in place to strike a balance between open ecosystem contribution, developer
and enterprise needs. 

If you have commercial requirements not covered by this license, please reach out directly.

## Feedback

Please open an issue or reach out with ideas, requests, or contributions.