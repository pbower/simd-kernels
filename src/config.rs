// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

// These parameters should rarely need adjustment.

//! # **Configuration Constants** - *Runtime Behaviour Parameters*
//!
//! Global configuration constants controlling kernel behaviour and performance thresholds.
//! These values are compile-time constants optimised for typical workloads.

/// Maximum allowed repetitions for string multiplication operations.
///
/// Prevents excessive memory allocation when repeating strings through multiplication.
/// Operations exceeding this limit will return an error rather than allocating unbounded memory.
pub const STRING_MULTIPLICATION_LIMIT: usize = 1_000_000;

/// Threshold for dictionary size checks in categorical array operations.
///
/// Controls when the `cmp_dict_in` function in `kernels/logical.rs` switches from dictionary
/// lookups to direct string comparisons. Arrays with fewer unique values than this threshold
/// use optimised dictionary-based comparisons.
pub const MAX_DICT_CHECK: usize = 256;