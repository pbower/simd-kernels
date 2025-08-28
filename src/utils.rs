// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Utility Functions** - *SIMD Processing and Memory Management Utilities*
//!
//! Core utilities supporting SIMD kernel implementations with efficient memory handling,
//! bitmask operations, and performance-critical helper functions.

#[cfg(feature = "str_arithmetic")]
use std::mem::MaybeUninit;
use std::{
    collections::HashSet,
    simd::{LaneCount, Mask, MaskElement, SimdElement, SupportedLaneCount},
};

use minarrow::{Bitmask, CategoricalArray, Integer, MaskedArray, StringArray, Vec64};
#[cfg(feature = "str_arithmetic")]
use ryu::Float;

use crate::errors::KernelError;

/// Extracts a core::SIMD `Mask<M, N>` for a batch of N lanes from a Minarrow `Bitmask`.
///
/// - `mask_bytes`: packed Arrow validity bits (LSB=index 0, bit=1 means valid)
/// - `offset`: starting index (bit offset into the mask)
/// - `logical_len`: number of logical bits in the mask
/// - `M`: SIMD mask type (e.g., i64 for f64, i32 for f32, i8 for i8)
///
/// Returns: SIMD Mask<M, N> representing validity for these N lanes.
/// Bits outside the logical length (i.e., mask is shorter than offset+N)
/// are treated as valid.
#[inline(always)]
pub fn bitmask_to_simd_mask<const N: usize, M>(
    mask_bytes: &[u8],
    offset: usize,
    logical_len: usize,
) -> Mask<M, N>
where
    LaneCount<N>: SupportedLaneCount,
    M: MaskElement + SimdElement,
{
    let lane_limit = (offset + N).min(logical_len);
    let n_lanes = lane_limit - offset;
    let mut bits: u64 = 0;
    for j in 0..n_lanes {
        let idx = offset + j;
        let byte = mask_bytes[idx >> 3];
        if ((byte >> (idx & 7)) & 1) != 0 {
            bits |= 1u64 << j;
        }
    }
    if n_lanes < N {
        bits |= !0u64 << n_lanes;
    }
    Mask::<M, N>::from_bitmask(bits)
}

/// Converts a SIMD `Mask<M, N>` to a Minarrow `Bitmask` for the given logical length.
/// Used at the end of a block operation within SIMD-accelerated kernel functions.
#[inline(always)]
pub fn simd_mask_to_bitmask<const N: usize, M>(mask: Mask<M, N>, len: usize) -> Bitmask
where
    LaneCount<N>: SupportedLaneCount,
    M: MaskElement + SimdElement,
{
    let mut bits = Vec64::with_capacity((len + 7) / 8);
    bits.resize((len + 7) / 8, 0);

    let word = mask.to_bitmask();
    let bytes = word.to_le_bytes();

    let n_bytes = (len + 7) / 8;
    bits[..n_bytes].copy_from_slice(&bytes[..n_bytes]);

    if len % 8 != 0 {
        let last = n_bytes - 1;
        let mask_byte = (1u8 << (len % 8)) - 1;
        bits[last] &= mask_byte;
    }

    Bitmask {
        bits: bits.into(),
        len,
    }
}

/// Bulk-ORs a local bitmask block (from a SIMD mask or similar) into the global Minarrow bitmask at the correct byte offset.
/// The block (`block_mask`) is expected to contain at least ceil(n_lanes/8) bytes,
/// with the bit-packed validity bits starting from position 0.
///
/// Used to streamline repetitive boilerplate and ensure consistency across kernel null-mask handling.
///
/// ### Parameters
/// - `out_mask`: mutable reference to the output/global Bitmask
/// - `block_mask`: reference to the local Bitmask containing the block's bits
/// - `offset`: starting bit offset in the global mask
/// - `n_lanes`: number of bits in this block (usually SIMD lane count)
#[inline(always)]
pub fn write_global_bitmask_block(
    out_mask: &mut Bitmask,
    block_mask: &Bitmask,
    offset: usize,
    n_lanes: usize,
) {
    let n_bytes = (n_lanes + 7) / 8;
    let base = offset / 8;
    let block_bytes = &block_mask.bits[..n_bytes];
    for b in 0..n_bytes {
        if base + b < out_mask.bits.len() {
            out_mask.bits[base + b] |= block_bytes[b];
        }
    }
}

/// Determines whether nulls are present given an optional null count and mask reference.
/// Avoids computing mask cardinality to preserve performance guarantees.
#[inline(always)]
pub fn has_nulls(null_count: Option<usize>, mask: Option<&Bitmask>) -> bool {
    match null_count {
        Some(n) => n > 0,
        None => mask.is_some(),
    }
}

/// Creates a SIMD mask from a bitmask window for vectorised conditional operations.
/// 
/// Converts a contiguous section of a bitmask into a SIMD mask. 
/// The resulting mask can be used to selectively enable/disable SIMD lanes during
/// computation, providing efficient support for sparse or conditional operations.
/// 
/// # Type Parameters
/// - `T`: Mask element type implementing `MaskElement` (typically i8, i16, i32, or i64)
/// - `N`: Number of SIMD lanes, must match the SIMD vector width for the target operation
/// 
/// # Parameters
/// - `mask`: Source bitmask containing validity information
/// - `offset`: Starting bit offset within the bitmask
/// - `len`: Maximum number of bits to consider (bounds checking)
/// 
/// # Returns
/// A `Mask<T, N>` where each lane corresponds to the validity of the corresponding input element.
/// Lanes beyond `len` are set to false for safety.
/// 
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::utils::simd_mask;
/// 
/// // Create 8-lane mask for conditional SIMD operations  
/// let mask: Mask<i32, 8> = simd_mask(&bitmask, 0, 64);
/// let result = simd_vector.select(mask, default_vector);
/// ```
#[inline(always)]
pub fn simd_mask<T: MaskElement, const N: usize>(
    mask: &Bitmask,
    offset: usize,
    len: usize,
) -> Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut bits = [false; N];
    for l in 0..N {
        let idx = offset + l;
        bits[l] = idx < len && unsafe { mask.get_unchecked(idx) };
    }
    Mask::from_array(bits)
}

/// Merge two optional Bitmasks into a new output mask, computing per-row AND.
/// Returns None if both inputs are None (output is dense).
#[inline]
pub fn merge_bitmasks_to_new(
    lhs: Option<&Bitmask>,
    rhs: Option<&Bitmask>,
    len: usize,
) -> Option<Bitmask> {
    match (lhs, rhs) {
        (None, None) => None,
        (Some(l), None) | (None, Some(l)) => {
            debug_assert!(l.len() >= len, "Bitmask too short in merge");
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                out.set(i, l.get(i));
            }
            Some(out)
        }
        (Some(l), Some(r)) => {
            debug_assert!(l.len() >= len, "Left Bitmask too short in merge");
            debug_assert!(r.len() >= len, "Right Bitmask too short in merge");
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                out.set(i, l.get(i) && r.get(i));
            }
            Some(out)
        }
    }
}

/// Checks the mask capacity is large enough
/// Used so we can avoid bounds checks in the hot loop
#[inline(always)]
pub fn confirm_mask_capacity(cmp_len: usize, mask: Option<&Bitmask>) -> Result<(), KernelError> {
    if let Some(m) = mask {
        confirm_capacity("mask (Bitmask)", m.capacity(), cmp_len)?;
    }
    Ok(())
}

/// Strips '.0' from concatenated decimal values so 'Hello1.0' becomes 'Hello1'.
#[inline]
#[cfg(feature = "str_arithmetic")]
pub fn format_finite<F: Float>(buf: &mut [MaybeUninit<u8>; 24], f: F) -> &str {
    unsafe {
        let ptr = buf.as_mut_ptr() as *mut u8;
        let n = f.write_to_ryu_buffer(ptr);
        debug_assert!(n <= buf.len());

        let slice = core::slice::from_raw_parts(ptr, n);
        let s = core::str::from_utf8_unchecked(slice);

        // Strip trailing ".0" if present
        if s.ends_with(".0") {
            let trimmed_len = s.len() - 2;
            core::str::from_utf8_unchecked(&slice[..trimmed_len])
        } else {
            s
        }
    }
}

/// Estimate cardinality ratio on a sample from a CategoricalArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_categorical_cardinality(cat: &CategoricalArray<u32>, sample_size: usize) -> f64 {
    let len = cat.data.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { cat.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

/// Estimate cardinality ratio on a sample from a StringArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_string_cardinality<T: Integer>(arr: &StringArray<T>, sample_size: usize) -> f64 {
    let len = arr.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { arr.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

/// Validates that actual capacity matches expected capacity for kernel operations.
/// 
/// Essential validation function used throughout the kernel library to ensure data structure
/// capacities are correct before performing operations. Prevents buffer overruns and ensures
/// memory safety by catching capacity mismatches early with descriptive error messages.
/// 
/// # Parameters
/// - `label`: Descriptive label for the validation context (used in error messages)
/// - `actual`: The actual capacity of the data structure being validated
/// - `expected`: The expected capacity required for the operation
/// 
/// # Returns
/// `Ok(())` if capacities match, otherwise `KernelError::InvalidArguments` with detailed message.
/// 
/// # Error Conditions
/// Returns `KernelError::InvalidArguments` when `actual != expected`, providing a clear
/// error message indicating the mismatch and context.
#[inline(always)]
pub fn confirm_capacity(label: &str, actual: usize, expected: usize) -> Result<(), KernelError> {
    if actual != expected {
        return Err(KernelError::InvalidArguments(format!(
            "{}: capacity mismatch (expected {}, got {})",
            label, expected, actual
        )));
    }
    Ok(())
}

/// Validates that two lengths are equal for binary kernel operations.
/// 
/// Critical validation function ensuring input arrays have matching lengths before performing
/// binary operations like comparisons, arithmetic, or logical operations. Prevents undefined
/// behaviour and provides clear error diagnostics when length mismatches occur.
/// 
/// # Parameters
/// - `label`: Descriptive context label for error reporting (e.g., "compare numeric")
/// - `a`: Length of the first input array or data structure
/// - `b`: Length of the second input array or data structure
/// 
/// # Returns
/// `Ok(())` if lengths are equal, otherwise `KernelError::LengthMismatch` with diagnostic details.
#[inline(always)]
pub fn confirm_equal_len(label: &str, a: usize, b: usize) -> Result<(), KernelError> {
    if a != b {
        return Err(KernelError::LengthMismatch(format!(
            "{}: length mismatch (lhs: {}, rhs: {})",
            label, a, b
        )));
    }
    Ok(())
}

/// SIMD Alignment check. Returns true if the slice is properly
/// 64-byte aligned for SIMD operations, false otherwise.
#[inline(always)]
pub fn is_simd_aligned<T>(slice: &[T]) -> bool {
    if slice.is_empty() {
        true
    } else {
        (slice.as_ptr() as usize) % 64 == 0
    }
}
