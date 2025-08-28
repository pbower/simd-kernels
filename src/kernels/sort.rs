// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Sorting Algorithms Kernels Module** - *Array Sorting and Ordering Operations*
//!
//! Sorting kernels for ordering operations across Arrow-compatible 
//! data types with null-aware semantics and optimised comparison strategies. Essential foundation
//! for analytical operations requiring ordered data access and ranking computations.
//!
use std::cmp::Ordering;

use minarrow::{
    Bitmask, BooleanArray, CategoricalArray, Integer, MaskedArray, Vec64,
    traits::type_unions::Float,
};
use num_traits::{NumCast, Zero};

/// Total ordering for f32/f64 as per IEEE 754
///
/// - NaN sorts greater than all numbers, including +inf.
/// - -0.0 and +0.0 are distinct (we can change in future if needed - to filter first).
/// - Preserves all edge case ordering.

#[inline]
pub fn sort_float<T: Float>(slice: &mut [T]) {
    slice.sort_unstable_by(total_cmp_f);
}
/// TLDR: Instead of comparing the raw bit patterns directly (which places all negative‐sign bit values after the positives),
/// we transform each bit pattern into a “sort key”:
///     => If the sign bit is 1 (negative or negative‐NaN), we invert all bits: !bits.
///     => Otherwise (sign bit 0), we flip only the sign bit: bits ^ 0x80…0.
#[inline(always)]
pub fn total_cmp_f<T: Float>(a: &T, b: &T) -> Ordering {
    // We reinterpret the bits of `T` as either u32 or u64:
    if std::mem::size_of::<T>() == 4 {
        // f32 path
        let ua = unsafe { *(a as *const T as *const u32) };
        let ub = unsafe { *(b as *const T as *const u32) };
        // Negative values get inverted; non-negatives get their sign bit flipped.
        let ka = if ua & 0x8000_0000 != 0 {
            !ua
        } else {
            ua ^ 0x8000_0000
        };
        let kb = if ub & 0x8000_0000 != 0 {
            !ub
        } else {
            ub ^ 0x8000_0000
        };
        ka.cmp(&kb)
    } else if std::mem::size_of::<T>() == 8 {
        // f64 path
        let ua = unsafe { *(a as *const T as *const u64) };
        let ub = unsafe { *(b as *const T as *const u64) };
        let ka = if ua & 0x8000_0000_0000_0000 != 0 {
            !ua
        } else {
            ua ^ 0x8000_0000_0000_0000
        };
        let kb = if ub & 0x8000_0000_0000_0000 != 0 {
            !ub
        } else {
            ub ^ 0x8000_0000_0000_0000
        };
        ka.cmp(&kb)
    } else {
        unreachable!("Only f32 and f64 are valid Float types.")
    }
}

/// Returns a newly sorted Vec64, leaving the original slice untouched.
pub fn sorted_float<T: Float>(data: &[T]) -> Vec64<T> {
    let mut v = Vec64::from_slice(data);
    sort_float(&mut v);
    v
}

/// Performs in-place unstable sorting of integer slices with optimal performance.
/// 
/// High-performance sorting function optimised for integer types using unstable sort algorithms
/// that prioritise speed over preserving the relative order of equal elements. 
/// 
/// # Type Parameters
/// - `T`: Integer type implementing `Ord + Copy` (i8, i16, i32, i64, u8, u16, u32, u64, etc.)
/// 
/// # Parameters
/// - `slice`: Mutable slice to be sorted in-place
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sort_int;
/// 
/// let mut data = [64i32, 34, 25, 12, 22, 11, 90];
/// sort_int(&mut data);
/// // data is now [11, 12, 22, 25, 34, 64, 90]
/// ```
#[inline]
pub fn sort_int<T: Ord + Copy>(slice: &mut [T]) {
    slice.sort_unstable();
}

/// Creates a new sorted copy of integer data in a Vec64 container.
/// 
/// Clones input data into a Vec64 and sorts it using the optimised
/// integer sorting algorithm. Returns a new sorted container while leaving the original data
/// unchanged.
/// 
/// # Type Parameters
/// - `T`: Integer type implementing `Ord + Copy` (i8, i16, i32, i64, u8, u16, u32, u64, etc.)
/// 
/// # Parameters
/// - `data`: Source slice to be copied and sorted
/// 
/// # Returns
/// A new `Vec64<T>` containing the sorted elements from the input slice.
/// 
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sorted_int;
/// 
/// let data = [64i32, 34, 25, 12, 22, 11, 90];
/// let sorted = sorted_int(&data);
/// // sorted contains [11, 12, 22, 25, 34, 64, 90]
/// // original data unchanged
/// ```
pub fn sorted_int<T: Ord + Copy>(data: &[T]) -> Vec64<T> {
    let mut v = Vec64::from_slice(data);
    sort_int(&mut v);
    v
}

/// Performs in-place unstable sorting of string slice references with lexicographic ordering.
/// 
/// High-performance sorting function for string references using unstable sort algorithms.
/// Efficient lexicographic ordering for string processing, text analysis, and data organisation tasks.
/// 
/// # Parameters
/// - `slice`: Mutable slice of string references to be sorted in-place
/// 
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sort_str;
/// 
/// let mut words = ["zebra", "apple", "banana", "cherry"];
/// sort_str(&mut words);
/// // words is now ["apple", "banana", "cherry", "zebra"]
/// ```
#[inline]
pub fn sort_str(slice: &mut [&str]) {
    slice.sort_unstable();
}

/// Creates a new sorted collection of owned strings from string references.
/// 
/// Copies string references into owned String objects within a Vec64
/// container and sorts them lexicographically. Returns a new sorted collection while preserving
/// the original string references for scenarios requiring owned string data.
/// 
/// # Parameters
/// - `data`: Source slice of string references to be copied and sorted
/// 
/// # Returns
/// A new `Vec64<String>` containing owned, sorted string copies from the input references.
/// 
/// # Memory Allocation
/// - String allocation: Each string reference copied into a new String object
/// - Container allocation: Vec64 provides 64-byte aligned storage for optimal performance
/// - Heap usage: Total memory proportional to sum of string lengths plus container overhead
/// 
/// # Performance Characteristics
/// - O(n) string allocation and copying (proportional to total string length)
/// - O(n log n) sorting time complexity with string comparison overhead
/// - Memory overhead: Requires additional heap space for all string content
/// 
/// # String Ownership
/// - Input references: Original string references remain unchanged
/// - Output strings: New owned String objects independent of input lifetime
/// - Memory management: Vec64<String> handles automatic cleanup of owned strings
/// 
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sorted_str;
/// 
/// let words = ["zebra", "apple", "banana", "cherry"];
/// let sorted = sorted_str(&words);
/// // sorted contains owned Strings: ["apple", "banana", "cherry", "zebra"]
/// // original words slice unchanged
/// ```
pub fn sorted_str(data: &[&str]) -> Vec64<String> {
    let mut v = Vec64::from_slice(data);
    sort_str(&mut v);
    v.iter().map(|s| s.to_string()).collect()
}

/// For StringArray as contiguous utf8 buffer plus offsets.
/// Assumes offsets + values as in minarrow StringArray.
pub fn sort_string_array(offsets: &[usize], values: &str) -> (Vec64<usize>, String) {
    let n = offsets.len() - 1;
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_unstable_by(|&i, &j| {
        let a = &values[offsets[i]..offsets[i + 1]];
        let b = &values[offsets[j]..offsets[j + 1]];
        a.cmp(b)
    });

    // Precompute total length for the output string buffer
    let total_bytes: usize = indices
        .iter()
        .map(|&idx| offsets[idx + 1] - offsets[idx])
        .sum();

    // Preallocate buffers
    let mut new_values = String::with_capacity(total_bytes);
    let mut new_offsets = Vec64::<usize>::with_capacity(n + 1);

    // Pre-extend buffers for unchecked writes
    unsafe {
        new_offsets.set_len(n + 1);
    }
    unsafe {
        new_values.as_mut_vec().set_len(total_bytes);
    }

    let values_bytes = values.as_bytes();
    let out_bytes = unsafe { new_values.as_mut_vec() };

    // First offset is always 0
    unsafe {
        *new_offsets.get_unchecked_mut(0) = 0;
    }
    let mut current_offset = 0;
    let mut out_ptr = 0;
    for (i, &idx) in indices.iter().enumerate() {
        let start = offsets[idx];
        let end = offsets[idx + 1];
        let len = end - start;
        // Copy string bytes
        unsafe {
            std::ptr::copy_nonoverlapping(
                values_bytes.as_ptr().add(start),
                out_bytes.as_mut_ptr().add(out_ptr),
                len,
            );
            current_offset += len;
            *new_offsets.get_unchecked_mut(i + 1) = current_offset;
        }
        out_ptr += len;
    }
    // SAFETY: We filled up to `total_bytes`
    unsafe {
        new_values.as_mut_vec().set_len(current_offset);
    }

    (new_offsets, new_values)
}

/// Sorts a CategoricalArray lexically by its unique values, returning new indices and mask.
/// The category dictionary is preserved. Nulls sort first.
pub fn sort_categorical_lexical<T: Ord + Copy + Zero + NumCast + Integer>(
    cat: &CategoricalArray<T>,
) -> (Vec64<T>, Option<Bitmask>) {
    let len = cat.data.len();
    let mut indices: Vec<usize> = (0..len).collect();

    // Sort indices: nulls first, then by value, stable.
    indices.sort_by(|&i, &j| {
        let a_is_null = cat.is_null(i);
        let b_is_null = cat.is_null(j);
        match (a_is_null, b_is_null) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                // compare by the string value of the keys
                let a_key = &cat.unique_values[cat.data[i].to_usize()];
                let b_key = &cat.unique_values[cat.data[j].to_usize()];
                a_key.cmp(b_key)
            }
        }
    });

    // Build output data and mask
    let mut sorted_data = Vec64::<T>::with_capacity(len);
    let mut sorted_mask = cat
        .null_mask
        .as_ref()
        .map(|_| Bitmask::new_set_all(len, false));

    for (out_i, &orig_i) in indices.iter().enumerate() {
        sorted_data.push(cat.data[orig_i]);
        if let Some(ref mask) = cat.null_mask {
            if let Some(ref mut sm) = sorted_mask {
                if unsafe { mask.get_unchecked(orig_i) } {
                    unsafe { sm.set_unchecked(out_i, true) };
                }
            }
        }
    }

    (sorted_data, sorted_mask)
}

/// Unpacks a Bitmask into a Vec<bool>
#[inline]
fn unpack_bitmask(mask: &Bitmask) -> Vec64<bool> {
    let mut out = Vec64::with_capacity(mask.len);
    for i in 0..mask.len {
        out.push(unsafe { mask.get_unchecked(i) });
    }
    out
}

/// Packs a Vec<bool> into a Bitmask
#[inline]
fn pack_bitmask(bits: &[bool]) -> Bitmask {
    let mut mask = Bitmask::new_set_all(bits.len(), false);
    for (i, &b) in bits.iter().enumerate() {
        if b {
            unsafe { mask.set_unchecked(i, true) };
        }
    }
    mask
}

/// Sorts a BooleanArray in-place by value: all false first, then true.
/// Nulls sort first if present.
pub fn sort_boolean_array(arr: &mut BooleanArray<()>) {
    let bits: Vec64<bool> = unpack_bitmask(&arr.data);
    let nulls: Option<Vec64<bool>> = arr.null_mask.as_ref().map(unpack_bitmask);

    let mut indices: Vec<usize> = (0..arr.len).collect();

    // Nulls sort first (is_null = !is_valid)
    indices.sort_unstable_by(|&i, &j| {
        let a_null = !nulls.as_ref().map_or(true, |n| n[i]);
        let b_null = !nulls.as_ref().map_or(true, |n| n[j]);

        match (a_null, b_null) {
            (true, true) => Ordering::Equal,    // both null
            (true, false) => Ordering::Less,    // null < non-null
            (false, true) => Ordering::Greater, // non-null > null
            (false, false) => {
                // Both non-null: false < true
                match (bits[i], bits[j]) {
                    (false, false) => Ordering::Equal,
                    (false, true) => Ordering::Less,
                    (true, false) => Ordering::Greater,
                    (true, true) => Ordering::Equal,
                }
            }
        }
    });

    // Re-pack sorted logical values, forcing value=false for null slots
    let sorted_bits: Vec<bool> = indices
        .iter()
        .map(|&idx| {
            let is_null = !nulls.as_ref().map_or(true, |n| n[idx]);
            if is_null { false } else { bits[idx] }
        })
        .collect();
    arr.data = pack_bitmask(&sorted_bits);

    // Re-pack the (optional) null mask.
    if let Some(null_mask) = arr.null_mask.as_mut() {
        let sorted_nulls: Vec<bool> = indices
            .iter()
            .map(|&idx| nulls.as_ref().unwrap()[idx])
            .collect();
        *null_mask = pack_bitmask(&sorted_nulls);
    }
}

/// Sorts array data and applies the same permutation to the null mask.
pub fn sort_slice_with_mask<T: Ord + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
) -> (Vec64<T>, Option<Bitmask>) {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&i, &j| data[i].cmp(&data[j]));

    let mut sorted_data = Vec64::<T>::with_capacity(n);
    unsafe {
        sorted_data.set_len(n);
    }
    for (i, &idx) in indices.iter().enumerate() {
        unsafe {
            *sorted_data.get_unchecked_mut(i) = data[idx];
        }
    }

    let sorted_mask = mask.map(|m| {
        let mut out = Bitmask::new_set_all(n, false);
        for (i, &idx) in indices.iter().enumerate() {
            if unsafe { m.get_unchecked(idx) } {
                unsafe { out.set_unchecked(i, true) };
            }
        }
        out
    });

    (sorted_data, sorted_mask)
}

#[cfg(test)]
mod tests {
    use minarrow::vec64;

    use super::*;

    #[test]
    fn test_total_cmp_f32_ordering_basic() {
        let a = 1.0f32;
        let b = 2.0f32;
        assert_eq!(total_cmp_f(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(total_cmp_f(&b, &a), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&a, &a), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_f64_ordering_basic() {
        let a = 1.0f64;
        let b = 2.0f64;
        assert_eq!(total_cmp_f(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(total_cmp_f(&b, &a), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&a, &a), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_zero_and_negzero() {
        let pz = 0.0f32;
        let nz = -0.0f32;
        // -0.0 should not equal 0.0 in bitwise comparison
        assert_ne!(f32::to_bits(pz), f32::to_bits(nz));
        assert_ne!(total_cmp_f(&pz, &nz), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_nan_ordering_f32() {
        let nan = f32::NAN;
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let one = 1.0f32;

        // NaN is greater than everything in this ordering
        assert_eq!(total_cmp_f(&nan, &pos_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &neg_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &one), std::cmp::Ordering::Greater);
        // Self-equality
        assert_eq!(total_cmp_f(&nan, &nan), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_nan_ordering_f64() {
        let nan = f64::NAN;
        let pos_inf = f64::INFINITY;
        let neg_inf = f64::NEG_INFINITY;
        let one = 1.0f64;

        assert_eq!(total_cmp_f(&nan, &pos_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &neg_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &one), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &nan), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_sort_float_f32_all_edge_cases() {
        let mut v = vec64![
            3.0f32,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            f32::NAN,
            2.0,
            -2.0,
        ];
        sort_float(&mut v);
        // Sorted by bit-pattern, not by value
        // -2.0 < -1.0 < -0.0 < 0.0 < 1.0 < 2.0 < 3.0 < INF < NAN
        assert_eq!(v[0], f32::NEG_INFINITY);
        assert_eq!(v[1], -2.0);
        assert_eq!(v[2], -1.0);
        assert_eq!(v[3], -0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 1.0);
        assert_eq!(v[6], 2.0);
        assert_eq!(v[7], 3.0);
        assert_eq!(v[8], f32::INFINITY);
        assert!(v[9].is_nan());
    }

    #[test]
    fn test_sort_float_f64_all_edge_cases() {
        let mut v = vec64![
            3.0f64,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            -1.0,
            f64::NAN,
            2.0,
            -2.0,
        ];
        sort_float(&mut v);
        assert_eq!(v[0], f64::NEG_INFINITY);
        assert_eq!(v[1], -2.0);
        assert_eq!(v[2], -1.0);
        assert_eq!(v[3], -0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 1.0);
        assert_eq!(v[6], 2.0);
        assert_eq!(v[7], 3.0);
        assert_eq!(v[8], f64::INFINITY);
        assert!(v[9].is_nan());
    }

    #[test]
    fn test_sorted_float_immutability_and_return_type() {
        let v = vec64![1.0f32, 2.0, 3.0];
        let out = sorted_float(&v);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0]);
        // Ensure original is unchanged
        assert_eq!(*v, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sorted_float_correct_for_f64() {
        let v = vec64![3.0f64, 2.0, 1.0];
        let out = sorted_float(&v);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_float_empty_and_single() {
        let mut v: [f32; 0] = [];
        sort_float(&mut v);
        let mut v2 = [42.0f32];
        sort_float(&mut v2);
        assert_eq!(v2, [42.0]);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use minarrow::{Vec64, vec64};

        #[test]
        fn test_sort_int_empty_and_single() {
            let mut arr: [i32; 0] = [];
            sort_int(&mut arr);
            assert_eq!(arr, [] as [i32; 0]);
            let mut arr2 = vec64![42];
            sort_int(&mut arr2);
            assert_eq!(*arr2, [42]);
        }

        #[test]
        fn test_sort_int_order() {
            let mut arr = vec64![4, 2, 7, 1, 1, 6, 0, -5];
            sort_int(&mut arr);
            assert_eq!(*arr, [-5, 0, 1, 1, 2, 4, 6, 7]);
        }

        #[test]
        fn test_sorted_int_immutability_and_output() {
            let arr = vec64![5, 3, 7, 1, 2];
            let sorted = sorted_int(&arr);
            assert_eq!(sorted.as_slice(), &[1, 2, 3, 5, 7]);
            // ensure input not changed
            assert_eq!(*arr, [5, 3, 7, 1, 2]);
        }

        #[test]
        fn test_sort_str_basic() {
            let mut arr = vec64!["z", "b", "a", "d"];
            sort_str(&mut arr);
            assert_eq!(*arr, ["a", "b", "d", "z"]);
        }

        #[test]
        fn test_sorted_str_and_non_ascii() {
            let arr = vec64!["z", "ä", "b", "a", "d"];
            let sorted = sorted_str(&arr);
            assert_eq!(sorted.as_slice(), &["a", "b", "d", "z", "ä"]); // note: utf8, ä > z in Rust
            // input is untouched
            assert_eq!(*arr, ["z", "ä", "b", "a", "d"]);
        }

        #[test]
        fn test_sort_string_array_basic() {
            let offsets = vec![0, 1, 3, 5, 5, 6]; // ["a", "bc", "de", "", "f"]
            let values = "abcde".to_string() + "f";
            let (new_offsets, new_values) = sort_string_array(&offsets, &values);
            // Sorted order: "", "a", "bc", "de", "f"
            let slices: Vec<_> = new_offsets
                .windows(2)
                .map(|w| &new_values[w[0]..w[1]])
                .collect();
            assert_eq!(slices, &["", "a", "bc", "de", "f"]);
        }

        #[test]
        fn test_sort_categorical_lexical_basic_and_nulls() {
            // Categories: 0 = "apple", 1 = "banana", 2 = "pear"
            let unique = Vec64::from_slice_clone(&[
                "apple".to_string(),
                "banana".to_string(),
                "pear".to_string(),
            ]);
            let data = Vec64::from_slice(&[2u8, 0, 1, 1, 2, 0, 1]);
            let mask = Bitmask::from_bools(&[true, false, true, true, true, false, true]);
            let cat = CategoricalArray {
                data: data.into(),
                unique_values: unique,
                null_mask: Some(mask.clone()),
            };
            let (sorted_data, sorted_mask) = sort_categorical_lexical(&cat);

            // Nulls first
            let mask_out = sorted_mask.unwrap();
            let null_pos: Vec<_> = (0..mask_out.len()).filter(|&i| !mask_out.get(i)).collect();
            assert_eq!(null_pos, &[0, 1]);

            // Remaining valid values
            let sorted_as_str: Vec<_> = sorted_data
                .iter()
                .map(|&k| &cat.unique_values[k.to_usize()][..])
                .collect();
            let vals = &sorted_as_str[null_pos.len()..];
            assert_eq!(vals, &["banana", "banana", "banana", "pear", "pear"]);
        }

        #[test]
        fn test_sort_categorical_all_nulls_and_no_nulls() {
            // All null
            let unique = Vec64::from_slice_clone(&["x".to_string()]);
            let data = Vec64::from_slice(&[0u8, 0, 0]);
            let mask = Bitmask::from_bools(&[false, false, false]);
            let cat = CategoricalArray {
                data: data.clone().into(),
                unique_values: unique.clone(),
                null_mask: Some(mask.clone()),
            };
            let (_, sorted_mask) = sort_categorical_lexical(&cat);
            assert_eq!(
                unpack_bitmask(&sorted_mask.unwrap()),
                vec64![false, false, false]
            );
            // No nulls
            let mask2 = Bitmask::from_bools(&[true, true, true]);
            let cat2 = CategoricalArray {
                data: data.clone().into(),
                unique_values: unique,
                null_mask: Some(mask2.clone()),
            };
            let (_, sorted_mask2) = sort_categorical_lexical(&cat2);
            assert_eq!(
                unpack_bitmask(&sorted_mask2.unwrap()),
                vec64![true, true, true]
            );
        }
        #[test]
        fn test_sort_boolean_array_with_nulls() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[false, true, false, true, true, false]),
                null_mask: Some(pack_bitmask(&[true, false, true, true, false, true])),
                len: 6,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            // Nulls first (mask false)
            let bits = unpack_bitmask(&arr.data);
            let nulls = unpack_bitmask(arr.null_mask.as_ref().unwrap());
            assert_eq!(nulls, vec64![false, false, true, true, true, true]);
            // Sorted data for valid (true): all false first, then true
            assert_eq!(&bits[2..], [false, false, false, true]);
        }

        #[test]
        fn test_sort_boolean_array_all_true_and_all_false() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[true, true, true]),
                null_mask: None,
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            assert_eq!(unpack_bitmask(&arr.data), vec64![true, true, true]);

            let mut arr2 = BooleanArray {
                data: pack_bitmask(&[false, false, false]),
                null_mask: None,
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr2);
            assert_eq!(unpack_bitmask(&arr2.data), vec64![false, false, false]);
        }

        #[test]
        fn test_sort_boolean_array_all_nulls_and_none() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[true, false, true]),
                null_mask: Some(pack_bitmask(&[false, false, false])),
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            assert_eq!(
                unpack_bitmask(arr.null_mask.as_ref().unwrap()),
                vec64![false, false, false]
            );
        }

        #[test]
        fn test_sort_slice_with_mask_basic() {
            let data = vec64![3, 1, 2, 1];
            let mask = pack_bitmask(&[true, false, true, true]);
            let (sorted, mask_out) = sort_slice_with_mask(&data, Some(&mask));
            assert_eq!(sorted.as_slice(), &[1, 1, 2, 3]);
            assert_eq!(
                unpack_bitmask(&mask_out.unwrap()),
                vec64![false, true, true, true]
            );
        }

        #[test]
        fn test_sort_slice_with_mask_no_mask() {
            let data = vec64![3, 2, 1, 1, 0];
            let (sorted, mask_out) = sort_slice_with_mask(&data, None);
            assert_eq!(sorted.as_slice(), &[0, 1, 1, 2, 3]);
            assert!(mask_out.is_none());
        }

        #[test]
        fn test_sort_slice_with_mask_all_true_and_all_false() {
            let data = vec64![3, 2, 1, 0];
            let mask_true = pack_bitmask(&[true; 4]);
            let mask_false = pack_bitmask(&[false; 4]);
            let (_, mask_true_out) = sort_slice_with_mask(&data, Some(&mask_true));
            let (_, mask_false_out) = sort_slice_with_mask(&data, Some(&mask_false));
            assert_eq!(
                unpack_bitmask(&mask_true_out.unwrap()),
                vec64![true, true, true, true]
            );
            assert_eq!(
                unpack_bitmask(&mask_false_out.unwrap()),
                vec64![false, false, false, false]
            );
        }

        #[test]
        fn test_sort_int_with_duplicates_and_negatives() {
            let mut arr = vec64![-10, -1, 5, 0, 5, -10];
            sort_int(&mut arr);
            assert_eq!(*arr, [-10, -10, -1, 0, 5, 5]);
        }

        #[test]
        fn test_sort_str_empty_and_duplicate() {
            let mut arr = vec64!["", "a", "b", "a", ""];
            sort_str(&mut arr);
            assert_eq!(*arr, ["", "", "a", "a", "b"]);
        }
    }
}
