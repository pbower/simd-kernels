// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! # **Error Types** - *Kernel Operation Error Handling*
//!
//! Error types for kernel operations with structured error reporting.
//! Provides context for debugging and error recovery in computational pipelines.
//!
//! ## Error Categories  
//! - **Type Errors**: Mismatched or unsupported data types
//! - **Dimension Errors**: Array length and capacity mismatches
//! - **Operator Errors**: Invalid operations for given operands
//! - **Boundary Errors**: Out-of-bounds access and divide-by-zero conditions
//! - **Planning Errors**: Configuration and setup failures
//!
//! All errors include contextual message space for debugging. 

use core::fmt;
use std::error::Error;

/// Comprehensive error type for all kernel operations.
///
/// Each variant includes a contextual message string providing specific details
/// about the error condition, enabling precise debugging and error reporting.
#[derive(Debug, Clone)]
pub enum KernelError {
    /// Data type mismatch between operands or unsupported type combinations.
    TypeMismatch(String),
    
    /// Array length mismatch between operands.
    LengthMismatch(String),
    
    /// Invalid operator for the given operands or context.
    OperatorMismatch(String),
    
    /// Unsupported data type for the requested operation.
    UnsupportedType(String),
    
    /// Column or field not found in structured data.
    ColumnNotFound(String),
    
    /// Invalid arguments provided to kernel function.
    InvalidArguments(String),
    
    /// Planning or configuration error.
    Plan(String),
    
    /// Array index or memory access out of bounds.
    OutOfBounds(String),
    
    /// Division by zero or similar mathematical errors.
    DivideByZero(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            KernelError::LengthMismatch(msg) => write!(f, "Length mismatch: {}", msg),
            KernelError::OperatorMismatch(msg) => write!(f, "Operator mismatch: {}", msg),
            KernelError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            KernelError::ColumnNotFound(msg) => write!(f, "Column not found: {}", msg),
            KernelError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            KernelError::Plan(msg) => write!(f, "Planning error: {}", msg),
            KernelError::OutOfBounds(msg) => write!(f, "Out of bounds: {}", msg),
            KernelError::DivideByZero(msg) => write!(f, "Divide by Zero error: {}", msg),
        }
    }
}

impl Error for KernelError {}

/// Creates a formatted error message for length mismatches between left-hand side (LHS) and right-hand side (RHS) arrays.
///
/// # Arguments
/// * `fname` - Function name where the mismatch occurred
/// * `lhs` - Length of the left-hand side array
/// * `rhs` - Length of the right-hand side array
///
/// # Returns
/// A formatted error message string
pub fn log_length_mismatch(fname: String, lhs: usize, rhs: usize) -> String {
    return format!("{} => Length mismatch: LHS {} RHS {}", fname, lhs, rhs);
}
