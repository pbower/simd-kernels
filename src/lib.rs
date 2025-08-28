// Copyright 2025 Peter Garfield Bower
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![feature(portable_simd)]

// compile with RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features portable_simd

// Re-export minarrow's modules and top-level items so that internal `crate::*`
// paths in the kernel source files resolve to the upstream definitions.
pub use minarrow::*;
pub use minarrow::{aliases, conversions, enums, ffi, macros, structs, traits};

pub mod kernels {
    pub mod arithmetic;
    pub mod bitmask;
    pub mod string;
}

pub mod utils;