#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(all(test, feature = "bench"), feature(test))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

//! Implementations of binary (GF(2)) matrices.
//!
//! This is a work in progress.
//!
//! ```
//! use binary_matrix::{BinaryDenseVector, BinaryMatrix, BinaryMatrix64};
//!
//! // [[0, 0, 0, 1],
//! //  [1, 1, 1, 0],
//! //  [1, 1, 1, 1]]
//! let mut m = BinaryMatrix64::zero(3, 4);
//! m.set(0, 3, 1);
//! for c in 0..3 {
//!     m.set(1, c, 1);
//!     m.set(2, c, 1);
//! }
//! m.set(2, 3, 1);
//!
//! let left_kernel = m.left_kernel().unwrap();
//! assert_eq!(vec![BinaryDenseVector::from_bits(&[1, 1, 1])], left_kernel);
//! assert!((&left_kernel[0] * m).is_zero());
//! ```
//!
//! # Features
//!
//! - `rand`: random matrices and vectors.
//! - `simd`: `BinaryMatrixSimd`, built on `std::simd`. Requires nightly.
//! - `bench`: `#[bench]` benchmarks. Requires nightly.

extern crate core;

mod base_matrix;
mod binary_dense_matrix;
#[cfg(feature = "simd")]
mod binary_dense_matrix_simd;
mod binary_dense_vector;
#[cfg(test)]
mod matrix_tests;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod transpose64x64_asm_aarch;
mod transpose64x64_unroll;

pub use crate::base_matrix::*;
pub use crate::binary_dense_matrix::*;
#[cfg(feature = "simd")]
pub use crate::binary_dense_matrix_simd::*;
pub use crate::binary_dense_vector::BinaryDenseVector;
