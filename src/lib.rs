#![feature(portable_simd)]

//! Implementations of binary (GF(2)) matrices.
//!
//! This is a work in progress.
//!

mod base_matrix;
mod binary_dense_matrix;
#[cfg(feature = "simd")]
mod binary_dense_matrix_simd;
mod binary_dense_vector;

pub use crate::base_matrix::*;
pub use crate::binary_dense_matrix::*;
#[cfg(feature = "simd")]
pub use crate::binary_dense_matrix_simd::*;
