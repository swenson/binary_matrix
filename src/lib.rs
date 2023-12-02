#![feature(portable_simd)]
#![feature(test)]

//! Implementations of binary (GF(2)) matrices.
//!
//! This is a work in progress.
//!

extern crate core;

mod base_matrix;
mod binary_dense_matrix;
#[cfg(feature = "simd")]
mod binary_dense_matrix_simd;
mod binary_dense_vector;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod transpose64x64_asm_aarch;
mod transpose64x64_unroll;

pub use crate::base_matrix::*;
pub use crate::binary_dense_matrix::*;
#[cfg(feature = "simd")]
pub use crate::binary_dense_matrix_simd::*;
