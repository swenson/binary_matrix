use crate::base_matrix::{binary_matrix_fmt, slow_transpose, BinaryMatrix};
use crate::binary_dense_vector::{BinaryDenseVector, BITS};
use crate::transpose64x64_unroll::transpose_unroll_64x64;
#[cfg(feature = "simd")]
use crate::BinaryMatrixSimd;
#[cfg(feature = "rand")]
use rand::Rng;
use std::arch::{aarch64, asm};
use std::fmt;
use std::fmt::{Debug, Formatter, Write};
#[cfg(feature = "simd")]
use std::mem::transmute;
use std::ops;
#[cfg(feature = "simd")]
use std::simd::{u16x16, u32x32, u64x64};
use std::simd::{u8x8, Mask, Simd};
#[cfg(feature = "simd")]
use std::simd::{LaneCount, SupportedLaneCount};

/// A dense, binary matrix implementation by packing bits
/// into u64 elements. Column-oriented.
#[derive(Clone, PartialEq)]
pub struct BinaryMatrix64 {
    nrows: usize,
    pub(crate) columns: Vec<Vec<u64>>,
}

impl BinaryMatrix64 {
    /// Returns a new, empty matrix with zero rows and columns.
    pub fn new() -> Box<BinaryMatrix64> {
        Box::new(BinaryMatrix64 {
            nrows: 0,
            columns: vec![],
        })
    }

    /// Returns a new, zero matrix the given number of rows and columns.
    pub fn zero(rows: usize, cols: usize) -> Box<BinaryMatrix64> {
        let mut mat = Box::new(BinaryMatrix64 {
            nrows: 0,
            columns: vec![],
        });
        mat.expand(rows, cols);
        mat
    }

    /// Returns a new, square matrix the given number of rows and the diagonals set to 1.
    pub fn identity(rows: usize) -> Box<BinaryMatrix64> {
        let mut mat = BinaryMatrix64::zero(rows, rows);
        for i in 0..rows {
            mat.set(i, i, 1);
        }
        mat
    }

    // TODO: investigate using rotate instructions instead of swap/shifts
    // TODO: investigate SIMD instructions to speed this up
    pub(crate) fn transpose64(&self) -> Box<BinaryMatrix64> {
        let mut new = BinaryMatrix64::new();
        if self.nrows >= 64 && self.columns.len() >= 64 {
            new.expand(self.columns.len(), self.nrows);
            let n = (self.nrows & 0xffffffffffffffc0).min(self.columns.len() & 0xffffffffffffffc0);
            for i in (0..n).step_by(64) {
                for j in (i..n).step_by(64) {
                    if i == j {
                        let a = self.subtranspose(i, i);
                        new.write_submatrix(i, i, &a);
                    } else {
                        let a = self.subtranspose(i, j);
                        let b = self.subtranspose(j, i);
                        new.write_submatrix(i, j, &b);
                        new.write_submatrix(j, i, &a);
                    }
                }
            }
            // slow way for the leftovers
            for c in n..self.columns.len() {
                for r in 0..self.nrows {
                    new.set(c, r, self.get(r, c));
                }
            }
            for r in n..self.nrows {
                for c in 0..self.columns.len() {
                    new.set(c, r, self.get(r, c));
                }
            }
            return new;
        }
        slow_transpose(self, new.as_mut());
        new
    }

    #[cfg(feature = "simd")]
    pub fn as_simd<const LANES: usize>(&self) -> Box<BinaryMatrixSimd<LANES>>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let mut mat = BinaryMatrixSimd::zero(self.nrows, self.ncols());
        for c in 0..self.ncols() {
            let mut j = 0;
            let mut l = 0;
            for i in 0..self.columns[0].len() {
                mat.columns[c][j][l] = self.columns[c][i];
                l += 1;
                if l == LANES {
                    l = 0;
                    j += 1;
                }
            }
        }
        mat
    }

    /// Returns a new random matrix of the given size.
    #[cfg(feature = "rand")]
    pub fn random<R: Rng + ?Sized>(rows: usize, cols: usize, rng: &mut R) -> Box<BinaryMatrix64> {
        let mut mat = BinaryMatrix64::zero(rows, cols);
        for c in 0..cols {
            for r in 0..rows {
                if rng.gen() {
                    mat.set(r, c, 1);
                }
            }
        }
        mat
    }

    /// extract 64x64 submatrix and transpose
    fn subtranspose(&self, c: usize, r: usize) -> BinaryMatrix64x64 {
        assert_eq!(r % 64, 0);
        assert_eq!(c % 64, 0);

        let mut cols = [0u64; 64];
        for i in 0..64 {
            cols[i] = self.columns[c + i][r >> 6];
        }
        let mut m = BinaryMatrix64x64 { cols };
        m.transpose();
        m
    }

    fn write_submatrix(&mut self, c: usize, r: usize, mat: &BinaryMatrix64x64) {
        assert_eq!(r % 64, 0);
        for i in 0..64 {
            self.columns[c + i][r >> 6] = mat.cols[i];
        }
    }
}

impl BinaryMatrix for BinaryMatrix64 {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
    fn expand(&mut self, new_rows: usize, new_cols: usize) {
        self.nrows += new_rows;
        for c in 0..self.columns.len() {
            self.columns[c].resize((self.nrows >> 6) + 1, 0u64);
        }
        for _ in 0..new_cols {
            let mut col = Vec::with_capacity((self.nrows >> 6) + 1);
            col.resize((self.nrows >> 6) + 1, 0u64);
            self.columns.push(col);
        }
    }

    fn transpose(&self) -> Box<dyn BinaryMatrix> {
        self.transpose64() as Box<dyn BinaryMatrix>
    }

    fn get(&self, r: usize, c: usize) -> u8 {
        assert!(r < self.nrows);
        let x = self.columns[c][r >> 6];
        let shift = 0x3f - (r & 0x3f);
        ((x >> shift) & 1) as u8
    }

    fn set(&mut self, r: usize, c: usize, val: u8) {
        assert!(r < self.nrows);
        let shift = 0x3f - (r & 0x3f);
        if val == 1 {
            self.columns[c][r >> 6] |= 1 << shift;
        } else {
            self.columns[c][r >> 6] &= !(1 << shift);
        }
    }

    fn copy(&self) -> Box<dyn BinaryMatrix> {
        let mut cols = vec![];
        for c in 0..self.columns.len() {
            cols.push(self.columns[c].clone());
        }
        Box::new(BinaryMatrix64 {
            nrows: self.nrows,
            columns: cols,
        })
    }

    fn col(&self, c: usize) -> BinaryDenseVector {
        BinaryDenseVector {
            size: self.nrows,
            bits: self.columns[c].clone(),
        }
    }

    fn swap_columns(&mut self, c1: usize, c2: usize) -> () {
        self.columns.swap(c1, c2);
    }

    fn xor_col(&mut self, c1: usize, c2: usize) -> () {
        assert!(c1 < self.columns.len());
        assert!(c2 < self.columns.len());
        let maxc = self.columns[c1].len();
        unsafe {
            let x1 = self.columns[c1].as_mut_ptr();
            let x2 = self.columns[c2].as_ptr();
            for c in 0..maxc as isize {
                *x1.offset(c) ^= *x2.offset(c);
            }
        }
    }

    /// Returns true if the given column is zero between 0 < maxr.
    fn column_part_all_zero(&self, c: usize, maxr: usize) -> bool {
        for x in 0..maxr / 64 {
            if self.columns[c][x] != 0 {
                return false;
            }
        }
        for x in (maxr / 64) * 64..maxr {
            if self.get(x, c) == 1 {
                return false;
            }
        }
        return true;
    }
}

impl Debug for BinaryMatrix64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        binary_matrix_fmt(self, f)
    }
}

impl ops::Index<(usize, usize)> for &BinaryMatrix64 {
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &BITS[self.get(index.0, index.1) as usize]
    }
}

impl ops::Mul<Box<BinaryMatrix64>> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: Box<BinaryMatrix64>) -> Self::Output {
        self * rhs.as_ref()
    }
}

impl ops::Mul<&BinaryMatrix64> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &BinaryMatrix64) -> Self::Output {
        self * (rhs as &dyn BinaryMatrix)
    }
}

struct BinaryMatrix64x64 {
    cols: [u64; 64],
}

impl Debug for BinaryMatrix64x64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('[')?;
        for r in 0..64 {
            if r != 0 {
                f.write_str(", ")?;
            }
            f.write_char('[')?;
            for c in 0..64 {
                if c != 0 {
                    f.write_str(", ")?;
                }
                f.write_char(char::from(48 + (1 & (self.cols[c] >> r)) as u8))?;
            }
            f.write_char(']')?;
        }
        f.write_char(']')
    }
}

impl BinaryMatrix64x64 {
    fn transpose(&mut self) {
        self.transpose_unroll();
        //self.transpose_plain();
    }

    /// Based on Hacker's Delight (1st edition), Figure 7-3.
    fn transpose_plain(&mut self) {
        let mut j = 32;
        let mut m = 0xffffffffu64;
        while j != 0 {
            let mut k = 0;
            while k < 64 {
                unsafe {
                    let a = *self.cols.get_unchecked(k);
                    let b = *self.cols.get_unchecked(k + j);
                    let t = (a ^ (b >> j)) & m;
                    *self.cols.get_unchecked_mut(k) = a ^ t;
                    *self.cols.get_unchecked_mut(k + j) = b ^ (t << j);
                }

                k = (k + j + 1) & !j;
            }
            j >>= 1;
            m ^= m << j;
        }
    }

    //#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    //fn transpose_intrinsics(&mut self) {
    fn transpose_unroll(&mut self) {
        transpose_unroll_64x64(&mut self.cols);
        // unsafe {
        //     // let a0 = aarch64::vld4_u64(m);
        //     // let a1 = aarch64::vld4_u64(m.offset(4));
        //     // let a2 = aarch64::vld4_u64(m.offset(8));
        //     // let a3 = aarch64::vld4_u64(m.offset(12));
        //
        //     // let j = 32;
        //     // let m = 0xffffffffu64;
        //     //
        //
        // }
    }

    // #[cfg(feature = "simd")]
    // fn transpose(&mut self) {
    //     self.cols = *transpose_simd_64(u64x64::from_array(self.cols)).as_array()
    // }
}

#[cfg(feature = "simd")]
fn transpose_simd_64(x: u64x64) -> u64x64 {
    let mask32: Mask<isize, 32> = Mask::splat(true);
    let idx0_128: Simd<usize, 32> = Simd::<usize, 32>::from_array([
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
        48, 50, 52, 54, 56, 58, 60, 62,
    ]);
    let idx1_128: Simd<usize, 32> = Simd::<usize, 32>::from_array([
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
        49, 51, 53, 55, 57, 59, 61, 63,
    ]);
    let idx2_128: Simd<usize, 32> = Simd::<usize, 32>::from_array([
        64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106,
        108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
    ]);
    let idx3_128: Simd<usize, 32> = Simd::<usize, 32>::from_array([
        65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107,
        109, 111, 113, 115, 117, 119, 121, 123, 125, 127,
    ]);
    let zero: u32x32 = u32x32::splat(0);
    let arr0 = unsafe { transmute::<u64x64, [u32; 128]>(x) };
    let arr = arr0.as_slice();
    let y0 = u32x32::gather_select(arr, mask32, idx0_128, zero);
    let y1 = u32x32::gather_select(arr, mask32, idx1_128, zero);
    let y2 = u32x32::gather_select(arr, mask32, idx2_128, zero);
    let y3 = u32x32::gather_select(arr, mask32, idx3_128, zero);
    let z0 = transpose_simd_32(y0);
    let z1 = transpose_simd_32(y1);
    let z2 = transpose_simd_32(y2);
    let z3 = transpose_simd_32(y3);

    let mut out = [0u32; 128];
    z0.scatter(out.as_mut_slice(), idx0_128);
    // the part of the transpose here
    z2.scatter(out.as_mut_slice(), idx1_128);
    z1.scatter(out.as_mut_slice(), idx2_128);
    z3.scatter(out.as_mut_slice(), idx3_128);
    unsafe { transmute::<[u32; 128], u64x64>(out) }
}

#[cfg(feature = "simd")]
#[inline(always)]
fn transpose_simd_32(x: u32x32) -> u32x32 {
    let mask16: Mask<isize, 16> = Mask::splat(true);
    let idx0_64: Simd<usize, 16> =
        Simd::<usize, 16>::from_array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    let idx1_64: Simd<usize, 16> =
        Simd::<usize, 16>::from_array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]);
    let idx2_64: Simd<usize, 16> = Simd::<usize, 16>::from_array([
        32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
    ]);
    let idx3_64: Simd<usize, 16> = Simd::<usize, 16>::from_array([
        33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    ]);
    let zero: u16x16 = u16x16::splat(0);
    let arr0 = unsafe { transmute::<u32x32, [u16; 64]>(x) };
    let arr = arr0.as_slice();
    let y0 = u16x16::gather_select(arr, mask16, idx0_64, zero);
    let y1 = u16x16::gather_select(arr, mask16, idx1_64, zero);
    let y2 = u16x16::gather_select(arr, mask16, idx2_64, zero);
    let y3 = u16x16::gather_select(arr, mask16, idx3_64, zero);
    let z0 = transpose_simd_16(y0);
    let z1 = transpose_simd_16(y1);
    let z2 = transpose_simd_16(y2);
    let z3 = transpose_simd_16(y3);

    let mut out = [0u16; 64];
    z0.scatter(out.as_mut_slice(), idx0_64);
    // the part of the transpose here
    z2.scatter(out.as_mut_slice(), idx1_64);
    z1.scatter(out.as_mut_slice(), idx2_64);
    z3.scatter(out.as_mut_slice(), idx3_64);
    unsafe { transmute::<[u16; 64], u32x32>(out) }
}

#[cfg(feature = "simd")]
#[inline(always)]
fn transpose_simd_16(x: u16x16) -> u16x16 {
    let mask8: Mask<isize, 8> = Mask::splat(true);
    let idx0_32: Simd<usize, 8> = Simd::<usize, 8>::from_array([0, 2, 4, 6, 8, 10, 12, 14]);
    let idx1_32: Simd<usize, 8> = Simd::<usize, 8>::from_array([1, 3, 5, 7, 9, 11, 13, 15]);
    let idx2_32: Simd<usize, 8> = Simd::<usize, 8>::from_array([16, 18, 20, 22, 24, 26, 28, 30]);
    let idx3_32: Simd<usize, 8> = Simd::<usize, 8>::from_array([17, 19, 21, 23, 25, 27, 29, 31]);
    let zero: u8x8 = u8x8::splat(0);
    let arr0 = unsafe { transmute::<u16x16, [u8; 32]>(x) };
    let arr = arr0.as_slice();
    let y0 = u8x8::gather_select(arr, mask8, idx0_32, zero);
    let y1 = u8x8::gather_select(arr, mask8, idx1_32, zero);
    let y2 = u8x8::gather_select(arr, mask8, idx2_32, zero);
    let y3 = u8x8::gather_select(arr, mask8, idx3_32, zero);
    let z0 = unsafe { transmute::<u64, u8x8>(transpose_simd_8(transmute::<u8x8, u64>(y0))) };
    let z1 = unsafe { transmute::<u64, u8x8>(transpose_simd_8(transmute::<u8x8, u64>(y1))) };
    let z2 = unsafe { transmute::<u64, u8x8>(transpose_simd_8(transmute::<u8x8, u64>(y2))) };
    let z3 = unsafe { transmute::<u64, u8x8>(transpose_simd_8(transmute::<u8x8, u64>(y3))) };

    let mut out = [0u8; 32];
    z0.scatter(out.as_mut_slice(), idx0_32);
    // the part of the transpose here
    z2.scatter(out.as_mut_slice(), idx1_32);
    z1.scatter(out.as_mut_slice(), idx2_32);
    z3.scatter(out.as_mut_slice(), idx3_32);
    unsafe { transmute::<[u8; 32], u16x16>(out) }
}

#[cfg(feature = "simd")]
#[inline(always)]
fn transpose_simd_8(a: u64) -> u64 {
    // Based on Hacker's Delight (1st edition), Figure 7-2.
    let mut x = (a >> 32) as u32;
    let mut y = (a & 0xffffffff) as u32;
    let mut t;
    t = (x ^ (x >> 7)) & 0x00AA00AA;
    x ^= t ^ (t << 7);
    t = (y ^ (y >> 7)) & 0x00AA00AA;
    y ^= t ^ (t << 7);

    t = (x ^ (x >> 14)) & 0x0000CCCC;
    x ^= t ^ (t << 14);
    t = (y ^ (y >> 14)) & 0x0000CCCC;
    y ^= t ^ (t << 14);

    t = (x & 0xf0f0f0f0) | ((y >> 4) & 0x0f0f0f0f);
    y = ((x << 4) & 0xf0f0f0f0) | (y & 0x0f0f0f0f);
    x = t;

    ((x as u64) << 32) | (y as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_dense_vector::BinaryDenseVector;
    #[cfg(feature = "rand")]
    use rand::prelude::*;
    #[cfg(feature = "rand")]
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_left_kernel() {
        let mut mat = BinaryMatrix64::new();
        assert_eq!(Some(vec![]), mat.left_kernel());

        mat.expand(3, 4);
        mat.set(0, 3, 1);
        mat.set(1, 0, 1);
        mat.set(1, 1, 1);
        mat.set(1, 2, 1);
        mat.set(2, 0, 1);
        mat.set(2, 1, 1);
        mat.set(2, 2, 1);
        mat.set(2, 3, 1);

        let left_kernel_option = mat.left_kernel();
        assert_eq!(
            Some(vec![BinaryDenseVector::from_bits(&[1, 1, 1])]),
            left_kernel_option
        );
        let left_kernel = left_kernel_option.unwrap();
        assert_eq!(1, left_kernel.len());
        assert!((&left_kernel[0] * mat).is_zero());
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_transposes() {
        for i in 1..128 {
            let mut rng = ChaCha8Rng::seed_from_u64(1234);
            let mat = BinaryMatrix64::random(i, i, &mut rng);
            let mut newmat = BinaryMatrix64::new();
            slow_transpose(&*mat, &mut newmat);
            assert_eq!(&*mat.transpose(), &*newmat);
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_large_left_kernel() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(66, 66, &mut rng);
        let left_kernel = mat.left_kernel().unwrap();
        assert_eq!(1, left_kernel.len());
        assert!((&left_kernel[0] * mat).is_zero());
    }

    #[test]
    fn test_format() {
        let mut mat = BinaryMatrix64::zero(5, 5);
        mat.set(0, 0, 1);
        mat.set(1, 1, 1);
        mat.set(3, 3, 1);
        assert_eq!(
            "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]",
            format!("{:?}", mat)
        );
    }

    #[test]
    fn test_transpose_empty() {
        let mat = BinaryMatrix64::zero(2, 3);
        assert_eq!(&*mat.transpose(), &*BinaryMatrix64::zero(3, 2));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_transpose_64x64() {
        //let mut rng = ChaCha8Rng::seed_from_u64(1234);
        //let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut mat = BinaryMatrix64::zero(64, 64);
        mat.set(0, 2, 1);
        let mut newmat = BinaryMatrix64::new();
        slow_transpose(mat.as_ref(), &mut newmat);
        assert_eq!(&*mat.transpose(), &*newmat);
    }

    #[test]
    #[cfg(feature = "rand")]
    #[cfg(feature = "simd")]
    fn test_transpose_8() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut m = 0u64;
        for c in 0..8 {
            for r in 0..8 {
                if mat.get(r, c) == 1 {
                    m |= 1 << ((7 - r) * 8 + 7 - c);
                }
            }
        }
        m = transpose_simd_8(m);
        let matt = mat.transpose();
        let mut n = 0u64;
        for c in 0..8 {
            for r in 0..8 {
                if matt.get(r, c) == 1 {
                    n |= 1 << ((7 - r) * 8 + 7 - c);
                }
            }
        }
        assert_eq!(m, n);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_transpose_8x8() {
        let m = 0x8040201008040201u64;
        assert_eq!(m, transpose_simd_8(m));
        let m = 0xffffffffffffffffu64;
        assert_eq!(m, transpose_simd_8(m));
        let m = 0x0100000000000000u64;
        assert_eq!(0x80, transpose_simd_8(m));
        let m = 0x0200000000000000u64;
        assert_eq!(0x8000, transpose_simd_8(m));
        let m = 0x0400000000000000u64;
        assert_eq!(0x800000, transpose_simd_8(m));
        let m = 0x0800000000000000u64;
        assert_eq!(0x80000000, transpose_simd_8(m));
        let m = 0x1000000000000000u64;
        assert_eq!(0x8000000000, transpose_simd_8(m));
        let m = 0x2000000000000000u64;
        assert_eq!(0x800000000000, transpose_simd_8(m));
        let m = 0x4000000000000000u64;
        assert_eq!(0x80000000000000, transpose_simd_8(m));

        let m = 0x0000000000000002u64;
        assert_eq!(0x0000000000000100, transpose_simd_8(m));
        let m = 0x0000000000000004u64;
        assert_eq!(0x0000000000010000, transpose_simd_8(m));
        let m = 0x0000000000000008u64;
        assert_eq!(0x0000000001000000, transpose_simd_8(m));
        let m = 0x0000000000000010u64;
        assert_eq!(0x0000000100000000, transpose_simd_8(m));
        let m = 0x0000000000000020u64;
        assert_eq!(0x0000010000000000, transpose_simd_8(m));
        let m = 0x0000000000000040u64;
        assert_eq!(0x0001000000000000, transpose_simd_8(m));
        let m = 0x0000000000000080u64;
        assert_eq!(0x0100000000000000u64, transpose_simd_8(m));
        let m = 0x0000000000000100u64;
        assert_eq!(0x0000000000000002u64, transpose_simd_8(m));
        let m = 0x0000000000000200u64;
        assert_eq!(m, transpose_simd_8(m));
        let m = 0x0000000000000400u64;
        assert_eq!(0x0000000000020000u64, transpose_simd_8(m));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_transpose_128x64() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(128, 64, &mut rng);
        let mattt = mat.transpose().transpose();
        assert_eq!(&*(mat as Box<dyn BinaryMatrix>), &*mattt);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_large_transpose() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(1000, 600, &mut rng);
        let mattt = mat.transpose().transpose();
        assert_eq!(&*(mat as Box<dyn BinaryMatrix>), &*mattt);
    }

    #[test]
    fn test_column_part_zero() {
        let mut mat = BinaryMatrix64::zero(200, 4);
        mat.set(199, 0, 1);
        mat.set(198, 1, 1);
        mat.set(0, 2, 1);
        mat.set(67, 3, 1);
        for i in 0..199 {
            assert!(mat.column_part_all_zero(0, i));
        }
        assert!(!mat.column_part_all_zero(0, 200));
        for i in 0..198 {
            assert!(mat.column_part_all_zero(1, i));
        }
        for i in 199..201 {
            assert!(!mat.column_part_all_zero(1, i));
        }
        assert!(mat.column_part_all_zero(2, 0));
        for i in 1..201 {
            assert!(!mat.column_part_all_zero(2, i));
        }
        for i in 0..68 {
            assert!(mat.column_part_all_zero(3, i));
        }
        for i in 69..201 {
            assert!(!mat.column_part_all_zero(3, i));
        }
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use crate::binary_dense_matrix::BinaryMatrix64x64;
    use crate::{BinaryMatrix, BinaryMatrix64};
    #[cfg(feature = "rand")]
    use rand::SeedableRng;
    #[cfg(feature = "rand")]
    use rand_chacha::ChaCha8Rng;
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose_64x64(b: &mut Bencher) {
        let mut mat = BinaryMatrix64::identity(64).subtranspose(0, 0);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_100x100(b: &mut Bencher) {
        let mat = BinaryMatrix64::identity(100);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_1000x1000(b: &mut Bencher) {
        let mat = BinaryMatrix64::identity(1000);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_10000x10000(b: &mut Bencher) {
        let mat = BinaryMatrix64::identity(10000);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_column_part_zero(b: &mut Bencher) {
        let mat = BinaryMatrix64::identity(10000);
        b.iter(|| {
            test::black_box(mat.column_part_all_zero(9999, 9999));
        });
    }

    #[bench]
    #[cfg(feature = "rand")]
    fn bench_left_kernel_1024_1024(b: &mut Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(1024, 1024, &mut rng);
        b.iter(|| {
            test::black_box(mat.left_kernel().unwrap());
        });
    }
}
