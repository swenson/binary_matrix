use crate::base_matrix::{binary_matrix_fmt, BinaryMatrix};
use crate::binary_dense_vector::{BinaryDenseVector, BITS};
#[cfg(feature = "rand")]
use rand::Rng;
use std::fmt;
use std::fmt::Debug;
use std::ops;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

/// A dense, binary matrix implementation by packing bits
/// into simd arrays of u64 of size LANES. Column-oriented.
#[derive(Clone)]
pub struct BinaryMatrixSimd<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    nrows: usize,
    columns: Vec<Vec<Simd<u64, LANES>>>,
}

impl<const LANES: usize> BinaryMatrixSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Returns a new, empty matrix with zero rows and columns.
    pub fn new() -> Box<dyn BinaryMatrix> {
        Box::new(BinaryMatrixSimd {
            nrows: 0,
            columns: vec![],
        })
    }

    /// Returns a new, zero matrix the given number of rows and columns.
    pub fn zero(rows: usize, cols: usize) -> Box<BinaryMatrixSimd<LANES>> {
        let mut mat = Box::new(BinaryMatrixSimd {
            nrows: 0,
            columns: vec![],
        });
        mat.expand(rows, cols);
        mat
    }

    /// Returns a new, square matrix the given number of rows and the diagonals set to 1.
    pub fn identity(rows: usize) -> Box<BinaryMatrixSimd<LANES>> {
        let mut mat = BinaryMatrixSimd::zero(rows, rows);
        for i in 0..rows {
            mat.set(i, i, 1);
        }
        mat
    }

    /// Returns a new random matrix of the given size.
    #[cfg(feature = "rand")]
    pub fn random<R: Rng + ?Sized>(
        rows: usize,
        cols: usize,
        rng: &mut R,
    ) -> Box<BinaryMatrixSimd<LANES>> {
        let mut mat = BinaryMatrixSimd::zero(rows, cols);
        for c in 0..cols {
            for r in 0..rows {
                if rng.gen() {
                    mat.set(r, c, 1);
                }
            }
        }
        mat
    }

    #[inline(always)]
    const fn simd_bits(&self) -> usize {
        (LANES.trailing_zeros() + u64::BITS.trailing_zeros()) as usize
    }

    #[inline(always)]
    const fn zero_simd(&self) -> Simd<u64, LANES> {
        Simd::from_array([0; LANES])
    }

    #[inline(always)]
    const fn simd_lane_mask(&self) -> usize {
        LANES - 1
    }
    #[inline(always)]
    const fn simd_base_mask(&self) -> usize {
        (u64::BITS - 1) as usize
    }
}

impl<const LANES: usize> BinaryMatrix for BinaryMatrixSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
    fn expand(&mut self, new_rows: usize, new_cols: usize) {
        let simd_bits = self.simd_bits();
        let zero_simd = self.zero_simd();
        self.nrows += new_rows;
        for c in 0..self.columns.len() {
            self.columns[c].resize((self.nrows >> simd_bits) + 1, zero_simd);
        }
        for _ in 0..new_cols {
            let mut col = Vec::with_capacity((self.nrows >> simd_bits) + 1);
            col.resize((self.nrows >> simd_bits) + 1, zero_simd);
            self.columns.push(col);
        }
    }

    // TODO: bit tilt algorithm?
    fn transpose(&self) -> Box<dyn BinaryMatrix> {
        let mut new = BinaryMatrixSimd::new();
        new.expand(self.ncols(), self.nrows());
        for c in 0..self.columns.len() {
            for r in 0..self.nrows {
                new.set(c, r, self[(r, c)]);
            }
        }
        new
    }

    fn get(&self, r: usize, c: usize) -> u8 {
        assert!(r < self.nrows);
        let x = self.columns[c][r >> self.simd_bits()].as_array()
            [(r >> u64::BITS.trailing_zeros()) & self.simd_lane_mask()];
        let shift = r & self.simd_base_mask();
        ((x >> shift) & 1) as u8
    }

    fn set(&mut self, r: usize, c: usize, val: u8) {
        assert!(r < self.nrows);
        let shift = r & self.simd_base_mask();
        let row_idx = r >> self.simd_bits();
        let lane_idx = (r >> u64::BITS.trailing_zeros()) & self.simd_lane_mask();
        if val == 1 {
            self.columns[c][row_idx].as_mut_array()[lane_idx] |= 1 << shift;
        } else {
            self.columns[c][row_idx].as_mut_array()[lane_idx] &= !(1 << shift);
        }
    }

    fn copy(&self) -> Box<dyn BinaryMatrix> {
        let mut cols = vec![];
        for c in 0..self.columns.len() {
            cols.push(self.columns[c].clone());
        }
        Box::new(BinaryMatrixSimd {
            nrows: self.nrows,
            columns: cols,
        })
    }

    fn swap_columns(&mut self, c1: usize, c2: usize) -> () {
        self.columns.swap(c1, c2);
    }

    fn xor_col(&mut self, c1: usize, c2: usize) -> () {
        assert!(c1 < self.columns.len());
        assert!(c2 < self.columns.len());
        let maxc = self.columns[c1].len();
        unsafe {
            let mut x1 = self.columns[c1].as_mut_ptr();
            let mut x2 = self.columns[c2].as_ptr();
            for _ in 0..maxc {
                //self.columns[c1][i] ^= self.columns[c2][i];
                *x1 ^= *x2;
                x1 = x1.offset(1);
                x2 = x2.offset(1);
            }
        }
    }
}

impl<const LANES: usize> ops::Index<(usize, usize)> for &BinaryMatrixSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &BITS[self.get(index.0, index.1) as usize]
    }
}

impl<const LANES: usize> Debug for BinaryMatrixSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        binary_matrix_fmt(self, f)
    }
}

impl<const LANES: usize> ops::Mul<Box<BinaryMatrixSimd<LANES>>> for &BinaryDenseVector
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = BinaryDenseVector;

    fn mul(self, rhs: Box<BinaryMatrixSimd<LANES>>) -> Self::Output {
        self * rhs.as_ref()
    }
}

impl<const LANES: usize> ops::Mul<&BinaryMatrixSimd<LANES>> for &BinaryDenseVector
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &BinaryMatrixSimd<LANES>) -> Self::Output {
        self * (rhs as &dyn BinaryMatrix)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::binary_dense_vector::BinaryDenseVector;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_transpose() {
        let mut mat = BinaryMatrixSimd::<64>::new();
        mat.expand(2, 3);
        mat.transpose();
    }

    #[test]
    fn test_left_kernel() {
        let mut mat = BinaryMatrixSimd::<64>::new();
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
    fn test_large_left_kernel() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrixSimd::<64>::random(1024, 1024, &mut rng);
        let left_kernel = mat.left_kernel().unwrap();
        assert_eq!(1, left_kernel.len());
        println!("{:?}", &left_kernel[0] * mat.as_ref());
        assert!((&left_kernel[0] * mat).is_zero());
    }

    #[test]
    fn test_format() {
        let mut mat = BinaryMatrixSimd::<64>::zero(5, 5);
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
        let mat = BinaryMatrixSimd::<64>::zero(2, 3);
        assert_eq!(
            format!("{:?}", mat.transpose()),
            format!("{:?}", BinaryMatrixSimd::<64>::zero(3, 2))
        );
    }
}
