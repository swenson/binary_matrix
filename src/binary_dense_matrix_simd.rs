use crate::base_matrix::{binary_matrix_fmt, BinaryMatrix};
use crate::binary_dense_vector::{BinaryDenseVector, BITS};
use crate::BinaryMatrix64;
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
    pub(crate) columns: Vec<Vec<Simd<u64, LANES>>>,
}

impl<const LANES: usize> BinaryMatrixSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Returns a new, empty matrix with zero rows and columns.
    pub fn new() -> Box<BinaryMatrixSimd<LANES>> {
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

    pub fn as_nonsimd(&self) -> Box<BinaryMatrix64> {
        let mut mat = BinaryMatrix64::zero(self.nrows, self.ncols());
        for c in 0..self.ncols() {
            for j in 0..self.columns[0].len() {
                for l in 0..LANES {
                    if j * LANES + l >= mat.columns[c].len() {
                        break;
                    }
                    mat.columns[c][j * LANES + l] = self.columns[c][j][l];
                }
            }
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

    // TODO: use those nice SIMD instructions somehow to do this faster
    fn transpose(&self) -> Box<dyn BinaryMatrix> {
        self.as_nonsimd().transpose64() as Box<dyn BinaryMatrix>
    }

    fn get(&self, r: usize, c: usize) -> u8 {
        assert!(r < self.nrows);
        let x = self.columns[c][r >> self.simd_bits()]
            [(r >> u64::BITS.trailing_zeros()) & self.simd_lane_mask()];
        let shift = self.simd_base_mask() - (r & self.simd_base_mask());
        ((x >> shift) & 1) as u8
    }

    fn set(&mut self, r: usize, c: usize, val: u8) {
        assert!(r < self.nrows);
        let shift = self.simd_base_mask() - (r & self.simd_base_mask());
        let row_idx = r >> self.simd_bits(); // >> 12
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

    /// Returns true if the given column is zero between 0 < maxr.
    fn column_part_all_zero(&self, c: usize, maxr: usize) -> bool {
        let zero_simd = self.zero_simd();
        for x in 0..maxr/64/LANES {
            if self.columns[c][x].ne(&zero_simd) {
                return false;
            }
        }
        for x in (maxr/64/LANES)*64*LANES..maxr {
            if self.get(x, c) == 1 {
                return false;
            }
        }
        return true;
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
        assert!(!left_kernel[0].is_zero());
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
            &*mat.transpose(),
            &*(BinaryMatrixSimd::<64>::zero(3, 2) as Box<dyn BinaryMatrix>)
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_as_simd() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(129, 129, &mut rng);
        let matsimd = (&mat).as_simd::<2>();
        let mat2 = (&matsimd).as_nonsimd();
        assert_eq!(&*mat as &dyn BinaryMatrix, &*matsimd);
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_column_part_zero() {
        let mut mat = BinaryMatrixSimd::<2>::zero(200, 4);
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
    use crate::{BinaryMatrix, BinaryMatrixSimd};
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose_simd_64x64(b: &mut Bencher) {
        let mat = BinaryMatrixSimd::<64>::identity(64);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_simd_100x100(b: &mut Bencher) {
        let mat = BinaryMatrixSimd::<64>::identity(100);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_simd_1000x1000(b: &mut Bencher) {
        let mat = BinaryMatrixSimd::<64>::identity(1000);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_transpose_simd_10000x10000(b: &mut Bencher) {
        let mat = BinaryMatrixSimd::<64>::identity(10000);
        b.iter(|| {
            test::black_box(mat.transpose());
        });
    }

    #[bench]
    fn bench_simd_column_part_zero(b: &mut Bencher) {
        let mat = BinaryMatrixSimd::<64>::identity(10000);
        b.iter(|| {
            test::black_box(mat.column_part_all_zero(9999, 9999));
        });
    }

}
