use crate::base_matrix::{binary_matrix_fmt, slow_transpose, BinaryMatrix};
use crate::binary_dense_vector::{BinaryDenseVector, BITS};
#[cfg(feature = "rand")]
use rand::Rng;
use std::fmt;
use std::fmt::{Debug, Formatter, Write};
use std::ops;

/// A dense, binary matrix implementation by packing bits
/// into u64 elements. Column-oriented.
#[derive(Clone, PartialEq)]
pub struct BinaryMatrix64 {
    nrows: usize,
    columns: Vec<Vec<u64>>,
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
            return new as Box<dyn BinaryMatrix>;
        }
        slow_transpose(self, new.as_mut());
        new as Box<dyn BinaryMatrix>
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
    /// Based on Hacker's Delight (1st edition), Figure 7-3.
    fn transpose(&mut self) {
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
            slow_transpose(mat.as_ref(), &mut newmat);
            assert_eq!(format!("{:?}", mat.transpose()), format!("{:?}", newmat));
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
        assert_eq!(
            format!("{:?}", mat.transpose()),
            format!("{:?}", BinaryMatrix64::zero(3, 2))
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_transpose_64x64() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut newmat = BinaryMatrix64::new();
        slow_transpose(mat.as_ref(), &mut newmat);
        assert_eq!(format!("{:?}", mat.transpose()), format!("{:?}", newmat));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_transpose_128x64() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(128, 64, &mut rng);
        assert_eq!(
            format!("{:?}", mat),
            format!("{:?}", mat.transpose().transpose())
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_large_transpose() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(1000, 600, &mut rng);
        assert_eq!(format!("{:?}", mat), format!("{:?}", mat.transpose().transpose()));
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use crate::{BinaryMatrix, BinaryMatrix64};
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose_64x64(b: &mut Bencher) {
        let mat = BinaryMatrix64::identity(64);
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
}
