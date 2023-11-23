use crate::base_matrix::BinaryMatrix;
use crate::binary_dense_vector::{BinaryDenseVector, BITS};
#[cfg(feature = "rand")]
use rand::Rng;
use std::fmt;
use std::fmt::{Debug, Write};
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
    pub fn new() -> Box<dyn BinaryMatrix> {
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

    // TODO: bit tilt algorithm?
    fn transpose(&self) -> Box<dyn BinaryMatrix> {
        let mut new = BinaryMatrix64::new();
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
        let x = self.columns[c][r >> 6];
        let shift = r & 0x3f;
        ((x >> shift) & 1) as u8
    }

    fn set(&mut self, r: usize, c: usize, val: u8) {
        assert!(r < self.nrows);
        let shift = r & 0x3f;
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
        f.write_char('[')?;
        for r in 0..self.nrows() {
            if r != 0 {
                f.write_str(", ")?;
            }
            f.write_char('[')?;
            for c in 0..self.ncols() {
                if c != 0 {
                    f.write_str(", ")?;
                }
                f.write_char(char::from(48 + self[(r, c)]))?;
            }
            f.write_char(']')?;
        }
        f.write_char(']')
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::binary_dense_vector::BinaryDenseVector;
    use rand::prelude::*;
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
    fn test_large_left_kernel() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(1024, 1024, &mut rng);
        let left_kernel = mat.left_kernel().unwrap();
        assert_eq!(1, left_kernel.len());
        println!("{:?}", &left_kernel[0] * mat.as_ref());
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
}
