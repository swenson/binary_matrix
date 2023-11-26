use crate::binary_dense_vector::{BinaryDenseVector, BITS};
use std::fmt::Write;
use std::{fmt, ops};

pub(crate) fn binary_matrix_fmt(s: &dyn BinaryMatrix, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_char('[')?;
    for r in 0..s.nrows() {
        if r != 0 {
            f.write_str(", ")?;
        }
        f.write_char('[')?;
        for c in 0..s.ncols() {
            if c != 0 {
                f.write_str(", ")?;
            }
            f.write_char(char::from(48 + s.get(r, c)))?;
        }
        f.write_char(']')?;
    }
    f.write_char(']')
}

pub(crate) fn slow_transpose<T>(s: &T, new: &mut T)
where
    T: BinaryMatrix,
{
    new.expand(s.ncols(), s.nrows());
    for c in 0..s.ncols() {
        for r in 0..s.nrows() {
            new.set(c, r, s.get(r, c));
        }
    }
}

/// Trait for binary (GF(2)) matrices.
pub trait BinaryMatrix {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Adds more rows and columns to the matrix.
    fn expand(&mut self, new_rows: usize, new_cols: usize);
    /// Returns a new matrix that is the transpose of this matrix.
    fn transpose(&self) -> Box<dyn BinaryMatrix>;
    /// Gets the value of the matrix at the row r and column c.
    fn get(&self, r: usize, c: usize) -> u8;
    /// Sets the value of the matrix at the row r and column c to val.
    fn set(&mut self, r: usize, c: usize, val: u8);
    /// Returns a copy of this matrix.
    fn copy(&self) -> Box<dyn BinaryMatrix>;
    /// Compute the left kernel of the matrix.
    /// Uses using basic algorithm, using Gaussian elimination, from
    /// <https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination>
    fn left_kernel(&self) -> Option<Vec<BinaryDenseVector>> {
        self.transpose().kernel()
    }
    /// Compute the kernel (right nullspace) of the matrix.
    /// Uses the basic algorithm, using Gaussian elimination, from
    /// <https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination>
    fn kernel(&self) -> Option<Vec<BinaryDenseVector>> {
        // compute kernel of matrix over GF(2) using Gaussian elimination
        let mut extend = self.copy();
        // extend out with an identity matrix of size RxR
        let maxr = extend.nrows();
        let maxc = extend.ncols();
        extend.expand(maxc, 0);
        for c in 0..maxc {
            extend.set(maxr + c, c, 1);
        }

        // pivot on a row for each column
        let mut nextc = 0;
        for r in 0..maxr {
            let mut pivot = -1i32;
            for c in nextc..maxc {
                if extend.get(r, c) != 0 {
                    pivot = c as i32;
                    break;
                }
            }
            if pivot == -1 {
                continue;
            }

            extend.swap_columns(nextc, pivot as usize);

            for c in (nextc as usize + 1)..maxc {
                if extend.get(r, c) == 1 {
                    extend.xor_col(c, nextc);
                }
            }
            nextc += 1;
        }

        let mut kernel = vec![];

        for c in 0..maxc {
            if extend.column_part_all_zero(c, maxr) {
                let v = extend.extract_column_part(c, maxr, extend.nrows() - maxr);
                kernel.push(v);
            }
        }

        Some(kernel)
    }
    /// Multiplies the given matrix on the left by the dense vector.
    fn left_mul(&self, result_vector: &BinaryDenseVector) -> BinaryDenseVector {
        assert_eq!(result_vector.size, self.nrows());
        let mut result = Vec::with_capacity(self.ncols());
        for c in 0..self.ncols() {
            let mut bit = 0;
            for r in 0..self.nrows() {
                bit ^= self.get(r, c) * result_vector[r];
            }
            result.push(bit);
        }
        BinaryDenseVector::from_bits(&*result)
    }
    /// Returns a copy of the column as a BinaryDenseVector.
    fn col(&self, c: usize) -> BinaryDenseVector {
        assert!(c < self.ncols());
        let mut v = BinaryDenseVector::zero(self.nrows());
        for i in 0..self.nrows() {
            v.set(i, self.get(i, c));
        }
        v
    }
    /// Swaps the columns of the matrix.
    fn swap_columns(&mut self, c1: usize, c2: usize);
    /// XORs the values of columns c2 into c1.
    fn xor_col(&mut self, c1: usize, c2: usize);
    /// Alias for xor_col.
    fn add_col(&mut self, c1: usize, c2: usize) {
        self.xor_col(c1, c2);
    }
    /// Returns true if the given column is zero between 0 < maxr.
    fn column_part_all_zero(&self, c: usize, maxr: usize) -> bool {
        for x in 0..maxr {
            if self.get(x, c) == 1 {
                return false;
            }
        }
        return true;
    }
    /// Returns a vector of the given column from maxr < maxr+size.
    fn extract_column_part(&self, c: usize, maxr: usize, size: usize) -> BinaryDenseVector {
        let mut bits: Vec<u8> = Vec::with_capacity(size);
        for r in maxr..maxr + size {
            bits.push(self.get(r, c));
        }
        BinaryDenseVector::from_bits(&*bits)
    }
}

impl ops::Index<(usize, usize)> for dyn BinaryMatrix {
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &BITS[self.get(index.0, index.1) as usize]
    }
}

impl ops::Index<(usize, usize)> for &dyn BinaryMatrix {
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &BITS[self.get(index.0, index.1) as usize]
    }
}

impl ops::Mul<&dyn BinaryMatrix> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &dyn BinaryMatrix) -> Self::Output {
        assert_eq!(self.size, rhs.nrows());
        if self.size == 0 {
            return BinaryDenseVector::new();
        }
        let mut acc = BinaryDenseVector::zero(self.size);
        for i in 0..self.size {
            acc.set(i, self * &rhs.col(i));
        }
        acc
    }
}

impl ops::Mul<Box<dyn BinaryMatrix>> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: Box<dyn BinaryMatrix>) -> Self::Output {
        self * rhs.as_ref()
    }
}

impl fmt::Debug for dyn BinaryMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        binary_matrix_fmt(self, f)
    }
}
