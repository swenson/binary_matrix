use core::ops;
#[cfg(feature = "rand")]
use rand::Rng;
use std::fmt;
use std::fmt::{Debug, Write};

pub(crate) const BITS: [u8; 2] = [0, 1];

/// A dense vector of bits, packed into u64 elements.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct BinaryDenseVector {
    pub(crate) size: usize,
    pub(crate) bits: Vec<u64>,
}

impl Default for BinaryDenseVector {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryDenseVector {
    pub fn new() -> BinaryDenseVector {
        BinaryDenseVector {
            size: 0,
            bits: vec![],
        }
    }

    /// Returns a new vector of zeros of the given size.
    pub fn zero(n: usize) -> BinaryDenseVector {
        let v = vec![0; n / 64 + 1];
        BinaryDenseVector { size: n, bits: v }
    }

    /// Creates a new vector from the array of bits (one per u8).
    pub fn from_bits(bits: &[u8]) -> BinaryDenseVector {
        let newvec = vec![0; bits.len() / 64 + 1];
        let mut v = BinaryDenseVector {
            size: bits.len(),
            bits: newvec,
        };
        for (i, b) in bits.iter().enumerate() {
            v.set(i, *b);
        }
        v
    }

    /// Returns a new random vector of the given size.
    #[cfg(feature = "rand")]
    pub fn random<R: Rng + ?Sized>(n: usize, rng: &mut R) -> BinaryDenseVector {
        let mut v = Vec::with_capacity(n / 64 + 1);
        for _ in 0..v.len() {
            v.push(rng.gen());
        }
        BinaryDenseVector { size: n, bits: v }
    }

    /// Return true if all entries in the vector are zero.
    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|x| *x == 0)
    }

    /// Returns the value of the vector at the given index.
    pub fn get(&self, i: usize) -> u8 {
        (self.bits[i >> 6] >> (i & 0x3f)) as u8 & 1
    }

    /// Sets the value at the given index to to b.
    pub fn set(&mut self, i: usize, b: u8) {
        let shift = i & 0x3f;
        self.bits[i >> 6] = (self.bits[i >> 6] & (!(1 << shift))) | ((b as u64) << shift);
    }

    /// Return the sum (mod 2) of all of the bits of the vector.
    pub fn parity(&self) -> u8 {
        let mut x = 0u8;
        for b in self.bits.iter() {
            x ^= b.count_ones() as u8 & 1;
        }
        x
    }
}

impl ops::Mul<&BinaryDenseVector> for &BinaryDenseVector {
    type Output = u8;

    fn mul(self, rhs: &BinaryDenseVector) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        let mut x = 0u8;
        for i in 0..(self.size >> 6) {
            x ^= (self.bits[i] & rhs.bits[i]).count_ones() as u8 & 1;
        }
        for i in (self.size & !0x3f)..self.size {
            x ^= self[i] & rhs[i];
        }
        x
    }
}

impl ops::Mul<BinaryDenseVector> for u8 {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: BinaryDenseVector) -> Self::Output {
        self * &rhs
    }
}

impl ops::Mul<&BinaryDenseVector> for u8 {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &BinaryDenseVector) -> Self::Output {
        rhs * self
    }
}

impl ops::Mul<&BinaryDenseVector> for &u8 {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &BinaryDenseVector) -> Self::Output {
        rhs * *self
    }
}

impl ops::Mul<u8> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: u8) -> Self::Output {
        if rhs == 0 {
            BinaryDenseVector::zero(self.size)
        } else {
            self.clone()
        }
    }
}

impl ops::Mul<&u8> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn mul(self, rhs: &u8) -> Self::Output {
        self * *rhs
    }
}
impl ops::Add<&BinaryDenseVector> for &BinaryDenseVector {
    type Output = BinaryDenseVector;

    fn add(self, rhs: &BinaryDenseVector) -> Self::Output {
        assert_eq!(self.size, rhs.size);
        let mut new_bits = Vec::with_capacity(self.bits.len());
        for i in 0..self.bits.len() {
            new_bits.push(self.bits[i] ^ rhs.bits[i]);
        }
        BinaryDenseVector {
            size: self.size,
            bits: new_bits,
        }
    }
}

impl ops::AddAssign<BinaryDenseVector> for BinaryDenseVector {
    fn add_assign(&mut self, rhs: BinaryDenseVector) {
        assert_eq!(self.size, rhs.size);
        for i in 0..self.bits.len() {
            self.bits[i] ^= rhs.bits[i];
        }
    }
}

impl ops::AddAssign<&BinaryDenseVector> for BinaryDenseVector {
    fn add_assign(&mut self, rhs: &BinaryDenseVector) {
        assert_eq!(self.size, rhs.size);
        for i in 0..self.bits.len() {
            self.bits[i] ^= rhs.bits[i];
        }
    }
}

impl ops::Index<usize> for &BinaryDenseVector {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &BITS[self.get(index) as usize]
    }
}

impl Debug for BinaryDenseVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('[')?;
        for c in 0..self.size {
            if c != 0 {
                f.write_str(", ")?;
            }
            f.write_char(char::from(48 + self[c]))?;
        }
        f.write_char(']')
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_format() {
        let mut v = BinaryDenseVector::zero(5);
        v.set(0, 1);
        v.set(1, 1);
        v.set(3, 1);
        assert_eq!("[1, 1, 0, 1, 0]", format!("{:?}", v));
    }
}
