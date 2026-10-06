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
        for _ in 0..n / 64 + 1 {
            v.push(rng.gen());
        }
        // clear the unused high bits so that equality, is_zero, and parity are correct
        *v.last_mut().unwrap() &= (1u64 << (n & 0x3f)) - 1;
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
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    // Sizes chosen to straddle the u64 word boundaries.
    const SIZES: [usize; 9] = [0, 1, 2, 63, 64, 65, 127, 128, 200];

    fn random_bits(n: usize, rng: &mut ChaCha8Rng) -> Vec<u8> {
        (0..n).map(|_| rng.gen_range(0..2)).collect()
    }

    #[test]
    fn test_format() {
        let mut v = BinaryDenseVector::zero(5);
        v.set(0, 1);
        v.set(1, 1);
        v.set(3, 1);
        assert_eq!("[1, 1, 0, 1, 0]", format!("{:?}", v));
        assert_eq!("[]", format!("{:?}", BinaryDenseVector::new()));
    }

    #[test]
    fn test_new_and_default() {
        let v = BinaryDenseVector::new();
        assert_eq!(0, v.size);
        assert!(v.is_zero());
        assert_eq!(v, BinaryDenseVector::default());
    }

    #[test]
    fn test_zero() {
        for n in SIZES {
            let v = BinaryDenseVector::zero(n);
            assert_eq!(n, v.size);
            assert!(v.is_zero());
            assert_eq!(0, v.parity());
            for i in 0..n {
                assert_eq!(0, v.get(i));
            }
        }
    }

    #[test]
    fn test_from_bits_get() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            let bits = random_bits(n, &mut rng);
            let v = BinaryDenseVector::from_bits(&bits);
            assert_eq!(n, v.size);
            for (i, b) in bits.iter().enumerate() {
                assert_eq!(*b, v.get(i));
                assert_eq!(*b, (&v)[i]);
            }
        }
    }

    #[test]
    fn test_set() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            let mut expected = vec![0u8; n];
            let mut v = BinaryDenseVector::zero(n);
            for _ in 0..4 * n {
                let i = rng.gen_range(0..n);
                let b = rng.gen_range(0..2);
                expected[i] = b;
                v.set(i, b);
            }
            assert_eq!(BinaryDenseVector::from_bits(&expected), v);
        }
    }

    #[test]
    fn test_is_zero() {
        for n in SIZES.into_iter().filter(|n| *n > 0) {
            let mut v = BinaryDenseVector::zero(n);
            v.set(n - 1, 1);
            assert!(!v.is_zero());
            v.set(n - 1, 0);
            assert!(v.is_zero());
        }
    }

    #[test]
    fn test_parity() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            let bits = random_bits(n, &mut rng);
            let expected = bits.iter().fold(0, |a, b| a ^ b);
            assert_eq!(expected, BinaryDenseVector::from_bits(&bits).parity());
        }
    }

    #[test]
    fn test_dot() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            for _ in 0..10 {
                let a = random_bits(n, &mut rng);
                let b = random_bits(n, &mut rng);
                let expected = a.iter().zip(&b).fold(0, |x, (a, b)| x ^ (a & b));
                let va = BinaryDenseVector::from_bits(&a);
                let vb = BinaryDenseVector::from_bits(&b);
                assert_eq!(expected, &va * &vb);
                assert_eq!(expected, &vb * &va);
            }
        }
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op, clippy::op_ref)] // exercise every operator impl
    fn test_scalar_mul() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            let v = BinaryDenseVector::from_bits(&random_bits(n, &mut rng));
            let zero = BinaryDenseVector::zero(n);
            assert_eq!(zero, 0u8 * v.clone());
            assert_eq!(zero, 0u8 * &v);
            assert_eq!(zero, &0u8 * &v);
            assert_eq!(zero, &v * 0u8);
            assert_eq!(zero, &v * &0u8);
            assert_eq!(v, 1u8 * v.clone());
            assert_eq!(v, 1u8 * &v);
            assert_eq!(v, &1u8 * &v);
            assert_eq!(v, &v * 1u8);
            assert_eq!(v, &v * &1u8);
        }
    }

    #[test]
    fn test_add() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for n in SIZES {
            let a = random_bits(n, &mut rng);
            let b = random_bits(n, &mut rng);
            let sum: Vec<u8> = a.iter().zip(&b).map(|(a, b)| a ^ b).collect();
            let expected = BinaryDenseVector::from_bits(&sum);
            let va = BinaryDenseVector::from_bits(&a);
            let vb = BinaryDenseVector::from_bits(&b);

            assert_eq!(expected, &va + &vb);
            assert!((&va + &va).is_zero());

            let mut c = va.clone();
            c += &vb;
            assert_eq!(expected, c);

            let mut c = va.clone();
            c += vb.clone();
            assert_eq!(expected, c);
        }
    }

    #[test]
    #[should_panic]
    fn test_add_size_mismatch() {
        let _ = &BinaryDenseVector::zero(3) + &BinaryDenseVector::zero(4);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random() {
        for n in SIZES {
            let mut rng = ChaCha8Rng::seed_from_u64(1234);
            let v = BinaryDenseVector::random(n, &mut rng);
            assert_eq!(n, v.size);
            for i in 0..n {
                assert!(v.get(i) <= 1);
            }
            // the stored words must not carry bits past the end of the vector
            let ones: u32 = v.bits.iter().map(|w| w.count_ones()).sum();
            let set = (0..n).filter(|i| v.get(*i) == 1).count() as u32;
            assert_eq!(set, ones);

            let mut rng = ChaCha8Rng::seed_from_u64(1234);
            assert_eq!(v, BinaryDenseVector::random(n, &mut rng));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        assert!(!BinaryDenseVector::random(1000, &mut rng).is_zero());
    }
}

#[cfg(all(test, feature = "bench"))]
mod bench {
    extern crate test;
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use test::bench::Bencher;

    fn random_vector(n: usize, rng: &mut ChaCha8Rng) -> BinaryDenseVector {
        let bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
        BinaryDenseVector::from_bits(&bits)
    }

    #[bench]
    fn bench_vector_from_bits_10000(b: &mut Bencher) {
        let bits = vec![1u8; 10000];
        b.iter(|| test::black_box(BinaryDenseVector::from_bits(&bits)));
    }

    #[bench]
    fn bench_vector_get_set_10000(b: &mut Bencher) {
        let mut v = BinaryDenseVector::zero(10000);
        b.iter(|| {
            for i in 0..10000 {
                v.set(i, 1 ^ v.get(i));
            }
            test::black_box(&v);
        });
    }

    #[bench]
    fn bench_vector_dot_10000(b: &mut Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let x = random_vector(10000, &mut rng);
        let y = random_vector(10000, &mut rng);
        b.iter(|| test::black_box(&x * &y));
    }

    #[bench]
    fn bench_vector_add_10000(b: &mut Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let x = random_vector(10000, &mut rng);
        let y = random_vector(10000, &mut rng);
        b.iter(|| test::black_box(&x + &y));
    }

    #[bench]
    fn bench_vector_add_assign_10000(b: &mut Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mut x = random_vector(10000, &mut rng);
        let y = random_vector(10000, &mut rng);
        b.iter(|| {
            x += &y;
            test::black_box(&x);
        });
    }

    #[bench]
    fn bench_vector_parity_10000(b: &mut Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let x = random_vector(10000, &mut rng);
        b.iter(|| test::black_box(x.parity()));
    }

    #[bench]
    fn bench_vector_is_zero_10000(b: &mut Bencher) {
        let x = BinaryDenseVector::zero(10000);
        b.iter(|| test::black_box(x.is_zero()));
    }
}
