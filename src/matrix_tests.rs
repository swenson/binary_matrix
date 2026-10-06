//! Conformance tests run against every `BinaryMatrix` implementation.
//!
//! Each operation is checked against a naive reference model (a `Vec` of rows of `u8`s).

use crate::base_matrix::BinaryMatrix;
use crate::binary_dense_vector::BinaryDenseVector;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Shapes chosen to straddle u64 word boundaries and the 64x64 transpose block size.
pub(crate) const SHAPES: [(usize, usize); 14] = [
    (0, 0),
    (0, 5),
    (5, 0),
    (1, 1),
    (1, 70),
    (70, 1),
    (3, 4),
    (4, 3),
    (63, 65),
    (64, 64),
    (65, 63),
    (64, 128),
    (130, 70),
    (200, 200),
];

pub(crate) type Reference = Vec<Vec<u8>>;

/// Fills the matrix with random bits and returns the matching reference.
pub(crate) fn fill_random(m: &mut dyn BinaryMatrix, rng: &mut ChaCha8Rng) -> Reference {
    let mut reference = vec![vec![0u8; m.ncols()]; m.nrows()];
    for (r, row) in reference.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = rng.gen_range(0..2);
            m.set(r, c, *x);
        }
    }
    reference
}

/// Fills the matrix with a random matrix of at most the given rank.
pub(crate) fn fill_low_rank(
    m: &mut dyn BinaryMatrix,
    rank: usize,
    rng: &mut ChaCha8Rng,
) -> Reference {
    let basis: Vec<Vec<u8>> = (0..rank)
        .map(|_| (0..m.ncols()).map(|_| rng.gen_range(0..2)).collect())
        .collect();
    let mut reference = vec![vec![0u8; m.ncols()]; m.nrows()];
    for (r, row) in reference.iter_mut().enumerate() {
        for b in basis.iter() {
            if rng.gen() {
                for (x, y) in row.iter_mut().zip(b) {
                    *x ^= y;
                }
            }
        }
        for (c, x) in row.iter().enumerate() {
            m.set(r, c, *x);
        }
    }
    reference
}

pub(crate) fn assert_matches(m: &dyn BinaryMatrix, reference: &Reference) {
    assert_eq!(reference.len(), m.nrows());
    for (r, row) in reference.iter().enumerate() {
        assert_eq!(row.len(), m.ncols());
        for (c, x) in row.iter().enumerate() {
            assert_eq!(*x, m.get(r, c), "mismatch at ({}, {})", r, c);
        }
    }
}

pub(crate) fn rank(rows: &[Vec<u8>]) -> usize {
    let mut rows = rows.to_vec();
    let ncols = rows.first().map_or(0, |r| r.len());
    let mut rank = 0;
    for c in 0..ncols {
        if let Some(p) = (rank..rows.len()).find(|r| rows[*r][c] == 1) {
            rows.swap(rank, p);
            for r in 0..rows.len() {
                if r != rank && rows[r][c] == 1 {
                    let pivot = rows[rank].clone();
                    for (x, y) in rows[r].iter_mut().zip(pivot) {
                        *x ^= y;
                    }
                }
            }
            rank += 1;
        }
    }
    rank
}

pub(crate) fn transpose(reference: &Reference, nrows: usize, ncols: usize) -> Reference {
    (0..ncols)
        .map(|c| (0..nrows).map(|r| reference[r][c]).collect())
        .collect()
}

fn to_bits(v: &BinaryDenseVector) -> Vec<u8> {
    (0..v.size).map(|i| v.get(i)).collect()
}

/// Checks that `basis` is a basis for the kernel of the matrix `reference` (with `ncols` columns).
pub(crate) fn assert_kernel(reference: &Reference, ncols: usize, basis: &[BinaryDenseVector]) {
    let basis: Vec<Vec<u8>> = basis.iter().map(to_bits).collect();
    for v in basis.iter() {
        assert_eq!(ncols, v.len());
        for row in reference.iter() {
            let dot = row.iter().zip(v).fold(0, |a, (x, y)| a ^ (x & y));
            assert_eq!(0, dot, "kernel vector {:?} is not in the kernel", v);
        }
    }
    // linearly independent, and spanning: dim ker = ncols - rank
    assert_eq!(basis.len(), rank(&basis), "kernel basis is not independent");
    assert_eq!(ncols - rank(reference), basis.len());
}

/// Generates the conformance test suite for a `BinaryMatrix` implementation.
///
/// The type must provide `new()`, `zero(rows, cols)`, and `identity(rows)` returning a `Box<Self>`.
macro_rules! matrix_conformance_tests {
    ($name:ident, $t:ty) => {
        mod $name {
            use crate::base_matrix::BinaryMatrix;
            use crate::binary_dense_vector::BinaryDenseVector;
            use crate::matrix_tests::*;

            fn random(rows: usize, cols: usize, rng: &mut ChaCha8Rng) -> (Box<$t>, Reference) {
                let mut m = <$t>::zero(rows, cols);
                let reference = fill_random(m.as_mut(), rng);
                (m, reference)
            }

            #[test]
            fn test_new() {
                let m = <$t>::new();
                assert_eq!(0, m.nrows());
                assert_eq!(0, m.ncols());
            }

            #[test]
            fn test_zero() {
                for (r, c) in SHAPES {
                    let m = <$t>::zero(r, c);
                    assert_matches(m.as_ref(), &vec![vec![0; c]; r]);
                }
            }

            #[test]
            fn test_identity() {
                for n in [0, 1, 63, 64, 65, 130] {
                    let m = <$t>::identity(n);
                    let reference = (0..n)
                        .map(|r| (0..n).map(|c| (r == c) as u8).collect())
                        .collect();
                    assert_matches(m.as_ref(), &reference);
                }
            }

            #[test]
            fn test_get_set() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (mut m, mut reference) = random(r, c, &mut rng);
                    assert_matches(m.as_ref(), &reference);
                    // overwrite random positions with both 0 and 1
                    for _ in 0..(r * c) {
                        let (i, j, x) = (
                            rng.gen_range(0..r),
                            rng.gen_range(0..c),
                            rng.gen_range(0..2),
                        );
                        m.set(i, j, x);
                        reference[i][j] = x;
                    }
                    assert_matches(m.as_ref(), &reference);
                }
            }

            #[test]
            #[should_panic]
            fn test_get_out_of_bounds() {
                <$t>::zero(64, 1).get(64, 0);
            }

            #[test]
            fn test_index() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                let (m, reference) = random(70, 70, &mut rng);
                let concrete: &$t = m.as_ref();
                let dynamic: &dyn BinaryMatrix = m.as_ref();
                for r in 0..70 {
                    for c in 0..70 {
                        assert_eq!(reference[r][c], concrete[(r, c)]);
                        assert_eq!(reference[r][c], dynamic[(r, c)]);
                    }
                }
            }

            #[test]
            fn test_expand() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    for (dr, dc) in [(0, 0), (1, 0), (0, 1), (64, 3), (100, 100)] {
                        let (mut m, mut reference) = random(r, c, &mut rng);
                        m.expand(dr, dc);
                        for row in reference.iter_mut() {
                            row.resize(c + dc, 0);
                        }
                        reference.resize(r + dr, vec![0; c + dc]);
                        assert_matches(m.as_ref(), &reference);
                        // the new space must be usable
                        if r + dr > 0 && c + dc > 0 {
                            m.set(r + dr - 1, c + dc - 1, 1);
                            assert_eq!(1, m.get(r + dr - 1, c + dc - 1));
                        }
                    }
                }
            }

            #[test]
            fn test_copy() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    let mut copy = m.copy();
                    assert_matches(copy.as_ref(), &reference);
                    // the copy must not share storage with the original
                    if r > 0 && c > 0 {
                        copy.set(0, 0, 1 ^ reference[0][0]);
                        assert_eq!(reference[0][0], m.get(0, 0));
                    }
                }
            }

            #[test]
            fn test_eq() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                let (m, _) = random(70, 70, &mut rng);
                let mut other = m.copy();
                assert_eq!(m.as_ref() as &dyn BinaryMatrix, other.as_ref());
                other.set(69, 69, 1 ^ m.get(69, 69));
                assert_ne!(m.as_ref() as &dyn BinaryMatrix, other.as_ref());
                assert_ne!(
                    <$t>::zero(2, 3).as_ref() as &dyn BinaryMatrix,
                    <$t>::zero(3, 2).as_ref() as &dyn BinaryMatrix
                );
            }

            #[test]
            fn test_transpose() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    assert_matches(m.transpose().as_ref(), &transpose(&reference, r, c));
                }
            }

            #[test]
            fn test_transpose_large() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in [(1000, 600), (600, 1000), (256, 256)] {
                    let (m, reference) = random(r, c, &mut rng);
                    assert_matches(m.transpose().as_ref(), &transpose(&reference, r, c));
                }
            }

            #[test]
            fn test_col() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    for j in 0..c {
                        let col: Vec<u8> = (0..r).map(|i| reference[i][j]).collect();
                        assert_eq!(BinaryDenseVector::from_bits(&col), m.col(j));
                    }
                }
            }

            #[test]
            fn test_swap_columns() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES.into_iter().filter(|(_, c)| *c > 0) {
                    let (mut m, mut reference) = random(r, c, &mut rng);
                    for _ in 0..10 {
                        let (c1, c2) = (rng.gen_range(0..c), rng.gen_range(0..c));
                        m.swap_columns(c1, c2);
                        for row in reference.iter_mut() {
                            row.swap(c1, c2);
                        }
                        assert_matches(m.as_ref(), &reference);
                    }
                }
            }

            #[test]
            fn test_xor_col() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES.into_iter().filter(|(_, c)| *c > 0) {
                    let (mut m, mut reference) = random(r, c, &mut rng);
                    for i in 0..10 {
                        let (c1, c2) = (rng.gen_range(0..c), rng.gen_range(0..c));
                        if i % 2 == 0 {
                            m.xor_col(c1, c2);
                        } else {
                            m.add_col(c1, c2);
                        }
                        for row in reference.iter_mut() {
                            row[c1] ^= row[c2];
                        }
                        assert_matches(m.as_ref(), &reference);
                    }
                }
            }

            #[test]
            fn test_column_part_all_zero() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for n in [1, 63, 64, 65, 200] {
                    let mut m = <$t>::zero(n, n);
                    // column c has a single bit set in row c
                    for c in 0..n {
                        m.set(c, c, 1);
                    }
                    for c in 0..n {
                        for maxr in 0..=n {
                            assert_eq!(maxr <= c, m.column_part_all_zero(c, maxr));
                        }
                    }
                    // and random sparse columns
                    let mut m = <$t>::zero(n, 4);
                    let mut first = [n; 4];
                    for c in 0..4 {
                        for r in 0..n {
                            if rng.gen_ratio(1, 32) {
                                m.set(r, c, 1);
                                first[c] = first[c].min(r);
                            }
                        }
                        for maxr in 0..=n {
                            assert_eq!(maxr <= first[c], m.column_part_all_zero(c, maxr));
                        }
                    }
                }
            }

            #[test]
            fn test_extract_column_part() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                let (m, reference) = random(200, 5, &mut rng);
                for c in 0..5 {
                    for (start, size) in [(0, 0), (0, 200), (10, 100), (64, 64), (199, 1)] {
                        let bits: Vec<u8> =
                            (start..start + size).map(|r| reference[r][c]).collect();
                        assert_eq!(
                            BinaryDenseVector::from_bits(&bits),
                            m.extract_column_part(c, start, size)
                        );
                    }
                }
            }

            #[test]
            fn test_left_mul() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    for _ in 0..5 {
                        let bits: Vec<u8> = (0..r).map(|_| rng.gen_range(0..2)).collect();
                        let v = BinaryDenseVector::from_bits(&bits);
                        let expected: Vec<u8> = (0..c)
                            .map(|j| (0..r).fold(0, |a, i| a ^ (bits[i] & reference[i][j])))
                            .collect();
                        let expected = BinaryDenseVector::from_bits(&expected);
                        assert_eq!(expected, m.left_mul(&v));
                        assert_eq!(expected, &v * m.as_ref());
                        assert_eq!(expected, &v * (m.as_ref() as &dyn BinaryMatrix));
                        assert_eq!(expected, &v * m.copy());
                        assert_eq!(expected, &v * m.clone());
                    }
                }
            }

            #[test]
            #[should_panic]
            fn test_left_mul_size_mismatch() {
                <$t>::zero(3, 3).left_mul(&BinaryDenseVector::zero(4));
            }

            #[test]
            fn test_kernel() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    assert_kernel(&reference, c, &m.kernel().unwrap());
                }
            }

            #[test]
            fn test_kernel_low_rank() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in [(10, 10), (70, 70), (100, 130), (130, 100)] {
                    for k in [0, 1, 5, 60] {
                        let mut m = <$t>::zero(r, c);
                        let reference = fill_low_rank(m.as_mut(), k, &mut rng);
                        assert_kernel(&reference, c, &m.kernel().unwrap());
                    }
                }
            }

            #[test]
            fn test_kernel_identity() {
                for n in [0, 1, 64, 65] {
                    assert_eq!(Some(vec![]), <$t>::identity(n).kernel());
                    assert_eq!(Some(vec![]), <$t>::identity(n).left_kernel());
                }
            }

            #[test]
            fn test_left_kernel() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in SHAPES {
                    let (m, reference) = random(r, c, &mut rng);
                    let basis = m.left_kernel().unwrap();
                    assert_kernel(&transpose(&reference, r, c), r, &basis);
                    for v in basis.iter() {
                        assert!((v * m.as_ref()).is_zero());
                    }
                }
            }

            #[test]
            fn test_left_kernel_low_rank() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                for (r, c) in [(10, 10), (70, 70), (100, 130), (130, 100)] {
                    for k in [0, 1, 5, 60] {
                        let mut m = <$t>::zero(r, c);
                        let reference = fill_low_rank(m.as_mut(), k, &mut rng);
                        let basis = m.left_kernel().unwrap();
                        assert_kernel(&transpose(&reference, r, c), r, &basis);
                    }
                }
            }

            #[test]
            fn test_format() {
                let mut rng = ChaCha8Rng::seed_from_u64(1234);
                let (m, reference) = random(3, 4, &mut rng);
                let rows: Vec<String> = reference
                    .iter()
                    .map(|row| {
                        let cells: Vec<String> = row.iter().map(|x| x.to_string()).collect();
                        format!("[{}]", cells.join(", "))
                    })
                    .collect();
                let expected = format!("[{}]", rows.join(", "));
                assert_eq!(expected, format!("{:?}", m));
                assert_eq!(expected, format!("{:?}", m.as_ref() as &dyn BinaryMatrix));
                assert_eq!("[]", format!("{:?}", <$t>::new()));
            }
        }
    };
}

matrix_conformance_tests!(binary_matrix_64, crate::BinaryMatrix64);
#[cfg(feature = "simd")]
matrix_conformance_tests!(binary_matrix_simd_1, crate::BinaryMatrixSimd<1>);
#[cfg(feature = "simd")]
matrix_conformance_tests!(binary_matrix_simd_2, crate::BinaryMatrixSimd<2>);
#[cfg(feature = "simd")]
matrix_conformance_tests!(binary_matrix_simd_64, crate::BinaryMatrixSimd<64>);
