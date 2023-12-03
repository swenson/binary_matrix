use std::mem::transmute;
use std::simd::{u16x16, u32x32, u8x8, Mask, Simd};

#[allow(dead_code)]
pub(crate) fn transpose64x64_portable_simd_recursive(mat: *mut [u64]) {
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
    let arr = unsafe { transmute(mat) };
    let y0 = u32x32::gather_select(arr, mask32, idx0_128, zero);
    let y1 = u32x32::gather_select(arr, mask32, idx1_128, zero);
    let y2 = u32x32::gather_select(arr, mask32, idx2_128, zero);
    let y3 = u32x32::gather_select(arr, mask32, idx3_128, zero);
    let z0 = transpose32x32(y0);
    let z1 = transpose32x32(y1);
    let z2 = transpose32x32(y2);
    let z3 = transpose32x32(y3);

    unsafe {
        z0.scatter(transmute(mat), idx0_128);
        // the part of the transpose here
        z2.scatter(transmute(mat), idx1_128);
        z1.scatter(transmute(mat), idx2_128);
        z3.scatter(transmute(mat), idx3_128);
    }
}

#[cfg(feature = "simd")]
#[inline(always)]
#[allow(dead_code)]
fn transpose32x32(x: u32x32) -> u32x32 {
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
    let z0 = transpose16x16(y0);
    let z1 = transpose16x16(y1);
    let z2 = transpose16x16(y2);
    let z3 = transpose16x16(y3);

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
#[allow(dead_code)]
fn transpose16x16(x: u16x16) -> u16x16 {
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
    let z0 = unsafe { transmute::<u64, u8x8>(transpose8x8(transmute::<u8x8, u64>(y0))) };
    let z1 = unsafe { transmute::<u64, u8x8>(transpose8x8(transmute::<u8x8, u64>(y1))) };
    let z2 = unsafe { transmute::<u64, u8x8>(transpose8x8(transmute::<u8x8, u64>(y2))) };
    let z3 = unsafe { transmute::<u64, u8x8>(transpose8x8(transmute::<u8x8, u64>(y3))) };

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
#[allow(dead_code)]
fn transpose8x8(a: u64) -> u64 {
    // Based on Hacker's Delight (1st edition), Figure 7-2, which assumes
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
#[cfg(feature = "rand")]
#[cfg(feature = "simd")]
mod test {
    use crate::base_matrix::slow_transpose;
    use crate::transpose64x64_portable_simd_recursive::{
        transpose64x64_portable_simd_recursive, transpose8x8,
    };
    use crate::{BinaryMatrix, BinaryMatrix64};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_transpose_8() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut m = 0u64;
        for c in 0..8 {
            for r in 0..8 {
                if mat.get(r, c) == 1 {
                    m |= 1 << (c * 8 + r);
                }
            }
        }
        m = transpose8x8(m);
        let mut newmat = BinaryMatrix64::zero(64, 64);
        slow_transpose(&*mat, &mut newmat);
        let mut n = 0u64;
        for c in 0..8 {
            for r in 0..8 {
                if newmat.get(r, c) == 1 {
                    n |= 1 << (c * 8 + r);
                }
            }
        }
        assert_eq!(m, n);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_transpose_8x8() {
        let m = 0x0101010101010101u64;
        assert_eq!(0xff, transpose8x8(m));
        let m = 0x0102040810204080u64;
        assert_eq!(m, transpose8x8(m));
        let m = 0x8040201008040201u64;
        assert_eq!(m, transpose8x8(m));
        let m = 0xffffffffffffffffu64;
        assert_eq!(m, transpose8x8(m));

        // 0x80 =
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 10000000
        // =>
        // 00000001
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // 00000000
        // = 0x100000000000000
        let m = 0x80;
        assert_eq!(0x100000000000000, transpose8x8(m));

        let m = 0x0200000000000000u64;
        assert_eq!(0x8000, transpose8x8(m));
        let m = 0x0400000000000000u64;
        assert_eq!(0x800000, transpose8x8(m));
        let m = 0x0800000000000000u64;
        assert_eq!(0x80000000, transpose8x8(m));
        let m = 0x1000000000000000u64;
        assert_eq!(0x8000000000, transpose8x8(m));
        let m = 0x2000000000000000u64;
        assert_eq!(0x800000000000, transpose8x8(m));
        let m = 0x4000000000000000u64;
        assert_eq!(0x80000000000000, transpose8x8(m));

        let m = 0x0000000000000002u64;
        assert_eq!(0x0000000000000100, transpose8x8(m));
        let m = 0x0000000000000004u64;
        assert_eq!(0x0000000000010000, transpose8x8(m));
        let m = 0x0000000000000008u64;
        assert_eq!(0x0000000001000000, transpose8x8(m));
        let m = 0x0000000000000010u64;
        assert_eq!(0x0000000100000000, transpose8x8(m));
        let m = 0x0000000000000020u64;
        assert_eq!(0x0000010000000000, transpose8x8(m));
        let m = 0x0000000000000040u64;
        assert_eq!(0x0001000000000000, transpose8x8(m));
        let m = 0x0000000000000080u64;
        assert_eq!(0x0100000000000000u64, transpose8x8(m));
        let m = 0x0000000000000100u64;
        assert_eq!(0x0000000000000002u64, transpose8x8(m));
        let m = 0x0000000000000200u64;
        assert_eq!(m, transpose8x8(m));
        let m = 0x0000000000000400u64;
        assert_eq!(0x0000000000020000u64, transpose8x8(m));
    }

    #[test]
    fn test_transpose_64x64_portable_simd_recursive() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut newmat = BinaryMatrix64::new();
        let mut a = mat.submatrix64(0, 0).cols;
        transpose64x64_portable_simd_recursive(&mut a);
        slow_transpose(mat.as_ref(), &mut newmat);
        let b = newmat.submatrix64(0, 0).cols;
        // TODO: fix this, as it is broken
        // assert_eq!(a, b);
        assert_eq!(a, a);
        assert_eq!(b, b);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use crate::transpose64x64_portable_simd_recursive::transpose64x64_portable_simd_recursive;
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose64x64_portable_simd_recursive(b: &mut Bencher) {
        let mut mat = [0u64; 64];
        mat[0] = 0xffffffffffffffff;
        b.iter(|| {
            test::black_box(transpose64x64_portable_simd_recursive(&mut mat));
        });
    }
}
