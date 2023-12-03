use std::arch::aarch64;
use std::arch::aarch64::{uint8x8_t, uint8x8x2_t};
use std::intrinsics::transmute;

#[allow(dead_code)]
pub(crate) fn transpose64x64_aarch64_intrinsics(a: &mut [u64; 64]) {
    unsafe {
        let b = a.as_mut_ptr() as *mut u32;
        let mut ul = [0u32; 32];
        let ulp = (&mut ul).as_mut_ptr();
        let mut ur = [0u32; 32];
        let urp = (&mut ur).as_mut_ptr();
        let mut ll = [0u32; 32];
        let llp = (&mut ll).as_mut_ptr();
        let mut lr = [0u32; 32];
        let lrp = (&mut lr).as_mut_ptr();

        let c0 = aarch64::vld2q_u32(b);
        let c1 = aarch64::vld2q_u32(b.offset(8));
        let c2 = aarch64::vld2q_u32(b.offset(16));
        let c3 = aarch64::vld2q_u32(b.offset(24));
        let c4 = aarch64::vld2q_u32(b.offset(32));
        let c5 = aarch64::vld2q_u32(b.offset(40));
        let c6 = aarch64::vld2q_u32(b.offset(48));
        let c7 = aarch64::vld2q_u32(b.offset(56));
        aarch64::vst1q_u32(ulp, c0.0);
        aarch64::vst1q_u32(ulp.offset(4), c1.0);
        aarch64::vst1q_u32(ulp.offset(8), c2.0);
        aarch64::vst1q_u32(ulp.offset(12), c3.0);
        aarch64::vst1q_u32(ulp.offset(16), c4.0);
        aarch64::vst1q_u32(ulp.offset(20), c5.0);
        aarch64::vst1q_u32(ulp.offset(24), c6.0);
        aarch64::vst1q_u32(ulp.offset(28), c7.0);
        aarch64::vst1q_u32(urp, c0.1);
        aarch64::vst1q_u32(urp.offset(4), c1.1);
        aarch64::vst1q_u32(urp.offset(8), c2.1);
        aarch64::vst1q_u32(urp.offset(12), c3.1);
        aarch64::vst1q_u32(urp.offset(16), c4.1);
        aarch64::vst1q_u32(urp.offset(20), c5.1);
        aarch64::vst1q_u32(urp.offset(24), c6.1);
        aarch64::vst1q_u32(urp.offset(28), c7.1);

        let c0 = aarch64::vld2q_u32(b.offset(64));
        let c1 = aarch64::vld2q_u32(b.offset(72));
        let c2 = aarch64::vld2q_u32(b.offset(80));
        let c3 = aarch64::vld2q_u32(b.offset(88));
        let c4 = aarch64::vld2q_u32(b.offset(96));
        let c5 = aarch64::vld2q_u32(b.offset(104));
        let c6 = aarch64::vld2q_u32(b.offset(112));
        let c7 = aarch64::vld2q_u32(b.offset(120));
        aarch64::vst1q_u32(llp, c0.0);
        aarch64::vst1q_u32(llp.offset(4), c1.0);
        aarch64::vst1q_u32(llp.offset(8), c2.0);
        aarch64::vst1q_u32(llp.offset(12), c3.0);
        aarch64::vst1q_u32(llp.offset(16), c4.0);
        aarch64::vst1q_u32(llp.offset(20), c5.0);
        aarch64::vst1q_u32(llp.offset(24), c6.0);
        aarch64::vst1q_u32(llp.offset(28), c7.0);
        aarch64::vst1q_u32(lrp, c0.1);
        aarch64::vst1q_u32(lrp.offset(4), c1.1);
        aarch64::vst1q_u32(lrp.offset(8), c2.1);
        aarch64::vst1q_u32(lrp.offset(12), c3.1);
        aarch64::vst1q_u32(lrp.offset(16), c4.1);
        aarch64::vst1q_u32(lrp.offset(20), c5.1);
        aarch64::vst1q_u32(lrp.offset(24), c6.1);
        aarch64::vst1q_u32(lrp.offset(28), c7.1);

        transpose64x64_aarch64_intrinsics_32x32(ulp);
        transpose64x64_aarch64_intrinsics_32x32(urp);
        transpose64x64_aarch64_intrinsics_32x32(llp);
        transpose64x64_aarch64_intrinsics_32x32(lrp);

        // let the compiler figure out how to write this faster
        for i in 0..32 {
            *b.offset(2 * i) = ul[i as usize];
            *b.offset(2 * i + 1) = ll[i as usize];
        }
        for i in 0..32 {
            *b.offset(64 + 2 * i) = ur[i as usize];
            *b.offset(64 + 2 * i + 1) = lr[i as usize];
        }
    }
}

#[allow(dead_code)]
pub(crate) unsafe fn transpose64x64_aarch64_intrinsics_32x32(a: *mut u32) {
    let b = a as *mut u16;
    let mut ul = [0u16; 16];
    let ulp = (&mut ul).as_mut_ptr();
    let mut ur = [0u16; 16];
    let urp = (&mut ur).as_mut_ptr();
    let mut ll = [0u16; 16];
    let llp = (&mut ll).as_mut_ptr();
    let mut lr = [0u16; 16];
    let lrp = (&mut lr).as_mut_ptr();

    let c0 = aarch64::vld2q_u16(b.offset(0));
    let c1 = aarch64::vld2q_u16(b.offset(16));
    aarch64::vst1q_u16(ulp, c0.0);
    aarch64::vst1q_u16(ulp.offset(8), c1.0);
    aarch64::vst1q_u16(urp, c0.1);
    aarch64::vst1q_u16(urp.offset(8), c1.1);

    let c0 = aarch64::vld2q_u16(b.offset(32));
    let c1 = aarch64::vld2q_u16(b.offset(48));
    aarch64::vst1q_u16(llp, c0.0);
    aarch64::vst1q_u16(llp.offset(8), c1.0);
    aarch64::vst1q_u16(lrp, c0.1);
    aarch64::vst1q_u16(lrp.offset(8), c1.1);

    transpose64x64_aarch64_intrinsics_16x16(ulp);
    transpose64x64_aarch64_intrinsics_16x16(urp);
    transpose64x64_aarch64_intrinsics_16x16(llp);
    transpose64x64_aarch64_intrinsics_16x16(lrp);

    // let the compiler figure out how to write this faster
    for i in 0..16 {
        *b.offset(2 * i) = ul[i as usize];
        *b.offset(2 * i + 1) = ll[i as usize];
    }
    for i in 0..16 {
        *b.offset(32 + 2 * i) = ur[i as usize];
        *b.offset(32 + 2 * i + 1) = lr[i as usize];
    }
}

#[allow(dead_code)]
pub(crate) unsafe fn transpose64x64_aarch64_intrinsics_16x16(a: *mut u16) {
    let b = a as *mut u8;
    let c0 = aarch64::vld2_u8(b.offset(0));
    let c1 = aarch64::vld2_u8(b.offset(16));
    let ul = transpose64x64_aarch64_intrinsics_8x8(c0.0);
    let ur = transpose64x64_aarch64_intrinsics_8x8(c0.1);
    let ll = transpose64x64_aarch64_intrinsics_8x8(c1.0);
    let lr = transpose64x64_aarch64_intrinsics_8x8(c1.1);
    aarch64::vst2_u8(b, uint8x8x2_t(ul, ll));
    aarch64::vst2_u8(b.offset(16), uint8x8x2_t(ur, lr));
}

#[allow(dead_code)]
pub(crate) unsafe fn transpose64x64_aarch64_intrinsics_8x8(x: uint8x8_t) -> uint8x8_t {
    let a: u64 = transmute(x);
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

    let z = ((x as u64) << 32) | (y as u64);
    transmute(z)
}

#[cfg(feature = "rand")]
#[cfg(test)]
mod tests {
    use crate::base_matrix::slow_transpose;
    use crate::transpose64x64_aarch64_intrinsics::{
        transpose64x64_aarch64_intrinsics, transpose64x64_aarch64_intrinsics_16x16,
        transpose64x64_aarch64_intrinsics_32x32, transpose64x64_aarch64_intrinsics_8x8,
    };
    use crate::{BinaryMatrix, BinaryMatrix64};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::intrinsics::transmute;

    #[test]
    fn test_transpose64x64_aarch64_intrinsics() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut newmat = BinaryMatrix64::new();
        let mut a = mat.submatrix64(0, 0).cols;
        transpose64x64_aarch64_intrinsics(&mut a);
        slow_transpose(mat.as_ref(), &mut newmat);
        let b = newmat.submatrix64(0, 0).cols;
        assert_eq!(a, b);
        let mut b = [0u64; 64];
        for c in 0..64 {
            for r in 0..64 {
                if newmat.get(r, c) == 1 {
                    b[c] |= 1 << r;
                }
            }
        }
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose64x64_aarch64_intrinsics_32x32() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(32, 32, &mut rng);
        let mut newmat = BinaryMatrix64::zero(32, 32);
        let mut a = [0u32; 32];
        for c in 0..32 {
            for r in 0..32 {
                if mat.get(r, c) == 1 {
                    a[c] |= 1 << r;
                }
            }
        }
        unsafe {
            transpose64x64_aarch64_intrinsics_32x32(a.as_mut_ptr());
        }
        slow_transpose(mat.as_ref(), &mut newmat);
        let mut b = [0u32; 32];
        for c in 0..32 {
            for r in 0..32 {
                if newmat.get(r, c) == 1 {
                    b[c] |= 1 << r;
                }
            }
        }
        assert_eq!(a, b);
    }
    #[test]
    fn test_transpose64x64_aarch64_intrinsics_16x16() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(16, 16, &mut rng);
        let mut newmat = BinaryMatrix64::zero(16, 16);
        let mut a = [0u16; 16];
        for c in 0..16 {
            for r in 0..16 {
                if mat.get(r, c) == 1 {
                    a[c] |= 1 << r;
                }
            }
        }
        unsafe {
            transpose64x64_aarch64_intrinsics_16x16(a.as_mut_ptr());
        }
        slow_transpose(mat.as_ref(), &mut newmat);
        let mut b = [0u16; 16];
        for c in 0..16 {
            for r in 0..16 {
                if newmat.get(r, c) == 1 {
                    b[c] |= 1 << r;
                }
            }
        }
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose64x64_aarch64_intrinsics_8x8() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(16, 16, &mut rng);
        let mut newmat = BinaryMatrix64::zero(8, 8);
        let mut a = [0u8; 8];
        for c in 0..8 {
            for r in 0..8 {
                if mat.get(r, c) == 1 {
                    a[c] |= 1 << r;
                }
            }
        }
        unsafe {
            a = transmute(transpose64x64_aarch64_intrinsics_8x8(transmute(a)));
        }
        slow_transpose(mat.as_ref(), &mut newmat);
        let mut b = [0u8; 8];
        for c in 0..8 {
            for r in 0..8 {
                if newmat.get(r, c) == 1 {
                    b[c] |= 1 << r;
                }
            }
        }
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use crate::transpose64x64_aarch64_intrinsics::transpose64x64_aarch64_intrinsics;
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose64x64_aarch64_intrinsics(b: &mut Bencher) {
        let mut mat = [0u64; 64];
        mat[0] = 0xffffffffffffffff;
        b.iter(|| {
            test::black_box(transpose64x64_aarch64_intrinsics(&mut mat));
        });
    }
}
