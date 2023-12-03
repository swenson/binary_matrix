use std::arch::asm;

// adapted from https://stackoverflow.com/a/71653601
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) unsafe fn transpose_asm_aarch64_2(cols: *mut u64) {
    let mut colscopy = cols;
    let mut out = [0u64; 64];
    let mut outptr = out.as_mut_ptr();
    asm! {
    "mov     {pSrc0}, {pSrc}",
    "add     {pSrc1}, {pSrc0}, #64",
    "movi    v6.16b, #0xcc",
    "mov     {stride}, #128",
    "movi    v7.16b, #0xaa",
    "sub     {pDst}, {pDst}, #32",
    "mov     w5, #2",

    "1:",
    "ld4     {{v16.16b, v17.16b, v18.16b, v19.16b}}, [{pSrc0}], {stride}",
    "ld4     {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{pSrc1}], {stride}",
    "ld4     {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{pSrc0}], {stride}",
    "ld4     {{v28.16b, v29.16b, v30.16b, v31.16b}}, [{pSrc1}], {stride}",

    "stp     q16, q20, [{pDst}, #32]!",
    "subs    w5, w5, #1",
    "stp     q17, q21, [{pDst}, #1*64]",
    "stp     q18, q22, [{pDst}, #2*64]",
    "stp     q19, q23, [{pDst}, #3*64]",
    "stp     q24, q28, [{pDst}, #4*64]",
    "stp     q25, q29, [{pDst}, #5*64]",
    "stp     q26, q30, [{pDst}, #6*64]",
    "stp     q27, q31, [{pDst}, #7*64]",
    "b.ne    1b",
    // 8x64 matrix transpose virtually finished.
    "nop",

    "sub     {pSrc0}, {pDst}, #32",
    "add     {pSrc1}, {pDst}, #256-32",
    "mov     w5, #4",
    "sub     {pDst0}, {pDst}, #32",
    "add     {pDst1}, {pSrc0}, #256",

    "1:",
    // 8x64 matrix transpose finished on-the-fly while reloading.
    "ld2     {{v24.16b, v25.16b}}, [{pSrc0}], #32",
    "ld2     {{v26.16b, v27.16b}}, [{pSrc1}], #32",
    "ld2     {{v28.16b, v29.16b}}, [{pSrc0}], #32",
    "ld2     {{v30.16b, v31.16b}}, [{pSrc1}], #32",
    "subs    w5, w5, #1",

    "// the trns below aren't part of the matrix transpose",
    "trn1    v16.2d, v24.2d, v25.2d", // row0
    "trn2    v17.2d, v24.2d, v25.2d", // row1
    "trn1    v18.2d, v26.2d, v27.2d", // row2
    "trn2    v19.2d, v26.2d, v28.2d", // row3
    "trn1    v20.2d, v28.2d, v29.2d", // row4
    "trn2    v21.2d, v28.2d, v29.2d", // row5
    "trn1    v22.2d, v30.2d, v31.2d", // row6
    "trn2    v23.2d, v30.2d, v31.2d", // row7

    "mov     v24.16b, v16.16b",
    "mov     v25.16b, v17.16b",
    "mov     v26.16b, v18.16b",
    "mov     v27.16b, v19.16b",

    "sli     v16.16b, v20.16b, #4",
    "sli     v17.16b, v21.16b, #4",
    "sli     v18.16b, v22.16b, #4",
    "sli     v19.16b, v23.16b, #4",
    "sri     v20.16b, v24.16b, #4",
    "sri     v21.16b, v25.16b, #4",
    "sri     v22.16b, v26.16b, #4",
    "sri     v23.16b, v27.16b, #4",

    "shl     v24.16b, v18.16b, #2",
    "shl     v25.16b, v19.16b, #2",
    "ushr    v26.16b, v16.16b, #2",
    "ushr    v27.16b, v17.16b, #2",
    "shl     v28.16b, v22.16b, #2",
    "shl     v29.16b, v23.16b, #2",
    "ushr    v30.16b, v20.16b, #2",
    "ushr    v31.16b, v21.16b, #2",

    "bit     v16.16b, v24.16b, v6.16b",
    "bit     v17.16b, v25.16b, v6.16b",
    "bif     v18.16b, v26.16b, v6.16b",
    "bif     v19.16b, v27.16b, v6.16b",
    "bit     v20.16b, v28.16b, v6.16b",
    "bit     v21.16b, v29.16b, v6.16b",
    "bif     v22.16b, v30.16b, v6.16b",
    "bif     v23.16b, v31.16b, v6.16b",

    "shl     v24.16b, v17.16b, #1",
    "ushr    v25.16b, v16.16b, #1",
    "shl     v26.16b, v19.16b, #1",
    "ushr    v27.16b, v18.16b, #1",
    "shl     v28.16b, v21.16b, #1",
    "ushr    v29.16b, v20.16b, #1",
    "shl     v30.16b, v23.16b, #1",
    "ushr    v31.16b, v22.16b, #1",

    "bit     v16.16b, v24.16b, v7.16b",
    "bif     v17.16b, v25.16b, v7.16b",
    "bit     v18.16b, v26.16b, v7.16b",
    "bif     v19.16b, v27.16b, v7.16b",
    "bit     v20.16b, v28.16b, v7.16b",
    "bif     v21.16b, v29.16b, v7.16b",
    "bit     v22.16b, v30.16b, v7.16b",
    "bif     v23.16b, v31.16b, v7.16b",

    "st4     {{v16.d, v17.d, v18.d, v19.d}}[0], [{pDst0}], #32",
    "st4     {{v16.d, v17.d, v18.d, v19.d}}[1], [{pDst1}], #32",
    "st4     {{v20.d, v21.d, v22.d, v23.d}}[0], [{pDst0}], #32",
    "st4     {{v20.d, v21.d, v22.d, v23.d}}[1], [{pDst1}], #32",
    "b.ne    1b",
    pSrc = inout(reg) colscopy,
    pDst = inout(reg) outptr,
    pSrc0 = out(reg) _,
    pSrc1 = out(reg) _,
    pDst0 = out(reg) _,
    pDst1 = out(reg) _,
    stride = out(reg) _,
    out("w5") _,
    out("v0") _,
    out("v1") _,
    out("v2") _,
    out("v3") _,
    out("v4") _,
    out("v5") _,
    out("v6") _,
    out("v7") _,
    out("v16") _,
    out("v17") _,
    out("v18") _,
    out("v19") _,
    out("v20") _,
    out("v21") _,
    out("v22") _,
    out("v23") _,
    out("v24") _,
    out("v25") _,
    out("v26") _,
    out("v27") _,
    out("v28") _,
    out("v29") _,
    out("v30") _,
    out("v31") _,
    }
    _ = colscopy;
    _ = outptr;

    for i in 0..64 {
        *cols.offset(i) = out[i as usize];
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod tests {
    use crate::base_matrix::slow_transpose;
    use crate::transpose64x64_asm_aarch2::transpose_asm_aarch64_2;
    use crate::BinaryMatrix64;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_transpose_64x64_aarch_asm_2() {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let mat = BinaryMatrix64::random(64, 64, &mut rng);
        let mut newmat = BinaryMatrix64::new();
        let mut a = mat.submatrix64(0, 0).cols;
        unsafe { transpose_asm_aarch64_2(a.as_mut_ptr()) };
        slow_transpose(mat.as_ref(), &mut newmat);
        let b = newmat.submatrix64(0, 0).cols;
        assert_eq!(a, b);
    }
}

#[cfg(test)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod bench {
    extern crate test;
    use crate::transpose64x64_asm_aarch2::transpose_asm_aarch64_2;
    use test::bench::Bencher;

    #[bench]
    fn bench_transpose64x64_aarch64_asm_2(b: &mut Bencher) {
        let mut mat = [0u64; 64];
        mat[0] = 0xffffffffffffffff;
        b.iter(|| {
            test::black_box(unsafe {
                transpose_asm_aarch64_2(mat.as_mut_ptr());
            });
        });
    }
}
