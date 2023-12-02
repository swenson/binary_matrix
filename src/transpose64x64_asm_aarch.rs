// adapted from https://github.com/clausecker/xpose64x64/blob/master/xpose_arm64.S
// Copyright (c) 2022 Robert Clausecker.  All rights reserved.
// Licensed under the 2-clause BSD license.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn transpose_asm_aarch64(cols: *mut u64) {
    unsafe {
        asm!(
        "mov {save_lr}, lr",

        "mov {t2}, {arr}",
        "bl	1f",
        "mov {t3}, {arr}",
        "bl 1f",

        // final step: transpose 64x64 bit matrices
        // we have to do this one in two parts as to not run
        // out of registers
        "mov {t5}, {t2}",
        "mov {t6}, {t3}",
        "bl	2f",
        "bl	2f",
        // end asm
        "mov lr, {save_lr}",
        "b 3f",

        // xpose_half
        "1:",
        "mov {save_arr}, {arr}",

        "ld4 {{ v16.2d, v17.2d, v18.2d, v19.2d }}, [{arr}], #64",
        "ld4 {{ v20.2d, v21.2d, v22.2d, v23.2d }}, [{arr}], #64",
        "ld4 {{ v24.2d, v25.2d, v26.2d, v27.2d }}, [{arr}], #64",
        "ld4 {{ v28.2d, v29.2d, v30.2d, v31.2d }}, [{arr}], #64",
        // "rbit v16.16b, v16.16b",
        // "rbit v17.16b, v17.16b",
        // "rbit v18.16b, v18.16b",
        // "rbit v19.16b, v19.16b",
        // "rbit v20.16b, v20.16b",
        // "rbit v21.16b, v21.16b",
        // "rbit v22.16b, v22.16b",
        // "rbit v23.16b, v23.16b",
        // "rbit v24.16b, v24.16b",
        // "rbit v25.16b, v25.16b",
        // "rbit v26.16b, v26.16b",
        // "rbit v27.16b, v27.16b",
        // "rbit v28.16b, v28.16b",
        // "rbit v29.16b, v29.16b",
        // "rbit v30.16b, v30.16b",
        // "rbit v31.16b, v31.16b",

        // 1st step: transpose 2x2 bit matrices
        "movi	v0.16b, #0x55",

        // xpstep	v16, v17, v0, 1
        "ushr v6.2d, v16.2d, #1",
        "shl v7.2d, v17.2d, #1",
        "bif v16.16b, v7.16b, v0.16b",
        "bit v17.16b, v6.16b, v0.16b",

        // xpstep	v18, v19, v0, 1
        "ushr v6.2d, v18.2d, #1",
        "shl v7.2d, v19.2d, #1",
        "bif v18.16b, v7.16b, v0.16b",
        "bit v19.16b, v6.16b, v0.16b",
        // xpstep	v20, v21, v0, 1
        "ushr v6.2d, v20.2d, #1",
        "shl v7.2d, v21.2d, #1",
        "bif v20.16b, v7.16b, v0.16b",
        "bit v21.16b, v6.16b, v0.16b",
        // xpstep	v22, v23, v0, 1
        "ushr v6.2d, v22.2d, #1",
        "shl v7.2d, v23.2d, #1",
        "bif v22.16b, v7.16b, v0.16b",
        "bit v23.16b, v6.16b, v0.16b",
        // xpstep	v24, v25, v0, 1
        "ushr v6.2d, v24.2d, #1",
        "shl v7.2d, v25.2d, #1",
        "bif v24.16b, v7.16b, v0.16b",
        "bit v25.16b, v6.16b, v0.16b",
        // xpstep	v26, v27, v0, 1
        "ushr v6.2d, v26.2d, #1",
        "shl v7.2d, v27.2d, #1",
        "bif v26.16b, v7.16b, v0.16b",
        "bit v27.16b, v6.16b, v0.16b",
        // xpstep	v28, v29, v0, 1
        "ushr v6.2d, v28.2d, #1",
        "shl v7.2d, v29.2d, #1",
        "bif v28.16b, v7.16b, v0.16b",
        "bit v29.16b, v6.16b, v0.16b",
        // xpstep	v30, v31, v0, 1
        "ushr v6.2d, v30.2d, #1",
        "shl v7.2d, v31.2d, #1",
        "bif v30.16b, v7.16b, v0.16b",
        "bit v31.16b, v6.16b, v0.16b",

        // 	# 2nd step: transpose 4x4 bit matrices
        "movi	v0.16b, #0x33",
        // 	xpstep	v16, v18, v0, 2
        "ushr v6.2d, v16.2d, #2",
        "shl v7.2d, v18.2d, #2",
        "bif v16.16b, v7.16b, v0.16b",
        "bit v18.16b, v6.16b, v0.16b",
        // 	xpstep	v17, v19, v0, 2
        "ushr v6.2d, v17.2d, #2",
        "shl v7.2d, v19.2d, #2",
        "bif v17.16b, v7.16b, v0.16b",
        "bit v19.16b, v6.16b, v0.16b",
        // 	xpstep	v20, v22, v0, 2
        "ushr v6.2d, v20.2d, #2",
        "shl v7.2d, v22.2d, #2",
        "bif v20.16b, v7.16b, v0.16b",
        "bit v22.16b, v6.16b, v0.16b",
        // 	xpstep	v21, v23, v0, 2
        "ushr v6.2d, v21.2d, #2",
        "shl v7.2d, v23.2d, #2",
        "bif v21.16b, v7.16b, v0.16b",
        "bit v23.16b, v6.16b, v0.16b",
        // 	xpstep	v24, v26, v0, 2
        "ushr v6.2d, v24.2d, #2",
        "shl v7.2d, v26.2d, #2",
        "bif v24.16b, v7.16b, v0.16b",
        "bit v26.16b, v6.16b, v0.16b",
        // 	xpstep	v25, v27, v0, 2
        "ushr v6.2d, v25.2d, #2",
        "shl v7.2d, v27.2d, #2",
        "bif v25.16b, v7.16b, v0.16b",
        "bit v27.16b, v6.16b, v0.16b",
        // 	xpstep	v28, v30, v0, 2
        "ushr v6.2d, v28.2d, #2",
        "shl v7.2d, v30.2d, #2",
        "bif v28.16b, v7.16b, v0.16b",
        "bit v30.16b, v6.16b, v0.16b",
        // 	xpstep	v29, v31, v0, 2
        "ushr v6.2d, v29.2d, #2",
        "shl v7.2d, v31.2d, #2",
        "bif v29.16b, v7.16b, v0.16b",
        "bit v31.16b, v6.16b, v0.16b",

        // immediate step: zip vectors to change
        // colocation.  As a side effect, every other
        // vector is temporarily relocated to the v0..v7
        // register range
        "zip1	v0.2d,  v16.2d, v17.2d",
        "zip2	v17.2d, v16.2d, v17.2d",
        "zip1	v1.2d,  v18.2d, v19.2d",
        "zip2	v19.2d, v18.2d, v19.2d",
        "zip1	v2.2d,  v20.2d, v21.2d",
        "zip2	v21.2d, v20.2d, v21.2d",
        "zip1	v3.2d,  v22.2d, v23.2d",
        "zip2	v23.2d, v22.2d, v23.2d",
        "zip1	v4.2d,	v24.2d, v25.2d",
        "zip2	v25.2d, v24.2d, v25.2d",
        "zip1	v5.2d,  v26.2d, v27.2d",
        "zip2	v27.2d, v26.2d, v27.2d",
        "zip1	v6.2d,  v28.2d, v29.2d",
        "zip2	v29.2d, v28.2d, v29.2d",
        "zip1	v7.2d,  v30.2d, v31.2d",
        "zip2	v31.2d, v30.2d, v31.2d",

        // macro for the 3rd transposition step
        // swap low 4 bit of each hi member with
        // high 4 bit of each orig member.  The orig
        // members are copied to lo in the process.
        //
        // 3rd step: transpose 8x8 bit matrices
        // special code is needed here since we need to
        // swap row n row line n+4, but these rows are
        // always colocated in the same register
        // xpstep3	v16, v17, v0
        "mov v16.2d, v0.2d",
        "sli v16.16b, v17.16b, #4",
        "sri v17.16b, v0.16b, #4",
        // 	xpstep3	v18, v19, v1
        "mov v18.2d, v1.2d",
        "sli v18.16b, v19.16b, #4",
        "sri v19.16b, v1.16b, #4",
        // 	xpstep3	v20, v21, v2
        "mov v20.2d, v2.2d",
        "sli v20.16b, v21.16b, #4",
        "sri v21.16b, v2.16b, #4",
        // 	xpstep3 v22, v23, v3
        "mov v22.2d, v3.2d",
        "sli v22.16b, v23.16b, #4",
        "sri v23.16b, v3.16b, #4",
        // 	xpstep3	v24, v25, v4
        "mov v24.2d, v4.2d",
        "sli v24.16b, v25.16b, #4",
        "sri v25.16b, v4.16b, #4",
        // 	xpstep3	v26, v27, v5
        "mov v26.2d, v5.2d",
        "sli v26.16b, v27.16b, #4",
        "sri v27.16b, v5.16b, #4",
        // 	xpstep3	v28, v29, v6
        "mov v28.2d, v6.2d",
        "sli v28.16b, v29.16b, #4",
        "sri v29.16b, v6.16b, #4",
        // 	xpstep3	v30, v31, v7
        "mov v30.2d, v7.2d",
        "sli v30.16b, v31.16b, #4",
        "sri v31.16b, v7.16b, #4",

        // registers now hold
        // v16: { 0,  1}  v17: { 4,  5}  v18: { 2,  3}  v19: { 6,  7}
        // v20: { 8,  9}  v21: {12, 13}  v22: {10, 11}  v23: {14, 15}
        // v24: {16, 17}  v25: {20, 21}  v26: {18, 19}  v27: {22, 23}
        // v28: {24, 25}  v29: {28, 29}  v30: {26, 27}  v31: {30, 31}

        // 4th step: transpose 16x16 bit matrices
        // this step again moves half the registers to v0--v7
        "trn1	v0.16b,  v16.16b, v20.16b",
        "trn2	v20.16b, v16.16b, v20.16b",
        "trn1	v1.16b,  v17.16b, v21.16b",
        "trn2	v21.16b, v17.16b, v21.16b",
        "trn1	v2.16b,  v18.16b, v22.16b",
        "trn2	v22.16b, v18.16b, v22.16b",
        "trn1	v3.16b,  v19.16b, v23.16b",
        "trn2	v23.16b, v19.16b, v23.16b",
        "trn1	v4.16b,	 v24.16b, v28.16b",
        "trn2	v28.16b, v24.16b, v28.16b",
        "trn1	v5.16b,  v25.16b, v29.16b",
        "trn2	v29.16b, v25.16b, v29.16b",
        "trn1	v6.16b,  v26.16b, v30.16b",
        "trn2	v30.16b, v26.16b, v30.16b",
        "trn1	v7.16b,	 v27.16b, v31.16b",
        "trn2	v31.16b, v27.16b, v31.16b",

        // 5th step: transpose 32x32 bit matrices
        // while we are at it, shuffle the order of
        // entries such that they are in order
        "trn1	v16.8h, v0.8h, v4.8h",
        "trn2	v24.8h, v0.8h, v4.8h",
        "trn1	v18.8h, v1.8h, v5.8h",
        "trn2	v26.8h, v1.8h, v5.8h",
        "trn1	v17.8h, v2.8h, v6.8h",
        "trn2	v25.8h, v2.8h, v6.8h",
        "trn1	v19.8h, v3.8h, v7.8h",
        "trn2	v27.8h, v3.8h, v7.8h",

        "trn1	v0.8h, v20.8h, v28.8h",
        "trn2	v4.8h, v20.8h, v28.8h",
        "trn1	v2.8h, v21.8h, v29.8h",
        "trn2	v6.8h, v21.8h, v29.8h",
        "trn1	v1.8h, v22.8h, v30.8h",
        "trn2	v5.8h, v22.8h, v30.8h",
        "trn1	v3.8h, v23.8h, v31.8h",
        "trn2	v7.8h, v23.8h, v31.8h",

        // now deposit the partially transposed matrix
        "st1	{{ v16.2d, v17.2d, v18.2d, v19.2d }}, [{save_arr}], #64",
        "st1	{{ v0.2d, v1.2d, v2.2d, v3.2d }}, [{save_arr}], #64",
        "st1	{{ v24.2d, v25.2d, v26.2d, v27.2d }}, [{save_arr}], #64",
        "st1	{{ v4.2d, v5.2d, v6.2d, v7.2d }}, [{save_arr}], #64",
        "ret",

        // xpose_final
        "2:",
        "ld1	{{ v16.2d, v17.2d, v18.2d, v19.2d }}, [{t2}], #64",
        "ld1	{{ v24.2d, v25.2d, v26.2d, v27.2d }}, [{t3}], #64",
        "ld1	{{ v20.2d, v21.2d, v22.2d, v23.2d }}, [{t2}], #64",
        "ld1	{{ v28.2d, v29.2d, v30.2d, v31.2d }}, [{t3}], #64",

        "trn1	v0.4s, v16.4s, v24.4s",
        "trn2	v4.4s, v16.4s, v24.4s",
        "trn1	v1.4s, v17.4s, v25.4s",
        "trn2	v5.4s, v17.4s, v25.4s",
        "trn1	v2.4s, v18.4s, v26.4s",
        "trn2	v6.4s, v18.4s, v26.4s",
        "trn1	v3.4s, v19.4s, v27.4s",
        "trn2	v7.4s, v19.4s, v27.4s",

        "trn1	v16.4s, v20.4s, v28.4s",
        "trn2	v24.4s, v20.4s, v28.4s",
        "trn1	v17.4s, v21.4s, v29.4s",
        "trn2	v25.4s, v21.4s, v29.4s",
        "trn1	v18.4s, v22.4s, v30.4s",
        "trn2	v26.4s, v22.4s, v30.4s",
        "trn1	v19.4s, v23.4s, v31.4s",
        "trn2	v27.4s, v23.4s, v31.4s",

        "st1 {{ v0.2d, v1.2d, v2.2d, v3.2d }}, [{t5}], #64",
        "st1 {{ v4.2d, v5.2d, v6.2d, v7.2d }}, [{t6}], #64",
        "st1 {{ v16.2d, v17.2d, v18.2d, v19.2d }}, [{t5}], #64",
        "st1 {{ v24.2d, v25.2d, v26.2d, v27.2d }}, [{t6}], #64",
        "ret",
        "3:",

        arr = in(reg) cols,
        save_lr = out(reg) _,
        save_arr = out(reg) _,
        t2 = out(reg) _,
        t3 = out(reg) _,
        t5 = out(reg) _,
        t6 = out(reg) _,
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
        )
    }
}
