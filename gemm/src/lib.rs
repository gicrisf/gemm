#![cfg_attr(not(feature = "std"), no_std)]
#![warn(rust_2018_idioms)]

mod gemm;

#[cfg(feature = "f16")]
pub use crate::gemm::f16;
#[cfg(feature = "bf16")]
pub use crate::gemm::bf16;
pub use crate::gemm::{c32, c64, gemm};
pub use crate::gemm::{
    PackedRhsInfo, packed_rhs_size, packed_rhs_alignment,
    packed_rhs_size_f32, prepack_rhs_f32, gemm_prepacked_rhs_f32,
};
pub use gemm_common::Parallelism;

pub use gemm_common::gemm::{
    get_lhs_packing_threshold_multi_thread, get_lhs_packing_threshold_single_thread,
    get_rhs_packing_threshold, get_threading_threshold, set_lhs_packing_threshold_multi_thread,
    set_lhs_packing_threshold_single_thread, set_rhs_packing_threshold, set_threading_threshold,
    DEFAULT_LHS_PACKING_THRESHOLD_MULTI_THREAD, DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD,
    DEFAULT_RHS_PACKING_THRESHOLD, DEFAULT_THREADING_THRESHOLD,
};
pub use gemm_common::{get_wasm_simd128, set_wasm_simd128, DEFAULT_WASM_SIMD128};

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::{vec, vec::Vec};
    use num_traits::Float;

    #[test]
    fn test_gemm_f16() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((63, 2, 10));
        mnks.push((16, 2, 1));
        mnks.push((0, 0, 4));
        mnks.push((16, 1, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((64, 64, 0));
        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);

                        for colmajor in [true, false] {
                            let alpha = f16::from_f32(alpha);
                            let beta = f16::from_f32(beta);
                            let a_vec: Vec<f16> = (0..(m * k))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let b_vec: Vec<f16> = (0..(k * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut c_vec: Vec<f16> = (0..(m * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            let eps = f16::from_f32(1e-1);
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                let eps_rel = c.abs() * eps;
                                let eps_abs = eps;
                                let eps = if eps_rel > eps_abs { eps_rel } else { eps_abs };
                                assert_approx_eq::assert_approx_eq!(c, d, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn test_gemm_bf16() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((63, 2, 10));
        mnks.push((16, 2, 1));
        mnks.push((0, 0, 4));
        mnks.push((16, 1, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((64, 64, 0));
        mnks.push((256, 256, 256));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
            ] {
                for alpha in [0.0f32, 1.0, 2.3] {
                    for beta in [0.0f32, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);

                        for colmajor in [true, false] {
                            let alpha = bf16::from_f32(alpha);
                            let beta = bf16::from_f32(beta);
                            let a_vec: Vec<bf16> = (0..(m * k))
                                .map(|_| bf16::from_f32(rand::random()))
                                .collect();
                            let b_vec: Vec<bf16> = (0..(k * n))
                                .map(|_| bf16::from_f32(rand::random()))
                                .collect();
                            let mut c_vec: Vec<bf16> = (0..(m * n))
                                .map(|_| bf16::from_f32(rand::random()))
                                .collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            let eps = bf16::from_f32(1e-1);
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                use num_traits::Float;
                                let eps_rel = c.abs() * eps;
                                let eps_abs = eps;
                                let eps = if eps_rel > eps_abs { eps_rel } else { eps_abs };
                                assert_approx_eq::assert_approx_eq!(c, d, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f32() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f32> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f32> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f32> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f64() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f64> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f64> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f64> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx32() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c32::new(0.0, 0.0);
            let one = c32::new(1.0, 0.0);
            let arbitrary = c32::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f32> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f32> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f32> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx64() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c64::new(0.0, 0.0);
            let one = c64::new(1.0, 0.0);
            let arbitrary = c64::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f64> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f64> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f64> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_prepacked_f32() {
        use crate::{
            packed_rhs_size_f32, prepack_rhs_f32, gemm_prepacked_rhs_f32,
            packed_rhs_alignment,
        };

        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((64, 64, 64));
        mnks.push((63, 63, 63));
        mnks.push((128, 256, 64));
        mnks.push((256, 128, 64));
        mnks.push((16, 16, 1));
        mnks.push((16, 16, 2));
        mnks.push((1, 1, 1));
        mnks.push((1, 64, 64));
        mnks.push((64, 1, 64));
        mnks.push((64, 64, 1));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
            ] {
                for alpha in [0.0f32, 1.0, 2.3] {
                    for beta in [0.0f32, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);

                        // Create test matrices
                        let a_vec: Vec<f32> = (0..(m * k)).map(|_| rand::random()).collect();
                        let b_vec: Vec<f32> = (0..(k * n)).map(|_| rand::random()).collect();
                        let mut c_vec: Vec<f32> = (0..(m * n)).map(|_| rand::random()).collect();
                        let mut d_vec = c_vec.clone();

                        // Pre-pack RHS
                        let (info, packed_size_bytes) = packed_rhs_size_f32(k, n);
                        let align = packed_rhs_alignment();
                        let layout = std::alloc::Layout::from_size_align(packed_size_bytes, align).unwrap();
                        let packed_rhs = unsafe { std::alloc::alloc(layout) as *mut f32 };

                        unsafe {
                            prepack_rhs_f32(
                                packed_rhs,
                                b_vec.as_ptr(),
                                k as isize,  // row stride (column-major)
                                1,           // column stride
                                &info,
                            );

                            // Test prepacked GEMM
                            gemm_prepacked_rhs_f32(
                                m,
                                n,
                                k,
                                c_vec.as_mut_ptr(),
                                m as isize,  // column stride (column-major)
                                1,           // row stride
                                true,
                                a_vec.as_ptr(),
                                m as isize,  // column stride
                                1,           // row stride
                                packed_rhs,
                                &info,
                                alpha,
                                beta,
                                parallelism,
                            );

                            // Reference: regular GEMM
                            gemm::gemm(
                                m,
                                n,
                                k,
                                d_vec.as_mut_ptr(),
                                m as isize,
                                1,
                                true,
                                a_vec.as_ptr(),
                                m as isize,
                                1,
                                b_vec.as_ptr(),
                                k as isize,
                                1,
                                alpha,
                                beta,
                                false,
                                false,
                                false,
                                parallelism,
                            );

                            std::alloc::dealloc(packed_rhs as *mut u8, layout);
                        }

                        // Compare results
                        for (i, (c, d)) in c_vec.iter().zip(d_vec.iter()).enumerate() {
                            assert_approx_eq::assert_approx_eq!(c, d, 1e-4,
                                "mismatch at index {} for m={}, n={}, k={}, alpha={}, beta={}",
                                i, m, n, k, alpha, beta
                            );
                        }
                    }
                }
            }
        }
    }
}
