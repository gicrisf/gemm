use crate::Parallelism;
use core::any::TypeId;
pub use gemm_common::prepack::{PackedRhsInfo, packed_rhs_size, packed_rhs_alignment};

#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;
#[cfg(feature = "f16")]
#[allow(non_camel_case_types)]
pub type f16 = gemm_f16::f16;
#[cfg(feature = "bf16")]
#[allow(non_camel_case_types)]
pub type bf16 = gemm_bf16::bf16;

unsafe fn gemm_dispatch<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    #[cfg(feature = "f16")]
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        return gemm_f16::gemm::f16::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut f16,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f16,
            lhs_cs,
            lhs_rs,
            rhs as *mut f16,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f16),
            *(&beta as *const T as *const f16),
            false,
            false,
            false,
            parallelism,
        );
    }

    #[cfg(feature = "bf16")]
    if TypeId::of::<T>() == TypeId::of::<bf16>() {
        return gemm_bf16::gemm::bf16::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut bf16,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut bf16,
            lhs_cs,
            lhs_rs,
            rhs as *mut bf16,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const bf16),
            *(&beta as *const T as *const bf16),
            false,
            false,
            false,
            parallelism,
        );
    }

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        gemm_f64::gemm::f64::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut f64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f64,
            lhs_cs,
            lhs_rs,
            rhs as *mut f64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f64),
            *(&beta as *const T as *const f64),
            false,
            false,
            false,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        gemm_f32::gemm::f32::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut f32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f32,
            lhs_cs,
            lhs_rs,
            rhs as *mut f32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f32),
            *(&beta as *const T as *const f32),
            false,
            false,
            false,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c64>() {
        gemm_c64::gemm::f64::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut c64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c64,
            lhs_cs,
            lhs_rs,
            rhs as *mut c64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c64),
            *(&beta as *const T as *const c64),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c32>() {
        gemm_c32::gemm::f32::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut c32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c32,
            lhs_cs,
            lhs_rs,
            rhs as *mut c32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c32),
            *(&beta as *const T as *const c32),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
        )
    } else {
        panic!();
    }
}

/// dst := alpha×dst + beta×lhs×rhs
///
/// # Panics
///
/// Panics if `T` is not `f32`, `f64`, `gemm::f16`, `gemm::c32`, or `gemm::c64`.
pub unsafe fn gemm<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    mut dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    // we want to transpose if the destination is column-oriented, since the microkernel prefers
    // column major matrices.
    let do_transpose = dst_cs.abs() < dst_rs.abs();

    let (
        m,
        n,
        mut dst_cs,
        mut dst_rs,
        mut lhs,
        mut lhs_cs,
        mut lhs_rs,
        mut rhs,
        mut rhs_cs,
        mut rhs_rs,
        conj_lhs,
        conj_rhs,
    ) = if do_transpose {
        (
            n, m, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs, conj_rhs, conj_lhs,
        )
    } else {
        (
            m, n, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, conj_lhs, conj_rhs,
        )
    };

    if dst_rs < 0 && m > 0 {
        dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
        dst_rs = -dst_rs;
        lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
        lhs_rs = -lhs_rs;
    }

    if dst_cs < 0 && n > 0 {
        dst = dst.wrapping_offset((n - 1) as isize * dst_cs);
        dst_cs = -dst_cs;
        rhs = rhs.wrapping_offset((n - 1) as isize * rhs_cs);
        rhs_cs = -rhs_cs;
    }

    if lhs_cs < 0 && k > 0 {
        lhs = lhs.wrapping_offset((k - 1) as isize * lhs_cs);
        lhs_cs = -lhs_cs;
        rhs = rhs.wrapping_offset((k - 1) as isize * rhs_rs);
        rhs_rs = -rhs_rs;
    }

    gemm_dispatch(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
    )
}

/// Compute the required buffer size for pre-packing an f32 RHS matrix.
///
/// # Arguments
///
/// * `k` - The K dimension (depth/inner dimension) of the RHS matrix
/// * `n` - The N dimension (columns) of the RHS matrix
///
/// # Returns
///
/// A tuple of `(PackedRhsInfo, size_in_bytes)` where `size_in_bytes` is the
/// number of bytes needed for the packed buffer.
#[inline]
pub fn packed_rhs_size_f32(k: usize, n: usize) -> (PackedRhsInfo, usize) {
    // Get the correct MR and NR for the current architecture
    // These must match what the prepack function will use
    let (mr, nr) = get_f32_microkernel_params();
    let (info, size_elements) = packed_rhs_size(k, n, mr, nr, core::mem::size_of::<f32>());
    (info, size_elements * core::mem::size_of::<f32>())
}

/// Returns (MR, NR) for the f32 microkernel on the current architecture.
#[inline]
fn get_f32_microkernel_params() -> (usize, usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "x86-v4")]
        if gemm_common::feature_detected!("avx512f") {
            // AVX-512: MR=64 (4*16), NR=6
            return (64, 6);
        }
        if gemm_common::feature_detected!("fma") {
            // FMA/AVX2: MR=16 (2*8), NR=6
            return (16, 6);
        }
        // Scalar fallback: MR=2, NR=4
        return (2, 4);
    }

    #[cfg(target_arch = "aarch64")]
    {
        if gemm_common::feature_detected!("neon") {
            // NEON: MR=16 (4*4), NR=4
            return (16, 4);
        }
        // Scalar fallback
        return (2, 4);
    }

    #[cfg(target_arch = "wasm32")]
    {
        if gemm_common::feature_detected!("simd128") {
            // WASM SIMD128: MR=12 (3*4), NR=4
            return (12, 4);
        }
        return (2, 4);
    }

    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32",
    )))]
    {
        // Generic scalar: MR=2, NR=4
        (2, 4)
    }
}

/// Pre-pack an f32 RHS matrix for later use with `gemm_prepacked_rhs_f32`.
///
/// This function packs the RHS (weight) matrix into a format optimized for
/// repeated GEMM operations. The packed buffer can be reused across multiple
/// calls to `gemm_prepacked_rhs_f32` with different LHS matrices.
///
/// # Safety
///
/// - `dst` must be valid for writes of at least `info.packed_size * sizeof(f32)` bytes
/// - `dst` should be aligned to at least `packed_rhs_alignment()` bytes
/// - `src` must be valid for reads according to `k`, `n`, and stride parameters
/// - `info` must have been computed by `packed_rhs_size_f32` with the same `k` and `n`
///
/// # Arguments
///
/// * `dst` - Pointer to the destination buffer for packed data
/// * `src` - Pointer to the source RHS matrix
/// * `k` - The K dimension (rows of RHS, must match `info.k`)
/// * `n` - The N dimension (columns of RHS, must match `info.n`)
/// * `src_rs` - Row stride of source matrix (stride between rows)
/// * `src_cs` - Column stride of source matrix (stride between columns)
/// * `info` - Packing info from `packed_rhs_size_f32`
#[inline]
pub unsafe fn prepack_rhs_f32(
    dst: *mut f32,
    src: *const f32,
    src_rs: isize,
    src_cs: isize,
    info: &PackedRhsInfo,
) {
    debug_assert_eq!(info.k, info.k);
    debug_assert_eq!(info.n, info.n);
    gemm_f32::gemm::f32::prepack::get_prepack_rhs_fn()(
        info,
        dst,
        src,
        src_rs,
        src_cs,
    )
}

/// Perform GEMM using a pre-packed f32 RHS matrix.
///
/// dst := alpha * dst + beta * lhs * packed_rhs
///
/// This function is optimized for the case where the same RHS matrix is used
/// repeatedly with different LHS matrices (e.g., weight matrices in neural networks).
///
/// # Safety
///
/// - All pointer parameters must be valid for their respective dimensions
/// - `packed_rhs` must have been packed using `prepack_rhs_f32` with matching parameters
/// - `packed_rhs_info` must describe the packed buffer accurately
///
/// # Arguments
///
/// * `m` - Number of rows in the output matrix
/// * `n` - Number of columns in the output matrix (must match `packed_rhs_info.n`)
/// * `k` - Inner dimension (must match `packed_rhs_info.k`)
/// * `dst` - Pointer to the output matrix
/// * `dst_cs` - Column stride of output matrix
/// * `dst_rs` - Row stride of output matrix
/// * `read_dst` - If true, the existing content of dst is read and scaled by alpha
/// * `lhs` - Pointer to the left-hand side matrix
/// * `lhs_cs` - Column stride of LHS matrix
/// * `lhs_rs` - Row stride of LHS matrix
/// * `packed_rhs` - Pointer to the pre-packed RHS matrix
/// * `packed_rhs_info` - Info about the packed RHS matrix
/// * `alpha` - Scalar multiplier for existing dst content
/// * `beta` - Scalar multiplier for the matrix product
/// * `parallelism` - Threading configuration
pub unsafe fn gemm_prepacked_rhs_f32(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    packed_rhs: *const f32,
    packed_rhs_info: &PackedRhsInfo,
    alpha: f32,
    beta: f32,
    parallelism: Parallelism,
) {
    debug_assert_eq!(packed_rhs_info.k, k);
    debug_assert_eq!(packed_rhs_info.n, n);

    gemm_f32::gemm::f32::prepack::get_gemm_prepacked_rhs_fn()(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        packed_rhs,
        packed_rhs_info,
        alpha,
        beta,
        parallelism,
    )
}

#[inline(never)]
#[cfg(test)]
pub unsafe fn gemm_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
) where
    T: num_traits::Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    (0..m).for_each(|row| {
        (0..n).for_each(|col| {
            let mut accum = <T as num_traits::Zero>::zero();
            for depth in 0..k {
                let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);

                let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

                accum = &accum + &(lhs * rhs);
            }
            accum = &accum * &beta;

            let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                accum = &accum + &(&alpha * &*dst);
            }
            *dst = accum
        });
    });
    return;
}

#[inline(never)]
#[cfg(test)]
pub(crate) unsafe fn gemm_cplx_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut num_complex::Complex<T>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const num_complex::Complex<T>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const num_complex::Complex<T>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: num_complex::Complex<T>,
    beta: num_complex::Complex<T>,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) where
    T: num_traits::Zero
        + Send
        + Sync
        + Copy
        + num_traits::Num
        + core::ops::Neg<Output = T>
        + core::fmt::Debug,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Sub<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    (0..m).for_each(|row| {
        (0..n).for_each(|col| {
            let mut accum = <num_complex::Complex<T> as num_traits::Zero>::zero();
            for depth in 0..k {
                let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);
                let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

                match (conj_lhs, conj_rhs) {
                    (true, true) => accum = &accum + &(lhs.conj() * rhs.conj()),
                    (true, false) => accum = &accum + &(lhs.conj() * rhs),
                    (false, true) => accum = &accum + &(lhs * rhs.conj()),
                    (false, false) => accum = &accum + &(lhs * rhs),
                }
            }
            accum = &accum * &beta;

            let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                match conj_dst {
                    true => accum = &accum + &(&alpha * (*dst).conj()),
                    false => accum = &accum + &(&alpha * &*dst),
                }
            }
            *dst = accum
        });
    });
    return;
}
