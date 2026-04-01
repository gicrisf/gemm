//! Pre-packing support for RHS matrices.
//!
//! This module provides functionality to pre-pack weight matrices (RHS) once at model load time
//! and reuse them across multiple inference calls, optimizing repeated GEMM operations.

use crate::cache::{kernel_params, DivCeil, KernelParams};

/// Metadata stored with pre-packed RHS buffer.
///
/// This struct contains all the information needed to use a pre-packed RHS matrix
/// in subsequent GEMM operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedRhsInfo {
    /// Original K dimension (depth/inner dimension)
    pub k: usize,
    /// Original N dimension (columns)
    pub n: usize,
    /// Block size in the K dimension used during packing
    pub kc: usize,
    /// Register block size in the N dimension
    pub nr: usize,
    /// Total buffer size in elements of the packed type
    pub packed_size: usize,
}

/// Calculate required buffer size and info for pre-packing RHS.
///
/// This function computes the packing parameters based on the matrix dimensions
/// and cache characteristics. It uses a large `m_hint` to ensure `kc` is stable
/// for typical inference workloads.
///
/// # Arguments
///
/// * `k` - The K dimension (depth/inner dimension) of the RHS matrix
/// * `n` - The N dimension (columns) of the RHS matrix
/// * `mr` - The register block size in the M dimension (used to compute cache params)
/// * `nr` - The register block size in the N dimension
/// * `sizeof` - Size in bytes of the *packed* element type (e.g., 4 for f32)
///
/// # Returns
///
/// A tuple of `(PackedRhsInfo, size_in_elements)` where `size_in_elements` is the
/// number of elements (not bytes) needed for the packed buffer.
///
/// # Note
///
/// For bf16 inputs, the packed buffer stores f32 values, so `sizeof` should be 4
/// and the returned size is in f32 elements.
pub fn packed_rhs_size(k: usize, n: usize, mr: usize, nr: usize, sizeof: usize) -> (PackedRhsInfo, usize) {
    // Use large m_hint so kc is stable across different inference batch sizes
    let m_hint = 1024;
    let KernelParams { kc, .. } = kernel_params(m_hint, n, k, mr, nr, sizeof);

    let n_panels = n.msrv_div_ceil(nr);
    let k_panels = k.msrv_div_ceil(kc);
    let packed_rhs_stride = kc * nr;
    let packed_size = k_panels * n_panels * packed_rhs_stride;

    (PackedRhsInfo { k, n, kc, nr, packed_size }, packed_size)
}

/// Calculate the required alignment for pre-packed buffers.
///
/// Returns the recommended alignment in bytes for pre-packed RHS buffers.
/// This should be used when allocating memory for packed matrices.
#[inline]
pub const fn packed_rhs_alignment() -> usize {
    crate::gemm::CACHELINE_ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_rhs_size_basic() {
        // Test with typical f32 GEMM parameters
        let (info, size) = packed_rhs_size(256, 256, 24, 4, 4);
        assert_eq!(info.k, 256);
        assert_eq!(info.n, 256);
        assert!(info.kc > 0);
        assert_eq!(info.nr, 4);
        assert!(size > 0);
        assert_eq!(size, info.packed_size);
    }

    #[test]
    fn test_packed_rhs_size_small() {
        // Test with small dimensions
        let (info, size) = packed_rhs_size(4, 4, 8, 4, 4);
        assert_eq!(info.k, 4);
        assert_eq!(info.n, 4);
        assert!(size > 0);
    }

    #[test]
    fn test_packed_rhs_size_non_multiple() {
        // Test with dimensions that aren't multiples of nr
        let (info, size) = packed_rhs_size(100, 13, 24, 4, 4);
        assert_eq!(info.k, 100);
        assert_eq!(info.n, 13);
        // n_panels should round up: ceil(13/4) = 4
        assert!(size > 0);
    }
}
