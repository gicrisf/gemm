//! BF16 (bfloat16) matrix multiplication support.
//!
//! This crate provides GEMM operations for BF16 matrices. Inputs and outputs are BF16,
//! but computation is performed in F32 for accuracy. The BF16↔F32 conversion happens
//! during the packing phase, avoiding the need to allocate temporary F32 tensors.
//!
//! # When to use BF16 GEMM
//!
//! BF16 GEMM is useful when your data is already stored in BF16 format. Instead of:
//!
//! 1. Allocating F32 tensors
//! 2. Upcasting BF16 → F32
//! 3. Running F32 GEMM
//!
//! You can directly run BF16 GEMM, which converts during packing and avoids the
//! allocation overhead.
//!
//! # Implementation
//!
//! - Reuses F32 microkernels from `gemm-f32`
//! - BF16 → F32 conversion: zero-extend (shift left 16 bits)
//! - F32 → BF16 conversion: truncate (keep upper 16 bits)
//! - Conversion is interleaved with packing to hide latency

#![cfg_attr(not(feature = "std"), no_std)]

pub mod gemm;
pub mod microkernel;
pub use half::bf16;

#[macro_use]
#[allow(unused_imports)]
extern crate gemm_common;
