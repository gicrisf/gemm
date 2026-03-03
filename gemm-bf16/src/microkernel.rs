// bf16 microkernels reuse gemm-f32's f32 microkernels directly.
// The x86 paths in gemm.rs import from gemm_f32::microkernel::*.
