pub mod f32 {
    type T = f32;
    gemm_common::gemm_def!(f32, 2);
    gemm_common::gemm_prepack_def!(f32, 2);
}
