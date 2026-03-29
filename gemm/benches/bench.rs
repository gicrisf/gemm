use aligned_vec::{avec, AVec};
use diol::prelude::*;
use gemm::*;
#[cfg(feature = "bf16")]
use gemm::bf16;
use num_traits::One;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Layout {
    Col,
    Row,
}

fn make_data<T: Copy + One>(
    layout: Layout,
    m: usize,
    n: usize,
    reg: usize,
) -> (isize, isize, AVec<T>) {
    let val = T::one();
    match layout {
        Layout::Col => (
            1,
            m.next_multiple_of(reg) as isize,
            avec![val; n * m.next_multiple_of(reg)],
        ),
        Layout::Row => (
            n.next_multiple_of(reg) as isize,
            1,
            avec![val; m * n.next_multiple_of(reg)],
        ),
    }
}

fn bench_gemm<T: One + Copy + 'static>(
    bencher: Bencher,
    list![par, dst, lhs, rhs, m, n, k]: List![
        Parallelism,
        Layout,
        Layout,
        Layout,
        usize,
        usize,
        usize
    ],
) {
    let reg = 64 / core::mem::size_of::<T>();

    let (dst_rs, dst_cs, mut dst) = make_data::<T>(dst, m, n, reg);
    let (lhs_rs, lhs_cs, mut lhs) = make_data::<T>(lhs, m, k, reg);
    let (rhs_rs, rhs_cs, mut rhs) = make_data::<T>(rhs, k, n, reg);

    lhs.fill(unsafe { core::mem::zeroed() });
    rhs.fill(unsafe { core::mem::zeroed() });
    dst.fill(unsafe { core::mem::zeroed() });

    bencher.bench(|| {
        unsafe {
            gemm(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                dst_cs,
                dst_rs,
                true,
                lhs.as_ptr(),
                lhs_cs,
                lhs_rs,
                rhs.as_ptr(),
                rhs_cs,
                rhs_rs,
                T::one(),
                T::one(),
                false,
                false,
                false,
                par,
            )
        };
    })
}

fn args() -> Vec<List![Parallelism, Layout, Layout, Layout, usize, usize, usize]> {
    use itertools::Itertools;
    let pow2 = |i| 1usize << i;
    let halfway = |i| 3usize << (i - 1);
    itertools::iproduct!(
        [].into_iter()
            .chain((5..13).map(pow2).map(|n| (n, n, n)))
            .chain((5..13).map(halfway).map(|n| (n, n, n)))
            .chain((5..13).map(halfway).map(|n| (16, 16, n)))
            .sorted_unstable(),
        [Parallelism::Rayon(0), Parallelism::None],
        [Layout::Col, Layout::Row],
        [Layout::Col, Layout::Row],
        [Layout::Col, Layout::Row]
    )
    .map(|((m, n, k), par, dst, lhs, rhs)| list![par, dst, lhs, rhs, m, n, k])
    .collect()
}

// =============================================================================
// BF16 comparison benchmarks
//
// These benchmarks compare mixed-precision bf16 GEMM against the naive approach
// of upcasting bf16 tensors to f32 before every matmul.
// =============================================================================

/// Mixed-precision BF16 GEMM: reads bf16, converts to f32 during packing, computes in f32.
/// Avoids allocating temporary f32 tensors.
#[cfg(feature = "bf16")]
fn bench_bf16_mixed(
    bencher: Bencher,
    list![m, n, k]: List![usize, usize, usize],
) {
    let reg = 64 / core::mem::size_of::<bf16>();

    let (lhs_rs, lhs_cs, lhs) = make_data::<bf16>(Layout::Col, m, k, reg);
    let (rhs_rs, rhs_cs, rhs) = make_data::<bf16>(Layout::Col, k, n, reg);
    let (dst_rs, dst_cs, mut dst) = make_data::<bf16>(Layout::Col, m, n, reg);

    bencher.bench(|| {
        unsafe {
            gemm(
                m, n, k,
                dst.as_mut_ptr(), dst_cs, dst_rs,
                false,
                lhs.as_ptr(), lhs_cs, lhs_rs,
                rhs.as_ptr(), rhs_cs, rhs_rs,
                bf16::ONE, bf16::ONE,
                false, false, false,
                Parallelism::None,
            );
        }
    });
}

/// Naive approach: allocate f32 tensors, upcast bf16→f32, then run f32 GEMM.
/// This is what frameworks like candle did before mixed-precision bf16 GEMM support.
#[cfg(feature = "bf16")]
fn bench_bf16_upcast(
    bencher: Bencher,
    list![m, n, k]: List![usize, usize, usize],
) {
    let reg_bf16 = 64 / core::mem::size_of::<bf16>();
    let reg_f32 = 64 / core::mem::size_of::<f32>();

    // BF16 source tensors (simulating bf16 model weights)
    let (_, _, lhs_bf16) = make_data::<bf16>(Layout::Col, m, k, reg_bf16);
    let (_, _, rhs_bf16) = make_data::<bf16>(Layout::Col, k, n, reg_bf16);

    // F32 output
    let (dst_rs, dst_cs, mut dst) = make_data::<f32>(Layout::Col, m, n, reg_f32);

    let lhs_len = m * k.next_multiple_of(reg_f32);
    let rhs_len = k * n.next_multiple_of(reg_f32);

    bencher.bench(|| {
        // Allocate f32 buffers (this is what the naive approach does)
        let mut lhs_f32: AVec<f32> = avec![0.0f32; lhs_len];
        let mut rhs_f32: AVec<f32> = avec![0.0f32; rhs_len];

        let lhs_rs = 1isize;
        let lhs_cs = m.next_multiple_of(reg_f32) as isize;
        let rhs_rs = 1isize;
        let rhs_cs = k.next_multiple_of(reg_f32) as isize;

        // Upcast bf16 → f32
        for (dst, src) in lhs_f32.iter_mut().zip(lhs_bf16.iter()) {
            *dst = src.to_f32();
        }
        for (dst, src) in rhs_f32.iter_mut().zip(rhs_bf16.iter()) {
            *dst = src.to_f32();
        }

        // Run f32 GEMM
        unsafe {
            gemm(
                m, n, k,
                dst.as_mut_ptr(), dst_cs, dst_rs,
                false,
                lhs_f32.as_ptr(), lhs_cs, lhs_rs,
                rhs_f32.as_ptr(), rhs_cs, rhs_rs,
                1.0f32, 1.0f32,
                false, false, false,
                Parallelism::None,
            );
        }
    });
}

#[cfg(feature = "bf16")]
fn bf16_args() -> Vec<List![usize, usize, usize]> {
    vec![
        // Tall-and-skinny shapes (small M, large K/N)
        // These are memory-bound and show the largest bf16 benefit
        list![1, 3072, 768],
        list![1, 8192, 2048],
        list![32, 3072, 768],
        list![32, 8192, 2048],
        list![128, 3072, 768],
        list![128, 8192, 2048],
        // Square matrices
        list![512, 512, 512],
        list![1024, 1024, 1024],
        list![2048, 2048, 2048],
    ]
}

fn main() -> std::io::Result<()> {
    let config = BenchConfig::from_args()?;

    gemm::set_wasm_simd128(true);

    let modifiers = [1];

    // Standard GEMM benchmarks for all types
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_gemm::<f32>, args());
        for modifier in modifiers {
            gemm::set_threading_threshold(gemm::DEFAULT_THREADING_THRESHOLD / modifier);
            bench.run().unwrap();
        }
    }
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_gemm::<f64>, args());
        for modifier in modifiers {
            gemm::set_threading_threshold(gemm::DEFAULT_THREADING_THRESHOLD / modifier);
            bench.run().unwrap();
        }
    }
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_gemm::<c32>, args());
        for modifier in modifiers {
            gemm::set_threading_threshold(gemm::DEFAULT_THREADING_THRESHOLD / modifier);
            bench.run().unwrap();
        }
    }
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_gemm::<c64>, args());
        for modifier in modifiers {
            gemm::set_threading_threshold(gemm::DEFAULT_THREADING_THRESHOLD / modifier);
            bench.run().unwrap();
        }
    }
    #[cfg(feature = "bf16")]
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_gemm::<bf16>, args());
        for modifier in modifiers {
            gemm::set_threading_threshold(gemm::DEFAULT_THREADING_THRESHOLD / modifier);
            bench.run().unwrap();
        }
    }

    // BF16 comparison: mixed-precision bf16 GEMM vs naive upcast approach
    #[cfg(feature = "bf16")]
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_bf16_mixed, bf16_args());
        bench.register(bench_bf16_upcast, bf16_args());
        bench.run().unwrap();
    }

    Ok(())
}
