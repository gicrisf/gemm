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

/// Simulates inference: multiple sequential GEMMs (layers) competing for cache.
/// Each layer has its own weight matrix, forcing cache eviction between layers.
/// Shape: (batch × hidden) × (hidden × hidden) → (batch × hidden)
fn bench_inference<T: One + Copy + 'static>(
    bencher: Bencher,
    list![par, num_layers, hidden, batch]: List![Parallelism, usize, usize, usize],
) {
    let reg = 64 / core::mem::size_of::<T>();

    // Create weight matrices for each layer (these compete for cache)
    let weights: Vec<_> = (0..num_layers)
        .map(|_| make_data::<T>(Layout::Col, hidden, hidden, reg))
        .collect();

    // Activation buffers (ping-pong between two)
    let (act_rs, act_cs, mut act_a) = make_data::<T>(Layout::Col, batch, hidden, reg);
    let (_, _, mut act_b) = make_data::<T>(Layout::Col, batch, hidden, reg);

    // Zero-fill activation buffers
    act_a.fill(unsafe { core::mem::zeroed() });
    act_b.fill(unsafe { core::mem::zeroed() });

    bencher.bench(|| {
        let mut src = &mut act_a;
        let mut dst = &mut act_b;

        for (w_rs, w_cs, w) in &weights {
            unsafe {
                // dst = src × W  (batch×hidden) × (hidden×hidden) → (batch×hidden)
                gemm(
                    batch,          // m: batch size
                    hidden,         // n: output features
                    hidden,         // k: input features
                    dst.as_mut_ptr(),
                    act_cs,
                    act_rs,
                    false,          // don't read dst (overwrite)
                    src.as_ptr(),
                    act_cs,
                    act_rs,
                    w.as_ptr(),
                    *w_cs,
                    *w_rs,
                    T::one(),       // alpha = 1
                    T::one(),       // beta = 1 (ignored since read_dst=false)
                    false,
                    false,
                    false,
                    par,
                );
            }
            // Swap buffers for next layer
            core::mem::swap(&mut src, &mut dst);
        }
    })
}

fn inference_args() -> Vec<List![Parallelism, usize, usize, usize]> {
    itertools::iproduct!(
        [Parallelism::Rayon(0), Parallelism::None],
        [12, 24],                          // num_layers (like BERT-base, BERT-large)
        [768, 1024, 2048],                 // hidden size
        [1, 8, 32]                         // batch size
    )
    .map(|(par, layers, hidden, batch)| list![par, layers, hidden, batch])
    .collect()
}

/// Memory-bandwidth-bound benchmark: small batch × large weight matrices.
/// This simulates real decoder inference where weights dominate and batch is small.
/// Shape: [seq_len, hidden] × [hidden, ffn_dim] where ffn_dim = 4 * hidden
fn bench_membw<T: One + Copy + 'static>(
    bencher: Bencher,
    list![par, seq_len, hidden]: List![Parallelism, usize, usize],
) {
    let reg = 64 / core::mem::size_of::<T>();
    let ffn_dim = 4 * hidden;  // typical FFN expansion

    // FFN weights: up projection and down projection
    let (w_up_rs, w_up_cs, w_up) = make_data::<T>(Layout::Col, hidden, ffn_dim, reg);
    let (w_down_rs, w_down_cs, w_down) = make_data::<T>(Layout::Col, ffn_dim, hidden, reg);

    // Activation buffers
    let (act_rs, act_cs, mut act) = make_data::<T>(Layout::Col, seq_len, hidden, reg);
    let (mid_rs, mid_cs, mut mid) = make_data::<T>(Layout::Col, seq_len, ffn_dim, reg);
    let (_, _, mut out) = make_data::<T>(Layout::Col, seq_len, hidden, reg);

    act.fill(unsafe { core::mem::zeroed() });
    mid.fill(unsafe { core::mem::zeroed() });
    out.fill(unsafe { core::mem::zeroed() });

    bencher.bench(|| {
        unsafe {
            // Up projection: [seq_len, hidden] × [hidden, ffn_dim] → [seq_len, ffn_dim]
            gemm(
                seq_len, ffn_dim, hidden,
                mid.as_mut_ptr(), mid_cs, mid_rs,
                false,
                act.as_ptr(), act_cs, act_rs,
                w_up.as_ptr(), w_up_cs, w_up_rs,
                T::one(), T::one(),
                false, false, false,
                par,
            );
            // Down projection: [seq_len, ffn_dim] × [ffn_dim, hidden] → [seq_len, hidden]
            gemm(
                seq_len, hidden, ffn_dim,
                out.as_mut_ptr(), act_cs, act_rs,
                false,
                mid.as_ptr(), mid_cs, mid_rs,
                w_down.as_ptr(), w_down_cs, w_down_rs,
                T::one(), T::one(),
                false, false, false,
                par,
            );
        }
    })
}

fn membw_args() -> Vec<List![Parallelism, usize, usize]> {
    itertools::iproduct!(
        [Parallelism::None],               // single-threaded to isolate memory bandwidth
        [1, 4, 16, 64, 128],               // seq_len (small = more memory-bound)
        [768, 2048, 4096]                  // hidden size
    )
    .map(|(par, seq_len, hidden)| list![par, seq_len, hidden])
    .collect()
}

/// The REAL comparison: bf16 GEMM vs what candle upstream did (full tensor upcast + f32 GEMM).
/// This shows why bf16 GEMM exists: it's faster than the naive upcast approach.
#[cfg(feature = "bf16")]
fn bench_bf16_gemm(
    bencher: Bencher,
    list![m, n, k]: List![usize, usize, usize],
) {
    let reg = 64 / core::mem::size_of::<bf16>();

    // BF16 input matrices (what you'd have if model stores bf16)
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

/// What candle upstream did: upcast entire bf16 tensor to f32, then run f32 GEMM.
/// This is the SLOW approach that bf16 GEMM replaces.
/// NOTE: This version pre-allocates f32 buffers (generous to upcast approach).
#[cfg(feature = "bf16")]
fn bench_upcast_then_f32(
    bencher: Bencher,
    list![m, n, k]: List![usize, usize, usize],
) {
    let reg_bf16 = 64 / core::mem::size_of::<bf16>();
    let reg_f32 = 64 / core::mem::size_of::<f32>();

    // BF16 input matrices (what you'd have if model stores bf16)
    let (_, _, lhs_bf16) = make_data::<bf16>(Layout::Col, m, k, reg_bf16);
    let (_, _, rhs_bf16) = make_data::<bf16>(Layout::Col, k, n, reg_bf16);

    // F32 output
    let (dst_rs, dst_cs, mut dst) = make_data::<f32>(Layout::Col, m, n, reg_f32);

    // Pre-allocate f32 buffers (generous: candle would allocate fresh each time)
    let (lhs_rs, lhs_cs, mut lhs_f32) = make_data::<f32>(Layout::Col, m, k, reg_f32);
    let (rhs_rs, rhs_cs, mut rhs_f32) = make_data::<f32>(Layout::Col, k, n, reg_f32);

    bencher.bench(|| {
        // Step 1: Upcast bf16 → f32 (what candle upstream did before every GEMM)
        for (dst, src) in lhs_f32.iter_mut().zip(lhs_bf16.iter()) {
            *dst = src.to_f32();
        }
        for (dst, src) in rhs_f32.iter_mut().zip(rhs_bf16.iter()) {
            *dst = src.to_f32();
        }

        // Step 2: Run f32 GEMM
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

/// Realistic version: allocate f32 buffers inside the loop (what candle actually did).
#[cfg(feature = "bf16")]
fn bench_upcast_then_f32_with_alloc(
    bencher: Bencher,
    list![m, n, k]: List![usize, usize, usize],
) {
    let reg_bf16 = 64 / core::mem::size_of::<bf16>();
    let reg_f32 = 64 / core::mem::size_of::<f32>();

    // BF16 input matrices
    let (_, _, lhs_bf16) = make_data::<bf16>(Layout::Col, m, k, reg_bf16);
    let (_, _, rhs_bf16) = make_data::<bf16>(Layout::Col, k, n, reg_bf16);

    // F32 output
    let (dst_rs, dst_cs, mut dst) = make_data::<f32>(Layout::Col, m, n, reg_f32);

    let lhs_len = m * k.next_multiple_of(reg_f32);
    let rhs_len = k * n.next_multiple_of(reg_f32);

    bencher.bench(|| {
        // Step 1: Allocate f32 buffers (what candle actually did)
        let mut lhs_f32: AVec<f32> = avec![0.0f32; lhs_len];
        let mut rhs_f32: AVec<f32> = avec![0.0f32; rhs_len];

        let lhs_rs = 1isize;
        let lhs_cs = m.next_multiple_of(reg_f32) as isize;
        let rhs_rs = 1isize;
        let rhs_cs = k.next_multiple_of(reg_f32) as isize;

        // Step 2: Upcast bf16 → f32
        for (dst, src) in lhs_f32.iter_mut().zip(lhs_bf16.iter()) {
            *dst = src.to_f32();
        }
        for (dst, src) in rhs_f32.iter_mut().zip(rhs_bf16.iter()) {
            *dst = src.to_f32();
        }

        // Step 3: Run f32 GEMM
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
fn bf16_comparison_args() -> Vec<List![usize, usize, usize]> {
    vec![
        // Typical decoder shapes: [seq_len, hidden] × [hidden, ffn]
        list![1, 3072, 768],      // seq=1, hidden=768, ffn=3072
        list![1, 8192, 2048],     // seq=1, hidden=2048, ffn=8192
        list![32, 3072, 768],     // seq=32
        list![32, 8192, 2048],
        list![128, 3072, 768],    // seq=128
        list![128, 8192, 2048],
        // Square matrices for comparison
        list![512, 512, 512],
        list![1024, 1024, 1024],
        list![2048, 2048, 2048],
    ]
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

fn main() -> std::io::Result<()> {
    let config = BenchConfig::from_args()?;

    gemm::set_wasm_simd128(true);

    let modifiers = [1];

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

    // Inference-like benchmarks: compare f32 vs bf16 under cache pressure
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_inference::<f32>, inference_args());
        bench.run().unwrap();
    }
    #[cfg(feature = "bf16")]
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_inference::<bf16>, inference_args());
        bench.run().unwrap();
    }

    // Memory-bandwidth-bound benchmarks: small batch, large weights
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_membw::<f32>, membw_args());
        bench.run().unwrap();
    }
    #[cfg(feature = "bf16")]
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_membw::<bf16>, membw_args());
        bench.run().unwrap();
    }

    // THE KEY COMPARISON: bf16 GEMM vs upcast-then-f32 (what candle upstream did)
    #[cfg(feature = "bf16")]
    {
        let mut bench = Bench::new(&config);
        bench.register(bench_bf16_gemm, bf16_comparison_args());
        bench.register(bench_upcast_then_f32, bf16_comparison_args());
        bench.register(bench_upcast_then_f32_with_alloc, bf16_comparison_args());
        bench.run().unwrap();
    }

    Ok(())
}
