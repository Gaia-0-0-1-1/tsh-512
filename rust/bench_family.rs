//! bench_family.rs — native economy re-bench (closes the seq-85 open item).
//!
//! Measures 1KB-input throughput of the family constructions (T1/T2/T3/T4,
//! via #[path] include of trit_family_v2.rs) and the founding artifacts
//! (tsh_512/pdh_512, via #[path] includes of tsh.rs / pdh.rs), std-only,
//! Instant-based: 20 warmup iterations, then 3 timed repeats per
//! construction with iteration counts chosen so each repeat runs > 0.5 s.
//! Reports MB/s as the mean of the 3 repeats (min/max also shown).

#[path = "trit_family_v2.rs"]
mod family;

#[path = "tsh.rs"]
mod tsh;

#[path = "pdh.rs"]
mod pdh;

use std::time::Instant;

const INPUT_KB: usize = 1024;
const WARMUP: usize = 20;
const REPEATS: usize = 3;

fn make_input() -> Vec<u8> {
    // deterministic 1KB pattern
    (0..INPUT_KB).map(|i| (i * 31 + 7) as u8).collect()
}

/// Run one bench: warmup, then REPEATS timed runs of `iters` calls.
/// Returns MB/s per repeat.
fn bench<F: FnMut(&[u8])>(name: &str, iters: usize, mut f: F) -> Vec<f64> {
    let input = make_input();
    for _ in 0..WARMUP {
        f(&input);
    }
    let mut mbps = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t = Instant::now();
        for _ in 0..iters {
            f(&input);
        }
        let dt = t.elapsed().as_secs_f64();
        mbps.push((iters * INPUT_KB) as f64 / 1.0e6 / dt);
    }
    let mean: f64 = mbps.iter().sum::<f64>() / mbps.len() as f64;
    let min = mbps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = mbps.iter().cloned().fold(0.0, f64::max);
    println!(
        "{:<10} iters={:>5}  mean={:>9.3} MB/s  (min {:.3}, max {:.3})",
        name, iters, mean, min, max
    );
    mbps
}

fn main() {
    // sanity: all constructions run and produce stable-size output
    let input = make_input();
    let t1 = family::t1_hash(&input, 0);
    let t2 = family::t2_hash(&input, 0);
    let t3 = family::t3_hash(&input, 0);
    let t4 = family::t4_hash(&input, 0);
    let ts = tsh::tsh_512(&input);
    let pd = pdh::pdh_512(&input);
    assert_eq!(t1.len(), 81);
    assert_eq!(t2.len(), 81);
    assert_eq!(t3.len(), 81);
    assert_eq!(t4.len(), 81);
    assert_eq!(ts.len(), 64);
    assert_eq!(pd.len(), 64);
    // determinism check (instrument self-check, LEARNINGS 3.1 discipline)
    assert_eq!(family::t1_hash(&input, 0), t1);
    assert_eq!(family::t4_hash(&input, 0), t4);
    assert_eq!(tsh::tsh_512(&input), ts);
    assert_eq!(pdh::pdh_512(&input), pd);

    println!("native economy bench: 1KB input, {} warmup, {} repeats", WARMUP, REPEATS);

    // iteration counts tuned so each repeat exceeds ~0.5s on this machine
    let t1_m = bench("T1", 200, |d| {
        std::hint::black_box(family::t1_hash(d, 0));
    });
    let t2_m = bench("T2", 2000, |d| {
        std::hint::black_box(family::t2_hash(d, 0));
    });
    let t3_m = bench("T3", 2000, |d| {
        std::hint::black_box(family::t3_hash(d, 0));
    });
    let t4_m = bench("T4", 20000, |d| {
        std::hint::black_box(family::t4_hash(d, 0));
    });
    let ts_m = bench("TSH-512", 20000, |d| {
        std::hint::black_box(tsh::tsh_512(d));
    });
    let pd_m = bench("PDH-512", 20000, |d| {
        std::hint::black_box(pdh::pdh_512(d));
    });

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (t1, t2, t3, t4) = (mean(&t1_m), mean(&t2_m), mean(&t3_m), mean(&t4_m));
    let (ts, pd) = (mean(&ts_m), mean(&pd_m));
    println!("\nranking (ternary family): T4 {:.3} > T2 {:.3} > T1 {:.3}  (T3 {:.3})", t4, t2, t1, t3);
    println!("ratios: T4/T1 = {:.1}x, T4/T2 = {:.1}x, T2/T1 = {:.1}x", t4 / t1, t4 / t2, t2 / t1);
    println!("founding: TSH-512 {:.3} MB/s, PDH-512 {:.3} MB/s", ts, pd);
    let _ = (t3, ts, pd);
}
