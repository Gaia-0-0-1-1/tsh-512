//! verify_vectors.rs — Phase-0 cross-language truth harness (std-only).
//!
//! Line protocol on stdin/stdout: each line "<alg> <input-hex>" is
//! answered with one line "<alg> <digest-hex>", where alg is `tsh512`
//! (rust/tsh.rs) or `pdh512` (rust/pdh.rs). Driven by
//! tools/verify_rust.py against vectors/tsh512_pdh512.json so Python
//! reference and Rust core consume the SAME frozen fixture — canon law:
//! one serialization, refuse-unfaithful.
//!
//! Build: rustc --edition 2021 -O rust/verify_vectors.rs
//!   -o rust/target/verify_vectors.exe

#[path = "tsh.rs"]
mod tsh;
#[path = "pdh.rs"]
mod pdh;

use std::io::{self, BufRead, Write};

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        // NB: do NOT trim the whole line — an empty input hex arrives
        // as "tsh512 " and the trailing space IS the separator.
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let (alg, hexin) = line
            .split_once(' ')
            .expect("protocol: '<alg> <input-hex>' per line");
        let input = hex_decode(hexin.trim()).expect("valid hex input");
        let digest = match alg {
            "tsh512" => tsh::tsh_512(&input),
            "pdh512" => pdh::pdh_512(&input),
            other => panic!("unknown alg: {other}"),
        };
        writeln!(out, "{} {}", alg, hex_encode(&digest)).expect("stdout");
    }
}
