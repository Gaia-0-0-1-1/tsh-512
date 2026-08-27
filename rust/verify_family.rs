//! verify_family.rs — cross-language truth harness for the v2 family
//! (std-only). Line protocol on stdin/stdout: each line
//! "<alg> <tick> <input-hex>" is answered "<alg> <486-char trit string>",
//! alg one of t1/t2/t3/t4 (rust/trit_family_v2.rs). Driven by
//! tools/verify_family_rust.py against vectors/trit_family_vectors_v2.json
//! — canon law: one serialization, refuse-unfaithful.
//!
//! Build: rustc --edition 2021 -O rust/verify_family.rs
//!   -o rust/target/verify_family.exe

#[path = "trit_family_v2.rs"]
mod fam;

use std::io::{self, BufRead, Write};

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let alg = parts.next().expect("alg");
        let tick: i64 = parts.next().expect("tick").parse().expect("tick int");
        let hexin = parts.next().expect("input hex");
        let input = hex_decode(hexin.trim()).expect("valid hex input");
        let digest = match alg {
            "t1" => fam::t1_hash(&input, tick),
            "t2" => fam::t2_hash(&input, tick),
            "t3" => fam::t3_hash(&input, tick),
            "t4" => fam::t4_hash(&input, tick),
            other => panic!("unknown alg: {other}"),
        };
        writeln!(out, "{} {}", alg, fam::output_string(&digest))
            .expect("stdout");
    }
}
