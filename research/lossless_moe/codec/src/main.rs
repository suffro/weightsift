//! `lmc`: the study's codec driver.
//!
//! `lmc bench JOB.json OUT.json` encodes every block of a job (symbol files written by the Python harness), checks the
//! round trip against the block's original stored bytes, and times decoding of all blocks at each thread count
//! (best and median of the job's repeats; decode tables are built inside the timed region). With Colibri compiled in,
//! int4 blocks that carry a Colibri table are also encoded and decoded by Colibri's own codec (scalar and branch-free
//! arms). `lmc random-access SPEC.json OUT.json` measures cold single-expert reads plus decodes (see `random_access`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use lossless_moe_codec::container::{self, Block};
use lossless_moe_codec::rans::{self, Table};
use serde::Deserialize;
use sha2::{Digest, Sha256};

mod random_access;
mod sys;

#[derive(Deserialize)]
struct Job {
    blocks: Vec<BlockSpec>,
    threads: Vec<usize>,
    repeats: usize,
    /// Write each block's serialized record here (for the random-access measurement), if set.
    #[serde(default)]
    containers: Option<String>,
}

fn one() -> usize {
    1
}

#[derive(Deserialize, Clone)]
struct BlockSpec {
    name: String,
    /// bf16 (symbols are exponents, `raw` holds sign and mantissa), int4 (symbols are nibbles), bytes
    format: String,
    symbols: String,
    #[serde(default)]
    raw: Option<String>,
    original: String,
    rows: usize,
    cols: usize,
    alphabet: usize,
    #[serde(default)]
    kind: u8,
    #[serde(default)]
    side: Option<String>,
    #[serde(default = "one")]
    n_contexts: usize,
    /// u32 little-endian frequencies [n_contexts * alphabet] (external tables, counted elsewhere)
    #[serde(default)]
    tables: Option<String>,
    scale_bits: u32,
    streams: usize,
    #[serde(default)]
    colibri_freq: Option<Vec<u32>>,
    #[serde(default)]
    colibri_scale_bits: Option<u32>,
    /// Kind 4: a class (u8) per (row, group) derived from the scales, and the group width
    #[serde(default)]
    grid: Option<String>,
    #[serde(default)]
    grid_group: Option<u32>,
}

fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Prepared {
    spec: BlockSpec,
    record: Vec<u8>,
    external: Option<Vec<Table>>,
    out_bytes: usize,
    raw: Vec<u8>,
    grid: Option<Vec<u8>>,
    #[cfg(colibri_rans)]
    colibri: Option<(
        lossless_moe_codec::colibri::Padded,
        lossless_moe_codec::colibri::ColibriTable,
    )>,
}

fn decode_record(p: &Prepared, out: &mut [u8]) -> Result<(), String> {
    let block = Block::parse_with(&p.record, p.external.as_deref(), p.grid.as_deref())?;
    match p.spec.format.as_str() {
        "bf16" => {
            // SAFETY: `out` has 2 bytes per weight; u16 has no invalid bit patterns and the view stays in bounds. The
            // output is little-endian like the checkpoint (x86).
            let words = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, out.len() / 2) };
            let mut sink = rans::Bf16 {
                raw: &p.raw,
                out: words,
            };
            container::decode(&block, &mut sink, true)
        }
        "int4" => container::decode(&block, &mut rans::Nibbles(out), true),
        _ => container::decode(&block, &mut rans::Bytes(out), true),
    }
}

fn external_tables(path: &str, n_contexts: usize, alphabet: usize, scale_bits: u32) -> Vec<Table> {
    let bytes = read(path);
    let freq: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(freq.len(), n_contexts * alphabet, "{path}: table size");
    freq.chunks(alphabet)
        .map(|f| Table::from_freq(f.to_vec(), scale_bits).expect("valid external table"))
        .collect()
}

fn timed<F: Fn(usize) -> Result<(), String> + Sync>(
    n: usize,
    threads: usize,
    repeats: usize,
    f: F,
) -> serde_json::Value {
    let mut walls = Vec::new();
    let mut cpu = Vec::new();
    for _ in 0..repeats {
        let next = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());
        let cpu0 = sys::process_cpu_seconds();
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    if let Err(e) = f(i) {
                        failures.lock().unwrap().push(e);
                    }
                });
            }
        });
        walls.push(t0.elapsed().as_secs_f64());
        cpu.push(sys::process_cpu_seconds() - cpu0);
        let failures = failures.into_inner().unwrap();
        assert!(
            failures.is_empty(),
            "decode failed: {:?}",
            &failures[..failures.len().min(3)]
        );
    }
    let mut sorted = walls.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    serde_json::json!({
        "threads": threads, "wall_s": walls, "cpu_s": cpu,
        "best_s": sorted[0], "median_s": sorted[sorted.len() / 2],
    })
}

fn bench(job_path: &str, out_path: &str) {
    let job: Job = serde_json::from_str(&String::from_utf8(read(job_path)).unwrap()).expect("job JSON");
    let mut prepared = Vec::new();
    let mut records = Vec::new();
    for spec in &job.blocks {
        let syms = read(&spec.symbols);
        let side = spec.side.as_deref().map(read).unwrap_or_default();
        let grid = spec.grid.as_deref().map(read);
        let external = spec
            .tables
            .as_deref()
            .map(|p| external_tables(p, spec.n_contexts, spec.alphabet, spec.scale_bits));
        let original = read(&spec.original);
        let raw = spec.raw.as_deref().map(read).unwrap_or_default();
        let t0 = Instant::now();
        let block = container::encode(
            &syms,
            spec.rows,
            spec.cols,
            spec.alphabet,
            spec.kind,
            side,
            spec.n_contexts,
            external.clone(),
            spec.scale_bits,
            spec.streams,
            grid.clone().map(|g| (g, spec.grid_group.expect("grid_group"))),
        )
        .unwrap_or_else(|e| panic!("{}: encode: {e}", spec.name));
        let encode_s = t0.elapsed().as_secs_f64();
        let record = block.to_bytes();
        let sizes = block.sizes();
        assert_eq!(sizes.total, record.len());
        let out_bytes = original.len();
        let mut p = Prepared {
            spec: spec.clone(),
            record,
            external,
            out_bytes,
            raw,
            grid,
            #[cfg(colibri_rans)]
            colibri: None,
        };
        let mut out = vec![0u8; out_bytes];
        let check = decode_record(&p, &mut out);
        let exact = check.is_ok() && out == original;
        let mut entry = serde_json::json!({
            "name": spec.name, "format": spec.format, "rows": spec.rows, "cols": spec.cols, "kind": spec.kind,
            "n_contexts": spec.n_contexts, "external_tables": p.external.is_some(), "streams": block.offsets.len(),
            "sizes": sizes, "raw_bytes": p.raw.len(), "original_bytes": out_bytes,
            "exact": exact, "decode_error": check.err(), "decoded_sha256": hex(&Sha256::digest(&out)),
            "original_sha256": hex(&Sha256::digest(&original)), "encode_s": encode_s,
        });
        #[cfg(colibri_rans)]
        if let (Some(freq), "int4") = (&spec.colibri_freq, spec.format.as_str()) {
            use lossless_moe_codec::colibri;
            let mut f = [0u32; 16];
            f.copy_from_slice(freq);
            let table = colibri::ColibriTable::new(f, spec.colibri_scale_bits.unwrap_or(14)).expect("Colibri table");
            let t1 = Instant::now();
            let rec = colibri::encode(&syms, &table).expect("Colibri encode");
            let c_encode_s = t1.elapsed().as_secs_f64();
            assert_eq!(colibri::packed_bytes(&rec).unwrap(), out_bytes);
            let padded = colibri::Padded::new(&rec);
            let mut exact_arms = serde_json::Map::new();
            for (path, name) in [(0, "scalar"), (1, "scalar_bf")] {
                let mut o = vec![0u8; out_bytes];
                let ok = colibri::decode(&padded, &table, path, &mut o).is_ok() && o == original;
                exact_arms.insert(name.into(), ok.into());
            }
            entry["colibri"] =
                serde_json::json!({"record_bytes": rec.len(), "exact": exact_arms, "encode_s": c_encode_s});
            p.colibri = Some((padded, table));
        }
        if let Some(dir) = &job.containers {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(format!("{dir}/{}.lmc", spec.name.replace('/', "_")), &p.record).unwrap();
        }
        records.push(entry);
        prepared.push(p);
    }
    let n = prepared.len();
    let outputs: Vec<Mutex<Vec<u8>>> = prepared.iter().map(|p| Mutex::new(vec![0u8; p.out_bytes])).collect();
    let total_out: usize = prepared.iter().map(|p| p.out_bytes).sum();
    let total_stored: usize = prepared.iter().map(|p| p.record.len() + p.raw.len()).sum();
    let mut timing = Vec::new();
    for &threads in &job.threads {
        let ours = timed(n, threads, job.repeats, |i| {
            decode_record(&prepared[i], &mut outputs[i].lock().unwrap())
        });
        let mut row = serde_json::json!({"threads": threads, "lmc": ours});
        #[cfg(colibri_rans)]
        if prepared.iter().all(|p| p.colibri.is_some()) {
            for (path, name) in [(0, "colibri_scalar"), (1, "colibri_scalar_bf")] {
                let r = timed(n, threads, job.repeats, |i| {
                    let (rec, table) = prepared[i].colibri.as_ref().unwrap();
                    lossless_moe_codec::colibri::decode(rec, table, path, &mut outputs[i].lock().unwrap())
                });
                row[name] = r;
            }
        }
        timing.push(row);
    }
    // the timed decodes must still be exact
    for (p, o) in prepared.iter().zip(&outputs) {
        assert_eq!(
            &*o.lock().unwrap(),
            &read(&p.spec.original),
            "{}: timed output differs",
            p.spec.name
        );
    }
    let result = serde_json::json!({
        "colibri_compiled": lossless_moe_codec::COLIBRI,
        "blocks": records, "total_output_bytes": total_out, "total_stored_bytes": total_stored,
        "timing": timing, "peak_working_set_bytes": sys::peak_working_set(),
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("bench") if args.len() == 4 => bench(&args[2], &args[3]),
        Some("random-access") if args.len() == 4 => random_access::run(&args[2], &args[3]),
        _ => {
            eprintln!("usage: lmc bench JOB.json OUT.json | lmc random-access SPEC.json OUT.json");
            std::process::exit(2);
        }
    }
}
