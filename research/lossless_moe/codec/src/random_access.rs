//! Cold random access to single experts: direct (unbuffered) reads of an expert's compressed record from a pack, then
//! its decode, against direct reads of the same expert's original bytes from the checkpoint.
//!
//! The spec (written by the Python harness) names the pack, each expert's 4 KiB-aligned extent in it and its matrices'
//! records, and the original extents. Reads bypass the OS cache (FILE_FLAG_NO_BUFFERING), so each read reaches the
//! drive; offsets, lengths and buffers are 4 KiB aligned, and the bytes physically read are those aligned extents.
//! Every decoded matrix is checked against its original sha256 after the timed region.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::time::Instant;

use lossless_moe_codec::container::{self, Block};
use lossless_moe_codec::rans;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const ALIGN: u64 = 4096;

#[derive(Deserialize)]
struct Spec {
    pack: String,
    experts: Vec<Expert>,
    order: Vec<usize>,
}

#[derive(Deserialize)]
struct Expert {
    name: String,
    offset: u64,
    length: u64,
    matrices: Vec<Matrix>,
    original: Vec<Extent>,
}

#[derive(Deserialize)]
struct Matrix {
    format: String,
    record_offset: u64,
    record_len: u64,
    raw_offset: u64,
    raw_len: u64,
    out_bytes: u64,
    original_sha256: String,
}

#[derive(Deserialize)]
struct Extent {
    file: String,
    offset: u64,
    length: u64,
}

struct Aligned {
    ptr: *mut u8,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(ALIGN as usize), ALIGN as usize).unwrap();
        // SAFETY: a nonzero size and a power-of-two alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Aligned {
            ptr,
            len: layout.size(),
        }
    }

    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` owns `len` initialized (zeroed) bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.ptr, Layout::from_size_align(self.len, ALIGN as usize).unwrap()) }
    }
}

fn open_direct(path: &str) -> File {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        o.custom_flags(0x2000_0000); // FILE_FLAG_NO_BUFFERING
    }
    o.open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"))
}

fn read_at(file: &File, buf: &mut [u8], offset: u64) {
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(windows)]
        let n = {
            use std::os::windows::fs::FileExt;
            file.seek_read(&mut buf[done..], offset + done as u64).expect("read")
        };
        #[cfg(not(windows))]
        let n = {
            use std::os::unix::fs::FileExt;
            file.read_at(&mut buf[done..], offset + done as u64).expect("read")
        };
        if n == 0 {
            // a direct read may stop at the file's end inside the last aligned block
            break;
        }
        done += n;
    }
}

/// Aligned extent [start, end) covering [offset, offset + length).
fn extent(offset: u64, length: u64) -> (u64, u64) {
    let start = offset / ALIGN * ALIGN;
    let end = (offset + length).div_ceil(ALIGN) * ALIGN;
    (start, end)
}

pub fn run(spec_path: &str, out_path: &str) {
    let spec: Spec = serde_json::from_str(&std::fs::read_to_string(spec_path).unwrap()).expect("spec JSON");
    let pack = open_direct(&spec.pack);
    let mut originals: std::collections::HashMap<String, File> = Default::default();
    let mut rows = Vec::new();
    let mut exact_all = true;
    for &i in &spec.order {
        let e = &spec.experts[i];
        // compressed: one aligned read of the expert's record, then decode every matrix
        let (start, end) = extent(e.offset, e.length);
        let mut buf = Aligned::new((end - start) as usize);
        let t0 = Instant::now();
        read_at(&pack, &mut buf.slice()[..(end - start) as usize], start);
        let t1 = Instant::now();
        let base = (e.offset - start) as usize;
        let mut outputs = Vec::new();
        for m in &e.matrices {
            let rec = &buf.slice()[base + m.record_offset as usize..base + (m.record_offset + m.record_len) as usize];
            let block = Block::parse(rec, None).expect("record");
            let mut out = vec![0u8; m.out_bytes as usize];
            match m.format.as_str() {
                "bf16" => {
                    let raw =
                        buf.slice()[base + m.raw_offset as usize..base + (m.raw_offset + m.raw_len) as usize].to_vec();
                    // SAFETY: two bytes per weight, u16 has no invalid patterns, the view is within `out`.
                    let words = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, out.len() / 2) };
                    container::decode(&block, &mut rans::Bf16 { raw: &raw, out: words }, true).expect("decode");
                }
                "int4" => container::decode(&block, &mut rans::Nibbles(&mut out), true).expect("decode"),
                _ => container::decode(&block, &mut rans::Bytes(&mut out), true).expect("decode"),
            }
            outputs.push(out);
        }
        let t2 = Instant::now();
        // original: direct reads of the expert's original extents
        let t3 = Instant::now();
        let mut original_bytes = 0u64;
        for x in &e.original {
            let f = originals.entry(x.file.clone()).or_insert_with(|| open_direct(&x.file));
            let (s, en) = extent(x.offset, x.length);
            let mut b = Aligned::new((en - s) as usize);
            read_at(f, &mut b.slice()[..(en - s) as usize], s);
            original_bytes += en - s;
        }
        let t4 = Instant::now();
        let exact = e.matrices.iter().zip(&outputs).all(|(m, o)| {
            Sha256::digest(o).iter().map(|b| format!("{b:02x}")).collect::<String>() == m.original_sha256
        });
        exact_all &= exact;
        rows.push(serde_json::json!({
            "expert": e.name, "record_bytes": e.length, "read_bytes": end - start,
            "read_s": (t1 - t0).as_secs_f64(), "decode_s": (t2 - t1).as_secs_f64(), "total_s": (t2 - t0).as_secs_f64(),
            "original_read_bytes": original_bytes, "original_read_s": (t4 - t3).as_secs_f64(), "exact": exact,
        }));
    }
    let result = serde_json::json!({"exact": exact_all, "reads": rows});
    std::fs::write(out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    assert!(exact_all, "a decoded expert differs from its original");
}
