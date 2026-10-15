//! Colibri's int4-rans256-g0 codec (`c/rans.h` through its own `c/tools/rans_ctypes.c`), as pinned in
//! `research/lossless_moe/upstream.json` and compiled by `build.rs`. Only its exported entry points are called.
//!
//! Its contracts, honoured here: 256 streams (the bridge refuses any other count), records decoded from a 4-byte
//! aligned buffer with at least 64 readable bytes past the record (RANS_SLACK), tables whose slot-to-symbol map agrees
//! with the frequencies.

use std::ffi::CStr;
use std::os::raw::c_char;

pub const STREAMS: u32 = 256;
pub const SLACK: usize = 64;

extern "C" {
    fn rc_record_encode(
        nibbles: *const u8,
        n: u64,
        freq: *const u32,
        start: *const u32,
        slot_to_symbol: *const u16,
        scale_bits: u32,
        n_streams: u32,
        out: *mut u8,
        out_cap: u64,
    ) -> i64;
    fn rc_record_bound(n_symbols: u64, n_streams: u32) -> u64;
    fn rc_record_decode(
        blob: *const u8,
        blob_len: u64,
        n_streams: u32,
        freq: *const u32,
        start: *const u32,
        slot_to_symbol: *const u16,
        scale_bits: u32,
        path: i32,
        out_packed: *mut u8,
    ) -> i32;
    fn rc_err_name(e: i32) -> *const c_char;
}

/// A Colibri table: 16 frequencies summing to 2^scale_bits, their prefix sums, and the slot map.
#[derive(Clone)]
pub struct ColibriTable {
    pub scale_bits: u32,
    pub freq: [u32; 16],
    pub start: [u32; 16],
    pub slot: Vec<u16>,
}

impl ColibriTable {
    pub fn new(freq: [u32; 16], scale_bits: u32) -> Result<Self, String> {
        let m: u32 = freq.iter().sum();
        if m != 1 << scale_bits {
            return Err(format!("Colibri frequencies sum to {m}"));
        }
        let mut start = [0u32; 16];
        let mut slot = vec![0u16; m as usize];
        let mut acc = 0;
        for s in 0..16 {
            start[s] = acc;
            for i in acc..acc + freq[s] {
                slot[i as usize] = s as u16;
            }
            acc += freq[s];
        }
        Ok(ColibriTable {
            scale_bits,
            freq,
            start,
            slot,
        })
    }
}

fn err_name(e: i64) -> String {
    // SAFETY: rc_err_name returns a pointer to a static NUL-terminated string for any code.
    unsafe { CStr::from_ptr(rc_err_name(e as i32)).to_string_lossy().into_owned() }
}

/// One int4-rans256-g0 record of `nibbles` (values 0..15).
pub fn encode(nibbles: &[u8], t: &ColibriTable) -> Result<Vec<u8>, String> {
    // SAFETY: rc_record_bound reads its two integer arguments only.
    let cap = unsafe { rc_record_bound(nibbles.len() as u64, STREAMS) } as usize;
    if cap == 0 {
        return Err("rc_record_bound refused".into());
    }
    let mut out = vec![0u8; cap];
    // SAFETY: the pointers cover `nibbles.len()` bytes, 16 u32 (freq and start), 2^scale_bits u16 (slot), and `cap`
    // writable bytes, which is the bound the encoder checks against; all outlive the call.
    let len = unsafe {
        rc_record_encode(
            nibbles.as_ptr(),
            nibbles.len() as u64,
            t.freq.as_ptr(),
            t.start.as_ptr(),
            t.slot.as_ptr(),
            t.scale_bits,
            STREAMS,
            out.as_mut_ptr(),
            cap as u64,
        )
    };
    if len < 0 {
        return Err(format!("rc_record_encode: {}", err_name(-len)));
    }
    out.truncate(len as usize);
    Ok(out)
}

/// A record copied into a 4-byte aligned buffer with SLACK readable bytes after it.
pub struct Padded {
    words: Vec<u32>,
    len: usize,
}

impl Padded {
    pub fn new(record: &[u8]) -> Self {
        let mut words = vec![0u32; (record.len() + SLACK).div_ceil(4)];
        // SAFETY: `words` holds at least record.len() + SLACK bytes; the byte view is within its allocation.
        let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 4) };
        bytes[..record.len()].copy_from_slice(record);
        Padded {
            words,
            len: record.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Decode a padded record into `out` (its packed bytes) with Colibri's arm `path` (0 scalar, 1 scalar_bf).
pub fn decode(record: &Padded, t: &ColibriTable, path: i32, out: &mut [u8]) -> Result<(), String> {
    // SAFETY: the record pointer is 4-byte aligned with `len + SLACK` readable bytes (Padded); the table pointers cover
    // 16 u32 and 2^scale_bits u16; `out` must hold the record's packed bytes, which the decoder writes and nothing more
    // (checked by the caller against the record's own count before the call).
    let e = unsafe {
        rc_record_decode(
            record.words.as_ptr() as *const u8,
            record.len as u64,
            STREAMS,
            t.freq.as_ptr(),
            t.start.as_ptr(),
            t.slot.as_ptr(),
            t.scale_bits,
            path,
            out.as_mut_ptr(),
        )
    };
    if e != 0 {
        return Err(format!("rc_record_decode: {}", err_name(e as i64)));
    }
    Ok(())
}

/// The packed byte count a record declares (its header's second field), checked against its first.
pub fn packed_bytes(record: &[u8]) -> Result<usize, String> {
    if record.len() < 16 {
        return Err("record too short".into());
    }
    let n = u64::from_le_bytes(record[0..8].try_into().unwrap());
    let packed = u64::from_le_bytes(record[8..16].try_into().unwrap());
    if packed != n / 2 + (n & 1) {
        return Err("record header counts disagree".into());
    }
    Ok(packed as usize)
}
