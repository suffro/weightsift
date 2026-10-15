//! Exactness, framing refusals and corruption detection of the prototype codec (and of Colibri's, when compiled in).

use lossless_moe_codec::container::{self, Block};
use lossless_moe_codec::rans::{self, Table};

/// A deterministic generator (SplitMix64), so the tests need no crate.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A skewed symbol in 0..alphabet (roughly geometric, like weight codes).
    fn skewed(&mut self, alphabet: usize) -> u8 {
        let mut s = 0;
        while s + 1 < alphabet && !self.next().is_multiple_of(3) {
            s += 1;
        }
        s as u8
    }
}

fn matrix(rows: usize, cols: usize, alphabet: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..rows * cols).map(|_| r.skewed(alphabet)).collect()
}

#[allow(clippy::too_many_arguments)]
fn round_trip(
    syms: &[u8],
    rows: usize,
    cols: usize,
    alphabet: usize,
    kind: u8,
    side: Vec<u8>,
    n_contexts: usize,
    streams: usize,
) {
    let block = container::encode(
        syms, rows, cols, alphabet, kind, side, n_contexts, None, 14, streams, None,
    )
    .expect("encode");
    let bytes = block.to_bytes();
    assert_eq!(bytes.len(), block.sizes().total);
    let parsed = Block::parse(&bytes, None).expect("parse");
    assert_eq!(parsed, block);
    let mut out = vec![0xAAu8; syms.len()];
    container::decode(&parsed, &mut rans::Bytes(&mut out), true).expect("decode");
    assert_eq!(out, syms, "kind {kind}, {rows}x{cols}, {streams} streams");
}

#[test]
fn normalize_keeps_every_present_symbol_and_sums_exactly() {
    for counts in [
        vec![1u64, 0, 0],
        vec![5, 0, 1_000_000, 1],
        vec![1; 256],
        vec![7, 7, 7],
        vec![0, 0, 9],
    ] {
        for bits in [8u32, 12, 15] {
            if counts.iter().filter(|&&c| c > 0).count() > 1 << bits {
                continue;
            }
            let f = rans::normalize(&counts, bits).unwrap();
            assert_eq!(f.iter().map(|&x| x as u64).sum::<u64>(), 1 << bits);
            for (c, x) in counts.iter().zip(&f) {
                assert_eq!(*c > 0, *x > 0);
            }
        }
    }
    assert!(rans::normalize(&[0, 0], 12).unwrap().iter().all(|&x| x == 0));
    assert!(rans::normalize(&vec![1u64; 300], 8).is_err());
}

#[test]
fn every_context_kind_round_trips() {
    let (rows, cols, k) = (37, 53, 16);
    let syms = matrix(rows, cols, k, 1);
    round_trip(&syms, rows, cols, k, 0, vec![], 1, 8);
    let col: Vec<u8> = (0..cols).map(|c| (c % 5) as u8).collect();
    round_trip(&syms, rows, cols, k, 1, col, 5, 8);
    let row: Vec<u8> = (0..rows).map(|r| (r % 3) as u8).collect();
    round_trip(&syms, rows, cols, k, 2, row, 3, 8);
    round_trip(&syms, rows, cols, k, 3, vec![], k + 1, 8);
}

#[test]
fn shapes_streams_and_degenerate_alphabets() {
    for (rows, cols, streams) in [
        (1, 1, 1),
        (1, 7, 32),
        (5, 1, 2),
        (33, 9, 32),
        (64, 64, 32),
        (65, 3, 8),
        (7, 7, 100),
    ] {
        let syms = matrix(rows, cols, 16, rows as u64 * 31 + cols as u64);
        round_trip(&syms, rows, cols, 16, 0, vec![], 1, streams);
        round_trip(&syms, rows, cols, 16, 3, vec![], 17, streams);
    }
    // one symbol only (its frequency is the whole table), an alphabet of 1 and of 256, every symbol present
    round_trip(&vec![3u8; 4096], 64, 64, 16, 0, vec![], 1, 32);
    round_trip(&[0u8; 100], 10, 10, 1, 0, vec![], 1, 4);
    let all: Vec<u8> = (0..65536u32).map(|i| (i % 256) as u8).collect();
    round_trip(&all, 256, 256, 256, 0, vec![], 1, 32);
    round_trip(&matrix(128, 100, 256, 9), 128, 100, 256, 3, vec![], 257, 16);
}

#[test]
fn external_tables_are_used_and_required() {
    let (rows, cols, k) = (40, 40, 16);
    let syms = matrix(rows, cols, k, 4);
    // a smoothed table from other data: every symbol codable
    let other = matrix(rows, cols, k, 5);
    let mut counts = vec![1u64; k];
    for &s in &other {
        counts[s as usize] += 1;
    }
    let t = Table::from_freq(rans::normalize(&counts, 14).unwrap(), 14).unwrap();
    let block = container::encode(&syms, rows, cols, k, 0, vec![], 1, Some(vec![t.clone()]), 14, 8, None).unwrap();
    assert!(block.external_tables && block.sizes().tables == 0);
    let bytes = block.to_bytes();
    assert!(
        Block::parse(&bytes, None).is_err(),
        "a block with external tables must refuse to parse without them"
    );
    let parsed = Block::parse(&bytes, Some(&[t])).unwrap();
    let mut out = vec![0u8; syms.len()];
    container::decode(&parsed, &mut rans::Bytes(&mut out), true).unwrap();
    assert_eq!(out, syms);
    // a table without the symbol refuses to encode
    let mut f = vec![0u32; k];
    f[0] = 1 << 14;
    let narrow = Table::from_freq(f, 14).unwrap();
    assert!(container::encode(&syms, rows, cols, k, 0, vec![], 1, Some(vec![narrow]), 14, 8, None).is_err());
}

#[test]
fn bf16_and_nibble_sinks_rebuild_stored_bytes() {
    let (rows, cols) = (16, 32);
    let mut r = Rng(11);
    let patterns: Vec<u16> = (0..rows * cols).map(|_| (r.next() & 0xFFFF) as u16).collect();
    let exps: Vec<u8> = patterns.iter().map(|p| ((p >> 7) & 0xFF) as u8).collect();
    let raw: Vec<u8> = patterns.iter().map(|p| (((p >> 15) << 7) | (p & 0x7F)) as u8).collect();
    let block = container::encode(&exps, rows, cols, 256, 0, vec![], 1, None, 15, 4, None).unwrap();
    let mut out = vec![0u16; rows * cols];
    container::decode(
        &block,
        &mut rans::Bf16 {
            raw: &raw,
            out: &mut out,
        },
        true,
    )
    .unwrap();
    assert_eq!(out, patterns);
    let nibbles = matrix(rows, cols, 16, 12);
    let packed: Vec<u8> = nibbles.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
    let block = container::encode(&nibbles, rows, cols, 16, 0, vec![], 1, None, 14, 4, None).unwrap();
    let mut out = vec![0u8; packed.len()];
    container::decode(&block, &mut rans::Nibbles(&mut out), true).unwrap();
    assert_eq!(out, packed);
}

#[test]
fn framing_refusals() {
    let syms = matrix(20, 20, 16, 2);
    let good = container::encode(&syms, 20, 20, 16, 0, vec![], 1, None, 14, 4, None)
        .unwrap()
        .to_bytes();
    assert!(Block::parse(&good, None).is_ok());
    assert!(Block::parse(&good[..good.len() - 1], None).is_err(), "truncated");
    let mut longer = good.clone();
    longer.push(0);
    assert!(Block::parse(&longer, None).is_err(), "trailing byte");
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(Block::parse(&magic, None).is_err(), "magic");
    let mut table = good.clone();
    table[container::HEADER_BYTES + 1] ^= 1; // first frequency's low byte: the table no longer sums to 2^14
    assert!(Block::parse(&table, None).is_err(), "table sum");
}

#[test]
fn every_single_bit_flip_of_the_payload_is_detected() {
    let (rows, cols) = (24, 40);
    let syms = matrix(rows, cols, 16, 3);
    let block = container::encode(&syms, rows, cols, 16, 0, vec![], 1, None, 14, 4, None).unwrap();
    let bytes = block.to_bytes();
    let payload_at = bytes.len() - block.payload.len();
    let (mut errors, mut differs) = (0, 0);
    for byte in payload_at..bytes.len() {
        for bit in 0..8 {
            let mut bad = bytes.clone();
            bad[byte] ^= 1 << bit;
            let parsed = Block::parse(&bad, None).unwrap();
            let mut out = vec![0u8; syms.len()];
            match container::decode(&parsed, &mut rans::Bytes(&mut out), true) {
                Err(_) => errors += 1,
                Ok(()) => {
                    assert_ne!(out, syms, "an undetected flip at byte {byte} bit {bit}");
                    differs += 1;
                }
            }
        }
    }
    // the checked decoder catches most flips itself; the rest change the output (the manifest's sha256 catches them)
    assert!(errors > differs, "{errors} refused, {differs} decoded to other bytes");
}

#[cfg(colibri_rans)]
#[test]
fn colibri_codec_round_trips_and_agrees_on_size() {
    use lossless_moe_codec::colibri;
    let nibbles = matrix(64, 256, 16, 21);
    let mut counts = [0u64; 16];
    for &s in &nibbles {
        counts[s as usize] += 1;
    }
    let f = rans::normalize(&counts, 14).unwrap();
    let table = colibri::ColibriTable::new(f.clone().try_into().unwrap(), 14).unwrap();
    let rec = colibri::encode(&nibbles, &table).unwrap();
    let padded = colibri::Padded::new(&rec);
    let packed: Vec<u8> = nibbles.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
    for path in [0, 1] {
        let mut out = vec![0u8; packed.len()];
        colibri::decode(&padded, &table, path, &mut out).unwrap();
        assert_eq!(out, packed, "Colibri arm {path}");
    }
    // the same table and coder construction: the payloads cost the same to within the streams' flush bytes
    let ours = container::encode(
        &nibbles,
        64,
        256,
        16,
        0,
        vec![],
        1,
        Some(vec![Table::from_freq(f, 14).unwrap()]),
        14,
        64,
        None,
    )
    .unwrap();
    let theirs_payload = rec.len() as i64 - 1056; // Colibri's header: 16 + 257 * 4 rounded to 16
    let ours_payload = ours.payload.len() as i64;
    assert!(
        (theirs_payload - ours_payload).abs() < 256 * 4 + 64 * 4 + 32,
        "{theirs_payload} vs {ours_payload}"
    );
}

#[test]
fn grid_classes_round_trip_and_are_required() {
    let (rows, cols, group) = (30, 96, 32);
    let syms = matrix(rows, cols, 16, 31);
    let per_row = cols.div_ceil(group);
    let classes: Vec<u8> = (0..rows * per_row).map(|i| (i % 4) as u8).collect();
    let block = container::encode(
        &syms,
        rows,
        cols,
        16,
        4,
        vec![],
        4,
        None,
        12,
        8,
        Some((classes.clone(), group as u32)),
    )
    .unwrap();
    let bytes = block.to_bytes();
    assert_eq!(bytes.len(), block.sizes().total);
    assert!(Block::parse(&bytes, None).is_err(), "the derived classes are required");
    assert!(
        Block::parse_with(&bytes, None, Some(&classes[1..])).is_err(),
        "wrong class count"
    );
    let parsed = Block::parse_with(&bytes, None, Some(&classes)).unwrap();
    let mut out = vec![0u8; syms.len()];
    container::decode(&parsed, &mut rans::Bytes(&mut out), true).unwrap();
    assert_eq!(out, syms);
    // a non-multiple width: the last group of each row is partial
    let (cols, group) = (100, 32);
    let syms = matrix(rows, cols, 16, 32);
    let per_row = cols.div_ceil(group);
    let classes: Vec<u8> = (0..rows * per_row).map(|i| (i % 3) as u8).collect();
    let block = container::encode(
        &syms,
        rows,
        cols,
        16,
        4,
        vec![],
        3,
        None,
        12,
        8,
        Some((classes.clone(), group as u32)),
    )
    .unwrap();
    let parsed = Block::parse_with(&block.to_bytes(), None, Some(&classes)).unwrap();
    let mut out = vec![0u8; syms.len()];
    container::decode(&parsed, &mut rans::Bytes(&mut out), true).unwrap();
    assert_eq!(out, syms);
}
