//! A fixture checkpoint on disk for the store's tests: two files of pseudo-random bytes and a catalog of two layers.
//!
//! Layer 0 has the layout of fused expert tensors: three plain segments (`w13`, gate and up rows back to back; `w2`; a
//! segment of scales), each row starting off the 4 KiB grid, and views `gate`, `up`, `down` with their scales. Layer 1
//! has one composed segment whose rows are three spans scattered over both files, and views without scales. These are
//! interface fixtures: they exercise the store's contract, nothing about any model.

#![allow(dead_code)]

use std::io::Write;
use std::path::PathBuf;

use weightsift_store::catalog::{
    Catalog, FileEntry, LayerEntry, PartEntry, ScalesEntry, SegmentEntry, ViewEntry, FORMAT, VERSION,
};

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

pub const EXPERTS0: u32 = 6;
pub const EXPERTS1: u32 = 4;
pub const HALF: u64 = 40_000; // gate and up rows of layer 0
pub const SCALES: u64 = 1_000;

pub struct Fixture {
    pub dir: PathBuf,
    pub data: Vec<Vec<u8>>,
    pub catalog: Catalog,
}

impl Fixture {
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("weightsift-store-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rng = Lcg(name.bytes().fold(7, |a, b| a * 31 + b as u64));
        let sizes = [1_200_000u64, 400_000];
        let data: Vec<Vec<u8>> = sizes
            .iter()
            .map(|&n| (0..n).map(|_| rng.next() as u8).collect())
            .collect();
        for (k, bytes) in data.iter().enumerate() {
            std::fs::File::create(dir.join(format!("f{k}.bin")))
                .unwrap()
                .write_all(bytes)
                .unwrap();
        }
        let w13 = 2 * HALF;
        let w13_offset = 4096 + 123;
        let w2_offset = w13_offset + EXPERTS0 as u64 * w13 + 777;
        let mut spans = Vec::new();
        let part_bytes = vec![10_000u64, 7_000, 5_000];
        let mut cursor = [w2_offset + EXPERTS0 as u64 * HALF + 5_000, 30_000];
        for _ in 0..EXPERTS1 {
            for &n in &part_bytes {
                let file = rng.below(2) as usize;
                let gap = [0, 0, 13, 4_096][rng.below(4) as usize];
                spans.push((file as u32, cursor[file] + gap));
                cursor[file] += gap + n;
            }
        }
        assert!(cursor[0] <= sizes[0] && cursor[1] <= sizes[1]);
        let view =
            |name: &str, part: u32, offset: u64, bytes: u64, format: &str, scales: Option<ScalesEntry>| ViewEntry {
                name: name.into(),
                part,
                offset,
                bytes,
                format: format.into(),
                rows: (bytes / 100) as i64,
                columns: 200,
                block_rows: 0,
                block_columns: if scales.is_none() { 32 } else { 0 },
                scales,
            };
        let scales = |offset: u64| {
            Some(ScalesEntry {
                part: 2,
                offset,
                bytes: SCALES,
                format: "f32".into(),
            })
        };
        let catalog = Catalog {
            format: FORMAT.into(),
            version: VERSION,
            name: format!("fixture {name}"),
            files: (0..2)
                .map(|k| FileEntry {
                    key: format!("f{k}"),
                    path: format!("f{k}.bin"),
                    size: Some(sizes[k]),
                    sha256: None,
                })
                .collect(),
            segments: vec![
                SegmentEntry {
                    name: "l0.w13".into(),
                    files: vec![0],
                    rows: EXPERTS0 as u64,
                    row_bytes: w13,
                    offset: Some(w13_offset),
                    part_bytes: None,
                    spans: None,
                },
                SegmentEntry {
                    name: "l0.w2".into(),
                    files: vec![0],
                    rows: EXPERTS0 as u64,
                    row_bytes: HALF,
                    offset: Some(w2_offset),
                    part_bytes: None,
                    spans: None,
                },
                SegmentEntry {
                    name: "l0.scales".into(),
                    files: vec![1],
                    rows: EXPERTS0 as u64,
                    row_bytes: 3 * SCALES,
                    offset: Some(50),
                    part_bytes: None,
                    spans: None,
                },
                SegmentEntry {
                    name: "l1.w".into(),
                    files: vec![0, 1],
                    rows: EXPERTS1 as u64,
                    row_bytes: part_bytes.iter().sum(),
                    offset: None,
                    part_bytes: Some(part_bytes),
                    spans: Some(spans),
                },
            ],
            layers: vec![
                LayerEntry {
                    layer: 0,
                    experts: EXPERTS0,
                    parts: vec![
                        PartEntry {
                            name: "w13".into(),
                            segment: 0,
                            first_row: 0,
                        },
                        PartEntry {
                            name: "w2".into(),
                            segment: 1,
                            first_row: 0,
                        },
                        PartEntry {
                            name: "scales".into(),
                            segment: 2,
                            first_row: 0,
                        },
                    ],
                    views: vec![
                        view("gate", 0, 0, HALF, "q4_row", scales(0)),
                        view("up", 0, HALF, HALF, "q4_row", scales(SCALES)),
                        view("down", 1, 0, HALF, "q4_row", scales(2 * SCALES)),
                    ],
                },
                LayerEntry {
                    layer: 7,
                    experts: EXPERTS1,
                    parts: vec![PartEntry {
                        name: "w".into(),
                        segment: 3,
                        first_row: 0,
                    }],
                    views: vec![
                        view("gate", 0, 0, 10_000, "fp4_native_block", None),
                        view("up", 0, 10_000, 7_000, "fp4_native_block", None),
                        view("down", 0, 17_000, 5_000, "fp4_native_block", None),
                    ],
                },
            ],
        };
        catalog.validate().unwrap();
        std::fs::write(dir.join("catalog.json"), catalog.to_json()).unwrap();
        Fixture { dir, data, catalog }
    }

    pub fn catalog_path(&self) -> PathBuf {
        self.dir.join("catalog.json")
    }

    /// Row `row` of segment `segment`, from the bytes written (not through any store).
    pub fn row(&self, segment: usize, row: u64) -> Vec<u8> {
        let s = &self.catalog.segments[segment];
        match (&s.offset, &s.part_bytes, &s.spans) {
            (Some(offset), _, _) => {
                let start = (offset + row * s.row_bytes) as usize;
                self.data[s.files[0] as usize][start..start + s.row_bytes as usize].to_vec()
            }
            (None, Some(part_bytes), Some(spans)) => {
                let mut out = Vec::new();
                for (p, &n) in part_bytes.iter().enumerate() {
                    let (file, offset) = spans[row as usize * part_bytes.len() + p];
                    let file = s.files[file as usize] as usize;
                    out.extend_from_slice(&self.data[file][offset as usize..(offset + n) as usize]);
                }
                out
            }
            _ => unreachable!(),
        }
    }

    /// The parts of expert `expert` of layer `layer`, from the bytes written.
    pub fn expert(&self, layer: u32, expert: u32) -> Vec<Vec<u8>> {
        let entry = self.catalog.layers.iter().find(|l| l.layer == layer).unwrap();
        entry
            .parts
            .iter()
            .map(|p| self.row(p.segment as usize, p.first_row + expert as u64))
            .collect()
    }

    /// Bytes of one expert of layer `layer`.
    pub fn expert_bytes(&self, layer: u32) -> u64 {
        self.expert(layer, 0).iter().map(|p| p.len() as u64).sum()
    }

    pub fn keys(&self) -> Vec<(u32, u32)> {
        let mut keys: Vec<(u32, u32)> = (0..EXPERTS0).map(|e| (0, e)).collect();
        keys.extend((0..EXPERTS1).map(|e| (7, e)));
        keys
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
