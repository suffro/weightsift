//! The expert catalog: where a checkpoint's routed experts are, written by a model adapter from the checkpoint's headers
//! and read by the store (format `weightsift-expert-catalog`, version 1).
//!
//! A catalog lists files, segments (Weightsift's storage layout of decisions 0006 and 0007: fixed-size rows of byte spans
//! of files, plain or composed), and layers. A layer has `experts` experts and a list of parts: expert `e`'s part `p` is
//! row `first_row + e` of the part's segment (a fused tensor `[E, ...]` is a plain segment as it is; per-expert tensors
//! are a composed segment). Views name ranges of an expert's parts with a tensor's shape and the format the bytes are
//! stored in (`format` and the scales' `format` are free strings that the engine reading them interprets; the store
//! never reads or converts them). Nothing here names a model: the catalog carries that knowledge as data.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "weightsift-expert-catalog";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub format: String,
    pub version: u32,
    /// What the catalog describes (a checkpoint and its revision), for reports.
    #[serde(default)]
    pub name: String,
    pub files: Vec<FileEntry>,
    pub segments: Vec<SegmentEntry>,
    pub layers: Vec<LayerEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub key: String,
    /// Absolute, or relative to the catalog's directory.
    pub path: String,
    /// The size the adapter saw; checked when the store opens the file.
    #[serde(default)]
    pub size: Option<u64>,
    /// The publisher's digest, recorded for audits (not recomputed when a store opens).
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentEntry {
    pub name: String,
    /// Indices into `files`: one for a plain segment, the files of its spans for a composed one.
    pub files: Vec<u32>,
    pub rows: u64,
    pub row_bytes: u64,
    /// A plain segment: row r is `row_bytes` bytes at `offset + r * row_bytes` of its file.
    #[serde(default)]
    pub offset: Option<u64>,
    /// A composed segment: row r, part p is `part_bytes[p]` bytes at `spans[r * parts + p]` = (index into `files`,
    /// offset).
    #[serde(default)]
    pub part_bytes: Option<Vec<u64>>,
    #[serde(default)]
    pub spans: Option<Vec<(u32, u64)>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerEntry {
    pub layer: u32,
    pub experts: u32,
    pub parts: Vec<PartEntry>,
    #[serde(default)]
    pub views: Vec<ViewEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartEntry {
    pub name: String,
    pub segment: u32,
    /// Expert e is row `first_row + e` of the segment.
    #[serde(default)]
    pub first_row: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewEntry {
    pub name: String,
    pub part: u32,
    #[serde(default)]
    pub offset: u64,
    pub bytes: u64,
    pub format: String,
    #[serde(default)]
    pub rows: i64,
    #[serde(default)]
    pub columns: i64,
    #[serde(default)]
    pub block_rows: u32,
    #[serde(default)]
    pub block_columns: u32,
    #[serde(default)]
    pub scales: Option<ScalesEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalesEntry {
    pub part: u32,
    #[serde(default)]
    pub offset: u64,
    pub bytes: u64,
    pub format: String,
}

fn text(what: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.contains('\0') {
        return Err(format!("{what} must be a non-empty string without NUL"));
    }
    Ok(())
}

impl Catalog {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let catalog: Catalog = serde_json::from_str(json).map_err(|e| format!("malformed catalog: {e}"))?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Read and check a catalog file; relative file paths are resolved against its directory.
    pub fn from_path(path: &Path) -> Result<(Self, PathBuf), String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read the catalog {}: {e}", path.display()))?;
        let catalog = Catalog::from_json(&json)?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Ok((catalog, base))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a catalog serializes")
    }

    /// The path of file `index`, resolved against `base`.
    pub fn file_path(&self, index: usize, base: &Path) -> PathBuf {
        let path = PathBuf::from(&self.files[index].path);
        if path.is_absolute() {
            path
        } else {
            base.join(path)
        }
    }

    /// Check that everything the catalog says is consistent (files, segments, layers, parts, views); files are
    /// checked against the disk when a store opens them.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT || self.version != VERSION {
            return Err(format!(
                "not a {FORMAT} v{VERSION} catalog: {} v{}",
                self.format, self.version
            ));
        }
        if self.files.is_empty() || self.layers.is_empty() {
            return Err("a catalog needs files and layers".into());
        }
        let mut keys = HashSet::new();
        for file in &self.files {
            text("a file key", &file.key)?;
            text("a file path", &file.path)?;
            if !keys.insert(&file.key) {
                return Err(format!("file key {} appears twice", file.key));
            }
        }
        for segment in &self.segments {
            text("a segment name", &segment.name)?;
            if segment.rows == 0 || segment.row_bytes == 0 {
                return Err(format!("{}: a segment needs rows of at least one byte", segment.name));
            }
            if segment.files.iter().any(|&f| f as usize >= self.files.len()) {
                return Err(format!("{}: refers to an unknown file", segment.name));
            }
            match (&segment.offset, &segment.part_bytes, &segment.spans) {
                (Some(_), None, None) if segment.files.len() == 1 => {}
                (None, Some(part_bytes), Some(spans)) => {
                    if part_bytes.is_empty() || part_bytes.contains(&0) {
                        return Err(format!("{}: composed parts must be non-empty", segment.name));
                    }
                    if part_bytes.iter().sum::<u64>() != segment.row_bytes {
                        return Err(format!("{}: parts do not make row_bytes", segment.name));
                    }
                    if spans.len() as u64 != segment.rows * part_bytes.len() as u64 {
                        return Err(format!("{}: expected one span per row and part", segment.name));
                    }
                    if spans.iter().any(|&(f, _)| f as usize >= segment.files.len()) {
                        return Err(format!(
                            "{}: a span names a file the segment does not list",
                            segment.name
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "{}: give one file and an offset, or part sizes and spans",
                        segment.name
                    ))
                }
            }
        }
        let mut layers = HashSet::new();
        for layer in &self.layers {
            let at = format!("layer {}", layer.layer);
            if !layers.insert(layer.layer) {
                return Err(format!("{at} appears twice"));
            }
            if layer.experts == 0 || layer.parts.is_empty() {
                return Err(format!("{at}: needs experts and parts"));
            }
            let mut names = HashSet::new();
            for part in &layer.parts {
                text("a part name", &part.name)?;
                if !names.insert(&part.name) {
                    return Err(format!("{at}: part {} appears twice", part.name));
                }
                let Some(segment) = self.segments.get(part.segment as usize) else {
                    return Err(format!("{at}: part {} names an unknown segment", part.name));
                };
                if part.first_row + layer.experts as u64 > segment.rows {
                    return Err(format!(
                        "{at}: part {} needs rows {}..{} of {}, which has {}",
                        part.name,
                        part.first_row,
                        part.first_row + layer.experts as u64,
                        segment.name,
                        segment.rows
                    ));
                }
            }
            let part_bytes = |part: u32| -> Result<u64, String> {
                let part = layer
                    .parts
                    .get(part as usize)
                    .ok_or_else(|| format!("{at}: a view names an unknown part"))?;
                Ok(self.segments[part.segment as usize].row_bytes)
            };
            let mut views = HashSet::new();
            for view in &layer.views {
                text("a view name", &view.name)?;
                text("a view format", &view.format)?;
                if !views.insert(&view.name) {
                    return Err(format!("{at}: view {} appears twice", view.name));
                }
                if view.bytes == 0 || view.offset + view.bytes > part_bytes(view.part)? {
                    return Err(format!("{at}: view {} lies outside its part", view.name));
                }
                if let Some(scales) = &view.scales {
                    text("a scales format", &scales.format)?;
                    if scales.bytes == 0 || scales.offset + scales.bytes > part_bytes(scales.part)? {
                        return Err(format!("{at}: the scales of view {} lie outside their part", view.name));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Catalog {
        Catalog {
            format: FORMAT.into(),
            version: VERSION,
            name: "fixture".into(),
            files: vec![FileEntry {
                key: "a".into(),
                path: "a.bin".into(),
                size: Some(1000),
                sha256: None,
            }],
            segments: vec![SegmentEntry {
                name: "w".into(),
                files: vec![0],
                rows: 4,
                row_bytes: 100,
                offset: Some(0),
                part_bytes: None,
                spans: None,
            }],
            layers: vec![LayerEntry {
                layer: 0,
                experts: 4,
                parts: vec![PartEntry {
                    name: "w".into(),
                    segment: 0,
                    first_row: 0,
                }],
                views: vec![ViewEntry {
                    name: "gate".into(),
                    part: 0,
                    offset: 0,
                    bytes: 60,
                    format: "q4_row".into(),
                    rows: 10,
                    columns: 12,
                    block_rows: 0,
                    block_columns: 0,
                    scales: Some(ScalesEntry {
                        part: 0,
                        offset: 60,
                        bytes: 40,
                        format: "f32".into(),
                    }),
                }],
            }],
        }
    }

    #[test]
    fn a_catalog_round_trips_through_json() {
        let catalog = sample();
        assert_eq!(Catalog::from_json(&catalog.to_json()).unwrap(), catalog);
    }

    #[test]
    fn inconsistent_catalogs_are_refused() {
        let mut bad = sample();
        bad.version = 2;
        assert!(bad.validate().is_err());
        let mut bad = sample();
        bad.layers[0].experts = 5; // rows 0..5 of a 4-row segment
        assert!(bad.validate().unwrap_err().contains("which has 4"));
        let mut bad = sample();
        bad.layers[0].views[0].scales.as_mut().unwrap().offset = 70; // 70 + 40 bytes past a 100-byte part
        assert!(bad.validate().unwrap_err().contains("scales"));
        let mut bad = sample();
        bad.layers[0].views[0].bytes = 101;
        assert!(bad.validate().unwrap_err().contains("outside its part"));
        let mut bad = sample();
        bad.segments[0].files = vec![3];
        assert!(bad.validate().is_err());
        let mut bad = sample();
        bad.layers.push(bad.layers[0].clone());
        assert!(bad.validate().unwrap_err().contains("twice"));
        let mut bad = sample();
        bad.layers[0].views[0].name = "ga\0te".into();
        assert!(bad.validate().is_err());
        assert!(Catalog::from_json("{\"format\": 1}").is_err());
        let unknown = sample().to_json().replacen("\"name\"", "\"surprise\": 1, \"name\"", 1);
        assert!(Catalog::from_json(&unknown).is_err(), "unknown fields are refused");
    }

    #[test]
    fn composed_segments_are_checked() {
        let mut catalog = sample();
        catalog.segments[0] = SegmentEntry {
            name: "c".into(),
            files: vec![0],
            rows: 4,
            row_bytes: 100,
            offset: None,
            part_bytes: Some(vec![60, 40]),
            spans: Some((0..8).map(|k| (0, k * 100)).collect()),
        };
        catalog.validate().unwrap();
        catalog.segments[0].part_bytes = Some(vec![60, 41]);
        assert!(catalog.validate().is_err());
    }
}
