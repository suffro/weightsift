//! The prototype's block record (`LMC1`): everything a decoder needs, every byte counted.
//!
//! ```text
//! offset 0   magic "LMC1"
//!        4   version u8 (1), alphabet u16, scale_bits u8, context kind u8 (0 none, 1 column, 2 row, 3 left,
//!            4 grid: a class per (row, group of columns) that the decoder derives from data it already holds, such
//!            as the stored scales), flags u8 (bit 0: tables are external, shared by many blocks and counted once
//!            elsewhere), n_contexts u16, rows u32, cols u32, streams u16                       (22 bytes in all)
//!       22   side information: a class (u8) per column (kind 1) or per row (kind 2); for kind 4 the group width
//!            (u32), the classes themselves never being stored; nothing otherwise
//!            tables, unless external: a bitmap of used contexts (ceil(n_contexts / 8) bytes), then for each used
//!            context `alphabet` frequencies (u16 little-endian)
//!            stream end offsets: `streams` u32 little-endian, relative to the payload
//!            payload: the streams, concatenated
//! ```
//! Parsing refuses any inconsistency (lengths, offsets, tables that do not sum to 2^scale_bits) by name.

use crate::rans::{self, Ctx, Table};

pub const MAGIC: &[u8; 4] = b"LMC1";
pub const HEADER_BYTES: usize = 22;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub alphabet: u16,
    pub scale_bits: u8,
    pub kind: u8,
    pub external_tables: bool,
    pub n_contexts: u16,
    pub rows: u32,
    pub cols: u32,
    pub side: Vec<u8>,
    /// Kind 4: the derived classes (never serialized: the decoder supplies them) and the group width.
    pub grid: Vec<u8>,
    pub grid_group: u32,
    pub tables: Vec<Table>,
    pub offsets: Vec<u32>,
    pub payload: Vec<u8>,
}

/// Byte accounting of a serialized block.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Sizes {
    pub header: usize,
    pub side: usize,
    pub tables: usize,
    pub offsets: usize,
    pub payload: usize,
    pub total: usize,
}

impl Block {
    pub fn ctx(&self) -> Ctx<'_> {
        match self.kind {
            1 => Ctx::Column(&self.side),
            2 => Ctx::Row(&self.side),
            3 => Ctx::Left {
                alphabet: self.alphabet as usize,
            },
            4 => Ctx::Grid {
                classes: &self.grid,
                group: self.grid_group as usize,
                per_row: (self.cols as usize).div_ceil(self.grid_group as usize),
            },
            _ => Ctx::None,
        }
    }

    pub fn sizes(&self) -> Sizes {
        let tables = if self.external_tables {
            0
        } else {
            (self.n_contexts as usize).div_ceil(8)
                + self.tables.iter().filter(|t| t.used()).count() * self.alphabet as usize * 2
        };
        let s = Sizes {
            header: HEADER_BYTES,
            side: self.side.len() + if self.kind == 4 { 4 } else { 0 },
            tables,
            offsets: self.offsets.len() * 4,
            payload: self.payload.len(),
            total: 0,
        };
        Sizes {
            total: s.header + s.side + s.tables + s.offsets + s.payload,
            ..s
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.sizes().total);
        out.extend_from_slice(MAGIC);
        out.push(1);
        out.extend_from_slice(&self.alphabet.to_le_bytes());
        out.push(self.scale_bits);
        out.push(self.kind);
        out.push(self.external_tables as u8);
        out.extend_from_slice(&self.n_contexts.to_le_bytes());
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.cols.to_le_bytes());
        out.extend_from_slice(&(self.offsets.len() as u16).to_le_bytes());
        out.extend_from_slice(&self.side);
        if self.kind == 4 {
            out.extend_from_slice(&self.grid_group.to_le_bytes());
        }
        if !self.external_tables {
            let mut bitmap = vec![0u8; (self.n_contexts as usize).div_ceil(8)];
            for (i, t) in self.tables.iter().enumerate() {
                if t.used() {
                    bitmap[i / 8] |= 1 << (i % 8);
                }
            }
            out.extend_from_slice(&bitmap);
            for t in self.tables.iter().filter(|t| t.used()) {
                for &f in &t.freq {
                    out.extend_from_slice(&(f as u16).to_le_bytes());
                }
            }
        }
        for &o in &self.offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse and validate; external tables must be given when the block says so.
    pub fn parse(buf: &[u8], external: Option<&[Table]>) -> Result<Block, String> {
        Self::parse_with(buf, external, None)
    }

    /// Parse and validate, supplying the derived classes of a kind 4 block (the decoder computes them from the
    /// scales it holds).
    pub fn parse_with(buf: &[u8], external: Option<&[Table]>, grid: Option<&[u8]>) -> Result<Block, String> {
        let need = |n: usize, at: usize| -> Result<(), String> {
            if at + n > buf.len() {
                Err(format!("truncated at byte {at} (need {n} more)"))
            } else {
                Ok(())
            }
        };
        need(HEADER_BYTES, 0)?;
        if &buf[0..4] != MAGIC || buf[4] != 1 {
            return Err("bad magic or version".into());
        }
        let alphabet = u16::from_le_bytes([buf[5], buf[6]]);
        let scale_bits = buf[7];
        let kind = buf[8];
        let external_tables = buf[9] & 1 == 1;
        if buf[9] & !1 != 0 || kind > 4 || alphabet == 0 || alphabet > 256 {
            return Err("bad header fields".into());
        }
        if !(1..=rans::MAX_SCALE_BITS as u8).contains(&scale_bits) {
            return Err("bad scale_bits".into());
        }
        let n_contexts = u16::from_le_bytes([buf[10], buf[11]]);
        let rows = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let cols = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let streams = u16::from_le_bytes([buf[20], buf[21]]) as usize;
        if streams == 0 || n_contexts == 0 {
            return Err("no streams or no contexts".into());
        }
        let expected_contexts = match kind {
            0 => Some(1usize),
            3 => Some(alphabet as usize + 1),
            _ => None,
        };
        if let Some(n) = expected_contexts {
            if n_contexts as usize != n {
                return Err("context count does not match the kind".into());
            }
        }
        let mut at = HEADER_BYTES;
        let side_len = match kind {
            1 => cols as usize,
            2 => rows as usize,
            _ => 0,
        };
        need(side_len, at)?;
        let side = buf[at..at + side_len].to_vec();
        at += side_len;
        if side.iter().any(|&c| c as u16 >= n_contexts) {
            return Err("a class outside the context count".into());
        }
        let (grid, grid_group) = if kind == 4 {
            need(4, at)?;
            let g = u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
            at += 4;
            if g == 0 {
                return Err("a zero group width".into());
            }
            let classes = grid.ok_or("the block needs its derived classes")?;
            let per_row = (cols as usize).div_ceil(g as usize);
            if classes.len() != rows as usize * per_row || classes.iter().any(|&c| c as u16 >= n_contexts) {
                return Err("derived classes do not match the block".into());
            }
            (classes.to_vec(), g)
        } else {
            (Vec::new(), 0)
        };
        let tables = if external_tables {
            let ext = external.ok_or("the block needs external tables")?;
            if ext.len() != n_contexts as usize {
                return Err("external tables do not match the context count".into());
            }
            ext.to_vec()
        } else {
            let bitmap_len = (n_contexts as usize).div_ceil(8);
            need(bitmap_len, at)?;
            let bitmap = buf[at..at + bitmap_len].to_vec();
            at += bitmap_len;
            let mut tables = Vec::with_capacity(n_contexts as usize);
            for i in 0..n_contexts as usize {
                if bitmap[i / 8] >> (i % 8) & 1 == 1 {
                    need(alphabet as usize * 2, at)?;
                    let freq: Vec<u32> = (0..alphabet as usize)
                        .map(|k| u16::from_le_bytes([buf[at + 2 * k], buf[at + 2 * k + 1]]) as u32)
                        .collect();
                    at += alphabet as usize * 2;
                    let t = Table::from_freq(freq, scale_bits as u32)?;
                    if !t.used() {
                        return Err(format!("context {i} is marked used but empty"));
                    }
                    tables.push(t);
                } else {
                    tables.push(Table::from_freq(vec![0; alphabet as usize], scale_bits as u32)?);
                }
            }
            tables
        };
        need(streams * 4, at)?;
        let offsets: Vec<u32> = (0..streams)
            .map(|i| u32::from_le_bytes(buf[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
            .collect();
        at += streams * 4;
        let mut prev = 0u32;
        for &o in &offsets {
            if o < prev + 4 {
                return Err("stream offsets not increasing by at least 4".into());
            }
            prev = o;
        }
        let payload_len = *offsets.last().unwrap() as usize;
        if at + payload_len != buf.len() {
            return Err(format!(
                "length mismatch: {} bytes, framing says {}",
                buf.len(),
                at + payload_len
            ));
        }
        let payload = buf[at..].to_vec();
        Ok(Block {
            alphabet,
            scale_bits,
            kind,
            external_tables,
            n_contexts,
            rows,
            cols,
            side,
            grid,
            grid_group,
            tables,
            offsets,
            payload,
        })
    }
}

/// Counts [n_contexts][alphabet] of a matrix's symbols under a context.
pub fn context_counts(
    syms: &[u8],
    rows: usize,
    cols: usize,
    ctx: &Ctx,
    n_contexts: usize,
    alphabet: usize,
) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![0u64; alphabet]; n_contexts];
    for r in 0..rows {
        let row = &syms[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let prev = if c > 0 { row[c - 1] } else { 0 };
            counts[ctx.id(r, c, prev)][row[c] as usize] += 1;
        }
    }
    counts
}

/// Encode a matrix. `tables` None: tables from the matrix's own counts, stored in the block (a two-part code);
/// Some: given tables (e.g., trained on calibration data), not stored (`external_tables`). `grid`: the derived classes
/// and group width of a kind 4 block (required for that kind only).
#[allow(clippy::too_many_arguments)]
pub fn encode(
    syms: &[u8],
    rows: usize,
    cols: usize,
    alphabet: usize,
    kind: u8,
    side: Vec<u8>,
    n_contexts: usize,
    tables: Option<Vec<Table>>,
    scale_bits: u32,
    streams: usize,
    grid: Option<(Vec<u8>, u32)>,
) -> Result<Block, String> {
    if syms.len() != rows * cols {
        return Err("symbol count does not match the shape".into());
    }
    if syms.iter().any(|&s| s as usize >= alphabet) {
        return Err("a symbol outside the alphabet".into());
    }
    let (grid, grid_group) = match (kind, grid) {
        (4, Some((classes, group))) => {
            let per_row = cols.div_ceil(group.max(1) as usize);
            if group == 0 || classes.len() != rows * per_row || classes.iter().any(|&c| c as usize >= n_contexts) {
                return Err("derived classes do not match the matrix".into());
            }
            (classes, group)
        }
        (4, None) => return Err("a grid block needs its classes".into()),
        _ => (Vec::new(), 0),
    };
    let streams = streams.min(rows).max(1);
    let ctx = match kind {
        1 => Ctx::Column(&side),
        2 => Ctx::Row(&side),
        3 => Ctx::Left { alphabet },
        4 => Ctx::Grid {
            classes: &grid,
            group: grid_group as usize,
            per_row: cols.div_ceil(grid_group as usize),
        },
        _ => Ctx::None,
    };
    let (tables, external) = match tables {
        Some(t) => (t, true),
        None => {
            let counts = context_counts(syms, rows, cols, &ctx, n_contexts, alphabet);
            let t = counts
                .iter()
                .map(|c| rans::normalize(c, scale_bits).and_then(|f| Table::from_freq(f, scale_bits)))
                .collect::<Result<Vec<_>, _>>()?;
            (t, false)
        }
    };
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(streams);
    for s in 0..streams {
        payload.extend(rans::encode_stream(syms, rows, cols, s, streams, &ctx, &tables)?);
        offsets.push(payload.len() as u32);
    }
    Ok(Block {
        alphabet: alphabet as u16,
        scale_bits: scale_bits as u8,
        kind,
        external_tables: external,
        n_contexts: n_contexts as u16,
        rows: rows as u32,
        cols: cols as u32,
        side,
        grid,
        grid_group,
        tables,
        offsets,
        payload,
    })
}

/// Decode a parsed block into a sink (the decode tables are built here, from the block's tables).
pub fn decode<S: rans::Sink>(block: &Block, sink: &mut S, checked: bool) -> Result<(), String> {
    rans::decode_block(
        &block.payload,
        &block.offsets,
        block.rows as usize,
        block.cols as usize,
        &block.ctx(),
        &block.tables,
        block.scale_bits as u32,
        sink,
        checked,
    )
}
