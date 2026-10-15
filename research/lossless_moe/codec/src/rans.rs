//! Static rANS: 32-bit state, byte renormalization, L = 2^23 (Fabian Giesen's ryg_rans construction, the one Colibri's
//! `rans.h` uses), with one frequency table per context.
//!
//! A block is a matrix of `rows` x `cols` symbols. Row `r` belongs to stream `r % streams`; a stream holds its rows in
//! ascending order, each left to right, so a decoder that runs several streams in lockstep still sees each row's
//! previous symbol (the `Left` context). Each stream is an independent rANS stream: its forward bytes begin with the
//! final encoder state (4 bytes, little-endian) and end exactly where its last renormalization byte was consumed; a
//! correct decode ends with the state equal to L, which the checked decoder verifies.

pub const RANS_L: u32 = 1 << 23;
pub const MAX_SCALE_BITS: u32 = 15;

/// One context's table: frequencies summing to `1 << scale_bits`, or all zero (a context that never occurs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub scale_bits: u32,
    pub freq: Vec<u32>,
    pub start: Vec<u32>,
}

impl Table {
    pub fn from_freq(freq: Vec<u32>, scale_bits: u32) -> Result<Self, String> {
        if !(1..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(format!("scale_bits {scale_bits} outside 1..={MAX_SCALE_BITS}"));
        }
        let total: u64 = freq.iter().map(|&f| f as u64).sum();
        if total != 0 && total != 1u64 << scale_bits {
            return Err(format!("frequencies sum to {total}, not {}", 1u64 << scale_bits));
        }
        let mut start = Vec::with_capacity(freq.len());
        let mut acc = 0u32;
        for &f in &freq {
            start.push(acc);
            acc += f;
        }
        Ok(Table {
            scale_bits,
            freq,
            start,
        })
    }

    pub fn used(&self) -> bool {
        self.freq.iter().any(|&f| f != 0)
    }
}

/// Frequencies summing to `1 << scale_bits` from counts. Every symbol with a nonzero count gets at least 1; absent
/// symbols get 0 (largest remainder, ties by symbol index). All-zero counts give an all-zero (unused) table.
pub fn normalize(counts: &[u64], scale_bits: u32) -> Result<Vec<u32>, String> {
    let m = 1u64 << scale_bits;
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return Ok(vec![0; counts.len()]);
    }
    let present = counts.iter().filter(|&&c| c > 0).count() as u64;
    if present > m {
        return Err(format!("{present} symbols cannot fit a table of {m} slots"));
    }
    let mut freq: Vec<u64> = counts
        .iter()
        .map(|&c| (c as u128 * m as u128 / total as u128) as u64)
        .collect();
    for (f, &c) in freq.iter_mut().zip(counts) {
        if c > 0 && *f == 0 {
            *f = 1;
        }
    }
    let sum: u64 = freq.iter().sum();
    if sum < m {
        // largest fractional remainder first, ties by index
        let mut order: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
        let rem = |i: usize| (counts[i] as u128 * m as u128) % total as u128;
        order.sort_by(|&a, &b| rem(b).cmp(&rem(a)).then(a.cmp(&b)));
        let mut deficit = m - sum;
        let mut k = 0;
        while deficit > 0 {
            freq[order[k % order.len()]] += 1;
            deficit -= 1;
            k += 1;
        }
    } else if sum > m {
        // take from the largest frequencies (each keeps at least 1)
        let mut excess = sum - m;
        while excess > 0 {
            let i = (0..freq.len())
                .max_by(|&a, &b| freq[a].cmp(&freq[b]).then(b.cmp(&a)))
                .expect("nonempty");
            if freq[i] <= 1 {
                return Err("cannot normalize".into());
            }
            freq[i] -= 1;
            excess -= 1;
        }
    }
    Ok(freq.into_iter().map(|f| f as u32).collect())
}

/// Ideal code length in bits of `counts` under `freq` (what the coder spends, renormalization aside).
pub fn cost_bits(counts: &[u64], freq: &[u32], scale_bits: u32) -> f64 {
    let m = (1u64 << scale_bits) as f64;
    counts
        .iter()
        .zip(freq)
        .filter(|(&c, _)| c > 0)
        .map(|(&c, &f)| c as f64 * (m / f as f64).log2())
        .sum()
}

/// A context's decode table: per slot, `freq | (slot - start) << 16` and the symbol.
#[derive(Clone)]
pub struct DecodeTable {
    pub entry: Vec<u32>,
    pub sym: Vec<u8>,
}

impl DecodeTable {
    pub fn new(t: &Table) -> Self {
        let m = 1usize << t.scale_bits;
        let mut entry = vec![0u32; m];
        let mut sym = vec![0u8; m];
        if t.used() {
            for (s, (&f, &st)) in t.freq.iter().zip(&t.start).enumerate() {
                for slot in st..st + f {
                    entry[slot as usize] = f | ((slot - st) << 16);
                    sym[slot as usize] = s as u8;
                }
            }
        }
        DecodeTable { entry, sym }
    }
}

/// How a symbol's table is chosen.
#[derive(Clone, Copy, Debug)]
pub enum Ctx<'a> {
    /// One table.
    None,
    /// A class per column (side information).
    Column(&'a [u8]),
    /// A class per row (side information).
    Row(&'a [u8]),
    /// The previous symbol of the row; `alphabet` at the row's start.
    Left { alphabet: usize },
    /// A class per (row, group of `group` columns), derived by the decoder from data it already holds (the stored
    /// scales): `classes[r * per_row + c / group]`.
    Grid {
        classes: &'a [u8],
        group: usize,
        per_row: usize,
    },
}

impl Ctx<'_> {
    #[inline(always)]
    pub fn id(&self, r: usize, c: usize, prev: u8) -> usize {
        match *self {
            Ctx::None => 0,
            Ctx::Column(cls) => cls[c] as usize,
            Ctx::Row(cls) => cls[r] as usize,
            Ctx::Left { alphabet } => {
                if c == 0 {
                    alphabet
                } else {
                    prev as usize
                }
            }
            Ctx::Grid {
                classes,
                group,
                per_row,
            } => classes[r * per_row + c / group] as usize,
        }
    }

    pub fn kind(&self) -> u8 {
        match self {
            Ctx::None => 0,
            Ctx::Column(_) => 1,
            Ctx::Row(_) => 2,
            Ctx::Left { .. } => 3,
            Ctx::Grid { .. } => 4,
        }
    }
}

/// Rows of stream `s` among `streams`, ascending.
pub fn stream_rows(s: usize, rows: usize, streams: usize) -> impl DoubleEndedIterator<Item = usize> {
    (s..rows).step_by(streams)
}

/// Encode one stream (its rows, in decode order) into forward bytes.
pub fn encode_stream(
    syms: &[u8],
    rows: usize,
    cols: usize,
    s: usize,
    streams: usize,
    ctx: &Ctx,
    tables: &[Table],
) -> Result<Vec<u8>, String> {
    let mut x = RANS_L;
    let mut rev: Vec<u8> = Vec::with_capacity(cols * rows.div_ceil(streams));
    for r in stream_rows(s, rows, streams).rev() {
        let row = &syms[r * cols..(r + 1) * cols];
        for c in (0..cols).rev() {
            let sym = row[c] as usize;
            let prev = if c > 0 { row[c - 1] } else { 0 };
            let t = &tables[ctx.id(r, c, prev)];
            let f = *t.freq.get(sym).ok_or("symbol outside the alphabet")?;
            if f == 0 {
                return Err(format!(
                    "symbol {sym} has no frequency in its context (row {r}, column {c})"
                ));
            }
            let x_max = ((RANS_L >> t.scale_bits) << 8) * f;
            while x >= x_max {
                rev.push(x as u8);
                x >>= 8;
            }
            x = ((x / f) << t.scale_bits) + (x % f) + t.start[sym];
        }
    }
    rev.extend_from_slice(&[(x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, x as u8]);
    rev.reverse();
    Ok(rev)
}

/// Where decoded symbols go: one per position `r * cols + c`.
pub trait Sink {
    fn put(&mut self, pos: usize, sym: u8);
    /// Whether the output holds `n` symbols (checked once, so that `put` may skip bounds checks).
    fn fits(&self, n: usize) -> bool;
}

/// One byte per symbol.
pub struct Bytes<'a>(pub &'a mut [u8]);
impl Sink for Bytes<'_> {
    #[inline(always)]
    fn put(&mut self, pos: usize, sym: u8) {
        // SAFETY: pos < n and `fits(n)` held before decoding.
        unsafe { *self.0.get_unchecked_mut(pos) = sym }
    }
    fn fits(&self, n: usize) -> bool {
        self.0.len() == n
    }
}

/// BF16 patterns from the decoded exponent and the stored sign-and-mantissa byte ((s << 7) | m).
pub struct Bf16<'a> {
    pub raw: &'a [u8],
    pub out: &'a mut [u16],
}
impl Sink for Bf16<'_> {
    #[inline(always)]
    fn put(&mut self, pos: usize, e: u8) {
        // SAFETY: pos < n and `fits(n)` held before decoding (both slices hold n entries).
        unsafe {
            let sm = *self.raw.get_unchecked(pos) as u16;
            *self.out.get_unchecked_mut(pos) = ((sm >> 7) << 15) | ((e as u16) << 7) | (sm & 0x7F);
        }
    }
    fn fits(&self, n: usize) -> bool {
        self.raw.len() == n && self.out.len() == n
    }
}

/// Packed nibbles, low nibble first (positions come in order within each row).
pub struct Nibbles<'a>(pub &'a mut [u8]);
impl Sink for Nibbles<'_> {
    #[inline(always)]
    fn put(&mut self, pos: usize, sym: u8) {
        // SAFETY: pos < n and `fits(n)` held before decoding (n even, n / 2 bytes).
        unsafe {
            let b = self.0.get_unchecked_mut(pos >> 1);
            if pos & 1 == 0 {
                *b = sym;
            } else {
                *b |= sym << 4;
            }
        }
    }
    fn fits(&self, n: usize) -> bool {
        n.is_multiple_of(2) && self.0.len() == n / 2
    }
}

const GROUP: usize = 8;

/// Every context's decode table in one allocation: entry and symbol of slot `s` of context `c` at `c * m + s`.
pub struct Flat {
    pub m: usize,
    pub entry: Vec<u32>,
    pub sym: Vec<u8>,
}

impl Flat {
    /// Decode entries of every context, from the tables themselves; an unused context's slots stay zero (never
    /// reached by a valid stream, and harmless to a corrupted one: frequency zero, bias zero).
    pub fn new(tables: &[Table], scale_bits: u32) -> Self {
        let m = 1usize << scale_bits;
        let mut entry = vec![0u32; m * tables.len()];
        let mut sym = vec![0u8; m * tables.len()];
        for (c, t) in tables.iter().enumerate() {
            if !t.used() {
                continue;
            }
            let base = c * m;
            for (s, (&f, &st)) in t.freq.iter().zip(&t.start).enumerate() {
                for slot in st..st + f {
                    entry[base + slot as usize] = f | ((slot - st) << 16);
                    sym[base + slot as usize] = s as u8;
                }
            }
        }
        Flat { m, entry, sym }
    }
}

/// A context as the decoder computes it; specialized at compile time.
trait Pick {
    fn pick(&self, r: usize, c: usize, prev: u8) -> usize;
}

struct PickNone;

impl Pick for PickNone {
    #[inline(always)]
    fn pick(&self, _: usize, _: usize, _: u8) -> usize {
        0
    }
}

struct PickColumn<'a>(&'a [u8]);

impl Pick for PickColumn<'_> {
    #[inline(always)]
    fn pick(&self, _: usize, c: usize, _: u8) -> usize {
        // SAFETY: c < cols == self.0.len(), checked by `decode_block` before decoding.
        unsafe { *self.0.get_unchecked(c) as usize }
    }
}

struct PickRow<'a>(&'a [u8]);

impl Pick for PickRow<'_> {
    #[inline(always)]
    fn pick(&self, r: usize, _: usize, _: u8) -> usize {
        // SAFETY: r < rows == self.0.len(), checked by `decode_block` before decoding.
        unsafe { *self.0.get_unchecked(r) as usize }
    }
}

struct PickGrid<'a> {
    classes: &'a [u8],
    group: usize,
    per_row: usize,
}

impl Pick for PickGrid<'_> {
    #[inline(always)]
    fn pick(&self, r: usize, c: usize, _: u8) -> usize {
        // SAFETY: r < rows, c / group < per_row and classes.len() == rows * per_row, checked by `decode_block`.
        unsafe { *self.classes.get_unchecked(r * self.per_row + c / self.group) as usize }
    }
}

struct PickLeft(usize);

impl Pick for PickLeft {
    #[inline(always)]
    fn pick(&self, _: usize, c: usize, prev: u8) -> usize {
        if c == 0 {
            self.0
        } else {
            prev as usize
        }
    }
}

/// Decode every stream of a block into `sink`, `GROUP` streams in lockstep. `checked` verifies that every stream is
/// consumed exactly and ends in state L (a corrupted stream almost never does).
#[allow(clippy::too_many_arguments)]
pub fn decode_block<S: Sink>(
    payload: &[u8],
    offsets: &[u32],
    rows: usize,
    cols: usize,
    ctx: &Ctx,
    tables: &[Table],
    scale_bits: u32,
    sink: &mut S,
    checked: bool,
) -> Result<(), String> {
    if !sink.fits(rows * cols) {
        return Err("output buffer of the wrong size".into());
    }
    if !(1..=MAX_SCALE_BITS).contains(&scale_bits) {
        return Err("bad scale_bits".into());
    }
    if tables.iter().any(|t| t.used() && t.scale_bits != scale_bits) {
        return Err("tables do not match the scale".into());
    }
    let flat = Flat::new(tables, scale_bits);
    let n_ctx = tables.len();
    // Every context id the decoder can compute must have a table.
    let ok = match *ctx {
        Ctx::None => n_ctx >= 1,
        Ctx::Column(cls) => cls.len() == cols && cls.iter().all(|&k| (k as usize) < n_ctx),
        Ctx::Row(cls) => cls.len() == rows && cls.iter().all(|&k| (k as usize) < n_ctx),
        Ctx::Left { alphabet } => {
            alphabet <= 256 && n_ctx > alphabet && flat.sym.iter().all(|&s| (s as usize) < alphabet)
        }
        Ctx::Grid {
            classes,
            group,
            per_row,
        } => {
            group > 0
                && per_row == cols.div_ceil(group)
                && classes.len() == rows * per_row
                && classes.iter().all(|&k| (k as usize) < n_ctx)
        }
    };
    if !ok {
        return Err("contexts and tables do not match".into());
    }
    let mut begin = 0usize;
    let mut bounds = Vec::with_capacity(offsets.len());
    for &end in offsets {
        let end = end as usize;
        if end < begin + 4 || end > payload.len() {
            return Err("malformed stream offsets".into());
        }
        bounds.push((begin, end));
        begin = end;
    }
    match *ctx {
        Ctx::None => run(
            payload, &bounds, rows, cols, &PickNone, &flat, scale_bits, sink, checked,
        ),
        Ctx::Column(cls) => run(
            payload,
            &bounds,
            rows,
            cols,
            &PickColumn(cls),
            &flat,
            scale_bits,
            sink,
            checked,
        ),
        Ctx::Row(cls) => run(
            payload,
            &bounds,
            rows,
            cols,
            &PickRow(cls),
            &flat,
            scale_bits,
            sink,
            checked,
        ),
        Ctx::Left { alphabet } => run(
            payload,
            &bounds,
            rows,
            cols,
            &PickLeft(alphabet),
            &flat,
            scale_bits,
            sink,
            checked,
        ),
        Ctx::Grid {
            classes,
            group,
            per_row,
        } => {
            let pick = PickGrid {
                classes,
                group,
                per_row,
            };
            run(payload, &bounds, rows, cols, &pick, &flat, scale_bits, sink, checked)
        }
    }
}

/// One symbol of one lane: the table lookup, the state update, the renormalization (at most two bytes for
/// scale_bits <= 15: after the update x >= 2^(23 - scale_bits) >= 2^8). With four readable bytes ahead in the stream the
/// refill is branch-free; near a stream's end it goes byte by byte, never past the end.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step(x: &mut u32, ptr: &mut usize, end: usize, payload: &[u8], flat: &Flat, ctx: usize, mask: u32, sb: u32) -> u8 {
    let slot = (*x & mask) as usize;
    let at = ctx * flat.m + slot;
    // SAFETY: ctx < the number of tables (validated per context kind in `decode_block`) and slot < m, so
    // at < entry.len() == sym.len().
    let (e, s) = unsafe { (*flat.entry.get_unchecked(at), *flat.sym.get_unchecked(at)) };
    let mut xx = (e & 0xFFFF) * (*x >> sb) + (e >> 16);
    if *ptr + 4 <= end {
        // SAFETY: ptr + 4 <= end <= payload.len().
        let word = unsafe { payload.get_unchecked(*ptr..*ptr + 4) };
        let be = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        let n = (xx < RANS_L) as u32 + (xx < (RANS_L >> 8)) as u32;
        xx = (((xx as u64) << (8 * n)) | ((be as u64) >> (32 - 8 * n))) as u32;
        *ptr += n as usize;
    } else {
        while xx < RANS_L && *ptr < end {
            // SAFETY: *ptr < end <= payload.len().
            xx = (xx << 8) | unsafe { *payload.get_unchecked(*ptr) } as u32;
            *ptr += 1;
        }
    }
    *x = xx;
    s
}

#[allow(clippy::too_many_arguments)]
fn run<P: Pick, S: Sink>(
    payload: &[u8],
    bounds: &[(usize, usize)],
    rows: usize,
    cols: usize,
    pick: &P,
    flat: &Flat,
    sb: u32,
    sink: &mut S,
    checked: bool,
) -> Result<(), String> {
    let streams = bounds.len();
    let mask = (1u32 << sb) - 1;
    for g in (0..streams).step_by(GROUP) {
        let lanes = GROUP.min(streams - g);
        let mut x = [RANS_L; GROUP];
        let mut ptr = [0usize; GROUP];
        let mut end = [0usize; GROUP];
        let mut prev = [0u8; GROUP];
        let mut nrows = [0usize; GROUP];
        for l in 0..lanes {
            let (b, e) = bounds[g + l];
            x[l] = u32::from_le_bytes([payload[b], payload[b + 1], payload[b + 2], payload[b + 3]]);
            ptr[l] = b + 4;
            end[l] = e;
            nrows[l] = stream_rows(g + l, rows, streams).count();
        }
        let common = if lanes == GROUP {
            nrows.iter().copied().min().unwrap_or(0)
        } else {
            0
        };
        for k in 0..common {
            let base = g + k * streams;
            for c in 0..cols {
                for l in 0..GROUP {
                    let r = base + l;
                    let ctx = pick.pick(r, c, prev[l]);
                    let s = step(&mut x[l], &mut ptr[l], end[l], payload, flat, ctx, mask, sb);
                    prev[l] = s;
                    sink.put(r * cols + c, s);
                }
            }
        }
        for l in 0..lanes {
            for k in common..nrows[l] {
                let r = g + l + k * streams;
                for c in 0..cols {
                    let ctx = pick.pick(r, c, prev[l]);
                    let s = step(&mut x[l], &mut ptr[l], end[l], payload, flat, ctx, mask, sb);
                    prev[l] = s;
                    sink.put(r * cols + c, s);
                }
            }
            if checked && (ptr[l] != end[l] || x[l] != RANS_L) {
                return Err(format!(
                    "stream {} did not end cleanly (consumed {} of {} bytes, final state {:#x})",
                    g + l,
                    ptr[l] - bounds[g + l].0,
                    end[l] - bounds[g + l].0,
                    x[l]
                ));
            }
        }
    }
    Ok(())
}
