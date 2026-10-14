//! Colibrì's engine-neutral layer-segment runtime (`c/segment_runtime.h`, ABI v1) as Weightsift's `ExecutionBackend`.
//!
//! Colibrì owns the model math, weights, kernels and sequence state behind it; an adapter is registered by its engine
//! (each engine has an explicit registration function; nothing registers itself). `SegmentEngine::open` opens a layer
//! range of a registered engine; sessions borrow the engine, so it is never closed under a live session (Colibrì refuses
//! that too). The structs below mirror the header's, and the tests compare their layout with the C compiler's.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::path::Path;
use std::ptr::NonNull;

use weightsift_runtime::{Capabilities, DType, ExecutionBackend, ExecutionError, ExecutionSession, RunRequest};

pub const ABI_VERSION: u32 = 1;

pub const CAP_TOKEN_IDS: u64 = 1 << 0;
pub const CAP_SNAPSHOT: u64 = 1 << 1;
pub const CAP_RANGE_NATIVE: u64 = 1 << 2;
pub const CAP_MULTI_SESSION: u64 = 1 << 3;
const BACKENDS: [(u64, &str); 5] = [
    (1 << 8, "cpu"),
    (1 << 9, "cuda"),
    (1 << 10, "hip"),
    (1 << 11, "metal"),
    (1 << 12, "vulkan"),
];

#[repr(C)]
pub struct ColiSegmentEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ColiSegmentSession {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColiSegmentCapabilities {
    pub struct_size: u32,
    pub abi_version: u32,
    pub flags: u64,
    pub engine_id: [c_char; 64],
    pub state_schema: [c_char; 128],
    pub numeric_class: [c_char; 96],
    pub state_dtype: u32,
    pub state_width: u32,
    pub max_batch_rows: u32,
    pub max_context_tokens: u32,
    pub num_layers: u32,
    pub reserved_u32: [u32; 7],
    pub reserved_u64: [u64; 4],
}

#[repr(C)]
pub struct ColiSegmentEngineOptions {
    pub struct_size: u32,
    pub model_dir: *const c_char,
    pub layer_begin: u32,
    pub layer_end: u32,
    pub context_tokens: u32,
    pub reserved_u32: u32,
    pub memory_limit_bytes: u64,
    pub backend_mask: u64,
    pub resource_plan: *const c_void,
    pub resource_plan_size: usize,
}

#[repr(C)]
pub struct ColiSegmentSessionOptions {
    pub struct_size: u32,
    pub context_tokens: u32,
    pub memory_limit_bytes: u64,
    pub reserved_u64: [u64; 4],
}

pub type CancelFn = Option<unsafe extern "C" fn(user_data: *mut c_void) -> c_int>;
pub type WriteFn = unsafe extern "C" fn(user_data: *mut c_void, data: *const c_void, size: usize) -> c_int;
pub type ReadFn = unsafe extern "C" fn(user_data: *mut c_void, data: *mut c_void, size: usize) -> c_int;

#[repr(C)]
pub struct ColiSegmentRunRequest {
    pub struct_size: u32,
    pub rows: u32,
    pub position: u64,
    pub token_ids: *const i32,
    pub token_count: usize,
    pub input: *const c_void,
    pub input_bytes: usize,
    pub output: *mut c_void,
    pub output_bytes: usize,
    pub should_cancel: CancelFn,
    pub cancel_user_data: *mut c_void,
    pub reserved_u64: [u64; 4],
}

extern "C" {
    pub fn coli_segment_adapter_count() -> c_int;
    pub fn coli_segment_engine_open(
        engine_id: *const c_char,
        options: *const ColiSegmentEngineOptions,
        engine: *mut *mut ColiSegmentEngine,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn coli_segment_engine_capabilities(
        engine: *const ColiSegmentEngine,
        capabilities: *mut ColiSegmentCapabilities,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn coli_segment_engine_close(engine: *mut ColiSegmentEngine, error: *mut c_char, error_size: usize) -> c_int;
    pub fn coli_segment_session_create(
        engine: *mut ColiSegmentEngine,
        options: *const ColiSegmentSessionOptions,
        session: *mut *mut ColiSegmentSession,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn coli_segment_session_destroy(session: *mut ColiSegmentSession);
    pub fn coli_segment_run(
        session: *mut ColiSegmentSession,
        request: *const ColiSegmentRunRequest,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn coli_segment_snapshot(
        session: *mut ColiSegmentSession,
        write_fn: WriteFn,
        write_user_data: *mut c_void,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn coli_segment_restore(
        session: *mut ColiSegmentSession,
        read_fn: ReadFn,
        read_user_data: *mut c_void,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
}

/// An error buffer for one call, and the error it holds after a failure.
struct ErrorBuffer([c_char; 512]);

impl ErrorBuffer {
    fn new() -> Self {
        ErrorBuffer([0; 512])
    }

    fn ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr()
    }

    fn error(&self, what: &str) -> ExecutionError {
        // SAFETY: zero-initialized and written NUL-terminated by Colibrì (its set_error uses snprintf).
        let message = unsafe { CStr::from_ptr(self.0.as_ptr()) }.to_string_lossy();
        ExecutionError(format!("{what}: {message}"))
    }
}

fn fixed_string(field: &[c_char]) -> String {
    let bytes: Vec<u8> = field.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A layer range of a Colibrì engine registered with the segment runtime.
pub struct SegmentEngine {
    raw: NonNull<ColiSegmentEngine>,
    capabilities: Capabilities,
}

impl SegmentEngine {
    /// The number of adapters registered with Colibrì's segment runtime.
    pub fn registered() -> usize {
        // SAFETY: a read of Colibrì's registry count.
        unsafe { coli_segment_adapter_count() as usize }
    }

    pub fn open(
        engine_id: &str,
        model_dir: &Path,
        layers: Range<u32>,
        context_tokens: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, ExecutionError> {
        let id = CString::new(engine_id).map_err(|_| ExecutionError("the engine id holds a NUL".into()))?;
        let dir = CString::new(model_dir.to_string_lossy().as_bytes())
            .map_err(|_| ExecutionError("the model directory holds a NUL".into()))?;
        let options = ColiSegmentEngineOptions {
            struct_size: size_of::<ColiSegmentEngineOptions>() as u32,
            model_dir: dir.as_ptr(),
            layer_begin: layers.start,
            layer_end: layers.end,
            context_tokens,
            reserved_u32: 0,
            memory_limit_bytes,
            backend_mask: 0,
            resource_plan: std::ptr::null(),
            resource_plan_size: 0,
        };
        let mut raw = std::ptr::null_mut();
        let mut error = ErrorBuffer::new();
        // SAFETY: valid strings and options alive for the call; Colibrì copies what it keeps.
        let status = unsafe { coli_segment_engine_open(id.as_ptr(), &options, &mut raw, error.ptr(), 512) };
        let raw = match NonNull::new(raw) {
            Some(raw) if status == 0 => raw,
            _ => return Err(error.error("opening the segment engine")),
        };
        // From here the engine closes itself on any error (its drop).
        let mut engine = SegmentEngine {
            raw,
            capabilities: Capabilities {
                engine_id: String::new(),
                state_schema: String::new(),
                numeric_class: String::new(),
                dtype: DType::F32,
                state_width: 0,
                max_batch_rows: 0,
                max_context_tokens: 0,
                num_layers: 0,
                layer_begin: layers.start,
                layer_end: layers.end,
                token_ids: false,
                snapshots: false,
                range_native: false,
                multi_session: false,
                backends: Vec::new(),
            },
        };
        // SAFETY: zeroed is a valid bit pattern for this plain C struct; Colibrì fills it.
        let mut caps: ColiSegmentCapabilities = unsafe { std::mem::zeroed() };
        caps.struct_size = size_of::<ColiSegmentCapabilities>() as u32;
        // SAFETY: a live engine and a capabilities struct of the size it says.
        let status = unsafe { coli_segment_engine_capabilities(raw.as_ptr(), &mut caps, error.ptr(), 512) };
        if status != 0 {
            return Err(error.error("reading the capabilities"));
        }
        let c = &mut engine.capabilities;
        c.dtype = match caps.state_dtype {
            1 => DType::F32,
            2 => DType::F16,
            3 => DType::BF16,
            other => return Err(ExecutionError(format!("unknown state dtype {other}"))),
        };
        c.engine_id = fixed_string(&caps.engine_id);
        c.state_schema = fixed_string(&caps.state_schema);
        c.numeric_class = fixed_string(&caps.numeric_class);
        c.state_width = caps.state_width;
        c.max_batch_rows = caps.max_batch_rows;
        c.max_context_tokens = caps.max_context_tokens;
        c.num_layers = caps.num_layers;
        c.token_ids = caps.flags & CAP_TOKEN_IDS != 0;
        c.snapshots = caps.flags & CAP_SNAPSHOT != 0;
        c.range_native = caps.flags & CAP_RANGE_NATIVE != 0;
        c.multi_session = caps.flags & CAP_MULTI_SESSION != 0;
        c.backends = BACKENDS
            .iter()
            .filter(|(bit, _)| caps.flags & bit != 0)
            .map(|&(_, name)| name)
            .collect();
        Ok(engine)
    }
}

impl Drop for SegmentEngine {
    fn drop(&mut self) {
        // Sessions borrow the engine, so none is alive here and Colibrì's close cannot refuse.
        let mut error = ErrorBuffer::new();
        // SAFETY: a live engine, closed once.
        unsafe { coli_segment_engine_close(self.raw.as_ptr(), error.ptr(), 512) };
    }
}

/// A session of a `SegmentEngine`: one conversation's state in its layer range.
pub struct SegmentSession<'a> {
    raw: NonNull<ColiSegmentSession>,
    engine: PhantomData<&'a SegmentEngine>,
}

impl ExecutionBackend for SegmentEngine {
    type Session<'a> = SegmentSession<'a>;

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn session(&self, context_tokens: u32, memory_limit_bytes: u64) -> Result<SegmentSession<'_>, ExecutionError> {
        let options = ColiSegmentSessionOptions {
            struct_size: size_of::<ColiSegmentSessionOptions>() as u32,
            context_tokens,
            memory_limit_bytes,
            reserved_u64: [0; 4],
        };
        let mut raw = std::ptr::null_mut();
        let mut error = ErrorBuffer::new();
        // SAFETY: a live engine (borrowed for the session's life) and options alive for the call.
        let status = unsafe { coli_segment_session_create(self.raw.as_ptr(), &options, &mut raw, error.ptr(), 512) };
        match NonNull::new(raw) {
            Some(raw) if status == 0 => Ok(SegmentSession {
                raw,
                engine: PhantomData,
            }),
            _ => Err(error.error("creating a session")),
        }
    }
}

unsafe extern "C" fn write_into(user: *mut c_void, data: *const c_void, size: usize) -> c_int {
    // SAFETY: `user` is the Vec the snapshot call lent for its duration; `data` holds `size` bytes (Colibrì's contract).
    let out = unsafe { &mut *(user as *mut Vec<u8>) };
    out.extend_from_slice(unsafe { std::slice::from_raw_parts(data as *const u8, size) });
    0
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

unsafe extern "C" fn read_from(user: *mut c_void, data: *mut c_void, size: usize) -> c_int {
    // SAFETY: `user` is the Reader the restore call lent for its duration; `data` has room for `size` bytes.
    let reader = unsafe { &mut *(user as *mut Reader<'_>) };
    if reader.data.len() - reader.at < size {
        return -1; // the snapshot ends early
    }
    unsafe { std::ptr::copy_nonoverlapping(reader.data.as_ptr().add(reader.at), data as *mut u8, size) };
    reader.at += size;
    0
}

impl ExecutionSession for SegmentSession<'_> {
    fn run(&mut self, request: RunRequest<'_>) -> Result<(), ExecutionError> {
        let (token_ids, token_count) = match request.token_ids {
            Some(ids) => (ids.as_ptr(), ids.len()),
            None => (std::ptr::null(), 0),
        };
        let raw = ColiSegmentRunRequest {
            struct_size: size_of::<ColiSegmentRunRequest>() as u32,
            rows: request.rows,
            position: request.position,
            token_ids,
            token_count,
            input: request.input.as_ptr() as *const c_void,
            input_bytes: request.input.len(),
            output: request.output.as_mut_ptr() as *mut c_void,
            output_bytes: request.output.len(),
            should_cancel: None,
            cancel_user_data: std::ptr::null_mut(),
            reserved_u64: [0; 4],
        };
        let mut error = ErrorBuffer::new();
        // SAFETY: a live session; the buffers are borrowed for the call and their sizes are the slices' (Colibrì checks
        // them against the engine's width before touching them).
        let status = unsafe { coli_segment_run(self.raw.as_ptr(), &raw, error.ptr(), 512) };
        if status == 0 {
            Ok(())
        } else {
            Err(error.error("running the segment"))
        }
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, ExecutionError> {
        let mut out: Vec<u8> = Vec::new();
        let mut error = ErrorBuffer::new();
        // SAFETY: a live session; `out` outlives the call that writes into it through `write_into`.
        let status = unsafe {
            coli_segment_snapshot(
                self.raw.as_ptr(),
                write_into,
                &mut out as *mut Vec<u8> as *mut c_void,
                error.ptr(),
                512,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(error.error("taking a snapshot"))
        }
    }

    fn restore(&mut self, snapshot: &[u8]) -> Result<(), ExecutionError> {
        let mut reader = Reader { data: snapshot, at: 0 };
        let mut error = ErrorBuffer::new();
        // SAFETY: a live session; `reader` outlives the call that reads from it through `read_from`.
        let status = unsafe {
            coli_segment_restore(
                self.raw.as_ptr(),
                read_from,
                &mut reader as *mut Reader<'_> as *mut c_void,
                error.ptr(),
                512,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(error.error("restoring a snapshot"))
        }
    }
}

impl Drop for SegmentSession<'_> {
    fn drop(&mut self) {
        // SAFETY: a live session, destroyed once, before its engine (it borrows it).
        unsafe { coli_segment_session_destroy(self.raw.as_ptr()) }
    }
}
