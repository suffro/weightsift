//! Process CPU time and peak working set (Windows; other platforms report nothing).

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    pub struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        pub fn seconds(&self) -> f64 {
            (((self.high as u64) << 32) | self.low as u64) as f64 * 1e-7
        }
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct MemoryCounters {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        pub fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut MemoryCounters, cb: u32) -> i32;
    }
}

/// User plus kernel CPU seconds of this process so far.
pub fn process_cpu_seconds() -> f64 {
    #[cfg(windows)]
    {
        let (mut c, mut e, mut k, mut u) = Default::default();
        // SAFETY: the pseudo-handle of the current process and four writable FILETIMEs.
        let ok = unsafe { win::GetProcessTimes(win::GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        if ok != 0 {
            let _ = (c, e);
            return k.seconds() + u.seconds();
        }
    }
    f64::NAN
}

/// Peak working set of this process in bytes, if known.
pub fn peak_working_set() -> Option<usize> {
    #[cfg(windows)]
    {
        let mut m = win::MemoryCounters {
            cb: std::mem::size_of::<win::MemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: the pseudo-handle of the current process and a writable, correctly sized counters struct.
        let ok = unsafe { win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut m, m.cb) };
        if ok != 0 {
            return Some(m.peak_working_set_size);
        }
    }
    None
}
