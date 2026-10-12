use std::time::Instant;

/*
    PROCESS USAGE

    cpu and memory of this process, read straight from the os, no crates needed
    only sampled a couple times a second, and only while the performance overlay is up
    macos only for now, anything else reports zeroes and the overlay just says N/A
*/

pub struct Usage {
    /// process cpu time over wall time, 1.0 = one core fully busy
    pub cores_busy: f32,
    /// resident memory of the process (bytes)
    pub resident: u64,
    /// physical memory of the machine (bytes), 0 if we couldn't find out
    pub total: u64,
    /// hardware threads of the machine
    pub cores: usize,

    last_wall: Instant,
    /// process cpu seconds at last_wall
    last_cpu: f64,
}

impl Usage {
    /// seconds between samples
    const INTERVAL: f32 = 0.5;

    pub fn new() -> Self {
        Self {
            cores_busy: 0.0,
            resident: os::resident_bytes(),
            total: os::total_bytes(),
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
            last_wall: Instant::now(),
            last_cpu: os::cpu_seconds(),
        }
    }

    /// resample, but only once INTERVAL has gone by
    pub fn update(&mut self) {
        let elapsed = self.last_wall.elapsed().as_secs_f32();

        if elapsed < Self::INTERVAL {
            return;
        }

        let cpu = os::cpu_seconds();

        self.cores_busy = ((cpu - self.last_cpu) / elapsed as f64) as f32;
        self.resident = os::resident_bytes();
        self.last_cpu = cpu;
        self.last_wall = Instant::now();
    }
}

#[cfg(target_os = "macos")]
mod os {
    use std::ffi::{c_char, c_void};

    /// struct timeval, suseconds_t is 32 bits on darwin
    #[repr(C)]
    struct TimeVal {
        sec: i64,
        usec: i32,
    }

    /// struct rusage, we only ever read the two times
    #[repr(C)]
    struct RUsage {
        user: TimeVal,
        system: TimeVal,
        rest: [i64; 14],
    }

    /// mach_task_basic_info, 12 natural_t words
    #[repr(C)]
    struct TaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }

    const RUSAGE_SELF: i32 = 0;
    const MACH_TASK_BASIC_INFO: u32 = 20;

    unsafe extern "C" {
        static mach_task_self_: u32;
        fn getrusage(who: i32, usage: *mut RUsage) -> i32;
        fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *const c_void,
            new_len: usize,
        ) -> i32;
    }

    /// user + system cpu seconds of the whole process
    pub fn cpu_seconds() -> f64 {
        // SAFETY: getrusage only writes the struct it is given
        let usage = unsafe {
            let mut usage = std::mem::zeroed::<RUsage>();
            if getrusage(RUSAGE_SELF, &mut usage) != 0 {
                return 0.0;
            }
            usage
        };

        let seconds = |t: &TimeVal| t.sec as f64 + t.usec as f64 * 1e-6;
        seconds(&usage.user) + seconds(&usage.system)
    }

    pub fn resident_bytes() -> u64 {
        let mut count = (size_of::<TaskBasicInfo>() / 4) as u32;

        // SAFETY: task_info writes at most count words into the struct
        unsafe {
            let mut info = std::mem::zeroed::<TaskBasicInfo>();
            let info_ptr = &mut info as *mut TaskBasicInfo as *mut i32;

            if task_info(mach_task_self_, MACH_TASK_BASIC_INFO, info_ptr, &mut count) != 0 {
                return 0;
            }

            info.resident_size
        }
    }

    pub fn total_bytes() -> u64 {
        let mut bytes = 0u64;
        let mut len = size_of::<u64>();

        // SAFETY: hw.memsize is a u64, len tells sysctl the buffer size
        let ok = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                &mut bytes as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null(),
                0,
            )
        };

        if ok == 0 { bytes } else { 0 }
    }
}

#[cfg(not(target_os = "macos"))]
mod os {
    pub fn cpu_seconds() -> f64 {
        0.0
    }

    pub fn resident_bytes() -> u64 {
        0
    }

    pub fn total_bytes() -> u64 {
        0
    }
}
