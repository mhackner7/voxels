#[cfg(target_arch = "x86_64")]
compile_error!("x86_64 SIMD backend not implemented yet, mirror src/simd/arm");

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "aarch64")]
pub use arm::*;
