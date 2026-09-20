use crate::simd::arm::simdf32::SimdF32;
use crate::simd::arm::simdu32::SimdU32;
use std::arch::aarch64::*;
use std::ops::{Add, BitAnd, BitOr, Sub};

#[repr(C)]
#[repr(align(16))] // just in case manually align to 16 bytes (or 128 bits)
#[derive(Debug, Clone, Copy)]
pub struct SimdI32 {
    pub data: int32x4_t,
}

impl SimdI32 {
    pub const ZEROES: Self = unsafe { std::mem::transmute([0; 4]) };
    pub const FOURS: Self = unsafe { std::mem::transmute([4; 4]) };

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        unsafe {
            let mut a = [0i32; 4];
            vst1q_s32(a.as_mut_ptr(), self.data);
            return a;
        }
    }

    #[inline(always)]
    /// select -> cond ? true : false
    pub fn select(condition_mask: SimdU32, if_true: Self, if_false: Self) -> Self {
        unsafe { vbslq_s32(condition_mask.data, if_true.data, if_false.data).into() }
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        unsafe { vmaxq_s32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn reinterpret_as_f32(self) -> SimdF32 {
        unsafe { vreinterpretq_f32_s32(self.data).into() }
    }
    #[inline(always)]
    pub fn reinterpret_as_u32(self) -> SimdU32 {
        unsafe { vreinterpretq_u32_s32(self.data).into() }
    }

    #[inline(always)]
    pub fn to_f32(self) -> SimdF32 {
        unsafe { vcvtq_f32_s32(self.data).into() }
    }

    /// logical left shift
    #[inline(always)]
    pub fn shll<const shamt: i32>(self) -> Self {
        unsafe { vshlq_n_s32::<shamt>(self.data).into() }
    }

    #[inline(always)]
    pub fn great_or_eq(self, other: Self) -> SimdU32 {
        unsafe { vcgeq_s32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn less_than(self, other: Self) -> SimdU32 {
        unsafe { vcltq_s32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        unsafe { vsubq_s32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn neg(self) -> Self {
        unsafe { vnegq_s32(self.data).into() }
    }

    /// logical and
    #[inline(always)]
    pub fn and(self, other: Self) -> Self {
        unsafe { vandq_s32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe { vaddq_s32(self.data, other.data).into() }
    }
    /// logical or
    #[inline(always)]
    pub fn or(self, other: Self) -> Self {
        unsafe { vorrq_s32(self.data, other.data).into() }
    }

    /// logical or
    #[inline(always)]
    pub fn outside_range(self, min: Self, max: Self) -> SimdU32 {
        unsafe {
            vorrq_u32(
                vcltq_s32(self.data, min.data),
                vcgtq_s32(self.data, max.data),
            )
            .into()
        }
    }
}

impl From<int32x4_t> for SimdI32 {
    #[inline(always)]
    fn from(value: int32x4_t) -> Self {
        Self { data: value }
    }
}

impl From<[i32; 4]> for SimdI32 {
    #[inline(always)]
    fn from(value: [i32; 4]) -> Self {
        unsafe {
            Self {
                data: vld1q_s32(value.as_ptr()),
            }
        }
    }
}

impl From<SimdI32> for [i32; 4] {
    #[inline(always)]
    fn from(value: SimdI32) -> Self {
        value.to_array()
    }
}

impl From<i32> for SimdI32 {
    fn from(value: i32) -> Self {
        unsafe {
            Self {
                data: vdupq_n_s32(value),
            }
        }
    }
}
impl BitAnd for SimdI32 {
    type Output = SimdI32;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}
impl BitOr for SimdI32 {
    type Output = SimdI32;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}
impl Add for SimdI32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}
impl Sub for SimdI32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}
