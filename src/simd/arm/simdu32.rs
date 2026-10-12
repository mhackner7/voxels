use std::{
    arch::aarch64::*,
    ops::{Add, BitAnd, BitOr},
};

use crate::simd::{simdf32::SimdF32, simdi32::SimdI32};

#[repr(C)]
#[repr(align(16))] // just in case manually align to 16 bytes (or 128 bits)
#[derive(Debug, Clone, Copy)]
pub struct SimdU32 {
    pub data: uint32x4_t,
}

impl SimdU32 {
    pub const ALL_ACTIVE: Self = Self::comptime([u32::MAX; 4]);
    pub const ALL_INACTIVE: Self = Self::comptime([0; 4]);
    pub const ZEROES: Self = Self::ALL_INACTIVE;
    pub const ONES: Self = Self::comptime([1; 4]);

    pub const fn comptime(arr: [u32; 4]) -> Self {
        unsafe {
            Self {
                data: std::mem::transmute::<[u32; 4], uint32x4_t>(arr),
            }
        }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        unsafe {
            let mut a = [0u32; 4];
            vst1q_u32(a.as_mut_ptr(), self.data);
            a
        }
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe { vaddq_u32(self.data, other.data).into() }
    }

    /// logical right shift
    #[inline(always)]
    pub fn shrl<const SHAMT: i32>(self) -> Self {
        unsafe { vshrq_n_u32::<SHAMT>(self.data).into() }
    }
    /// logical left shift
    #[inline(always)]
    pub fn shll<const SHAMT: i32>(self) -> Self {
        unsafe { vshlq_n_u32::<SHAMT>(self.data).into() }
    }
    /// logical left shift based on variable
    #[inline(always)]
    pub fn shllv(self, shift: Self) -> Self {
        unsafe { vshlq_u32(self.data, vreinterpretq_s32_u32(shift.data)).into() }
    }
    /// logical and
    #[inline(always)]
    pub fn and(self, other: Self) -> Self {
        unsafe { vandq_u32(self.data, other.data).into() }
    }
    /// logical or
    #[inline(always)]
    pub fn or(self, other: Self) -> Self {
        unsafe { vorrq_u32(self.data, other.data).into() }
    }
    /// logical xor
    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        unsafe { veorq_u32(self.data, other.data).into() }
    }
    /// wrapping multiply, keeps the low 32 bits
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        unsafe { vmulq_u32(self.data, other.data).into() }
    }

    /// top 24 bits as a float in [0, 1), used to turn hashes into random numbers
    #[inline(always)]
    pub fn to_f32_unit(self) -> SimdF32 {
        const INV_2_24: f32 = 1.0 / 16_777_216.0;
        unsafe { vmulq_n_f32(vcvtq_f32_u32(vshrq_n_u32::<8>(self.data)), INV_2_24).into() }
    }

    /// greater than or equal
    #[inline(always)]
    pub fn great_or_eq(self, other: Self) -> Self {
        unsafe { vcgeq_u32(self.data, other.data).into() }
    }

    /// not equal
    #[inline(always)]
    pub fn not_eq(self, other: Self) -> Self {
        unsafe { vmvnq_u32(vceqq_u32(self.data, other.data)).into() }
    }

    #[inline(always)]
    pub fn all_equal(self) -> bool {
        unsafe {
            let all_set = vdupq_n_u32(vgetq_lane_u32::<0>(self.data));
            let mask = vceqq_u32(all_set, self.data);
            let min = vminvq_u32(mask);
            min != 0
        }
    }

    /// number of leading zero bits per lane
    #[inline(always)]
    pub fn leading_zeros(self) -> Self {
        unsafe { vclzq_u32(self.data).into() }
    }

    #[inline(always)]
    pub fn reinterpret_as_i32(self) -> SimdI32 {
        unsafe { vreinterpretq_s32_u32(self.data).into() }
    }

    /// not
    #[inline(always)]
    pub fn not(self) -> Self {
        unsafe { vmvnq_u32(self.data).into() }
    }

    /// not equal
    #[inline(always)]
    pub fn is_eq_to(self, other: Self) -> Self {
        unsafe { vceqq_u32(self.data, other.data).into() }
    }

    /// which ones are not zero
    #[inline(always)]
    pub fn not_zero(self) -> Self {
        unsafe { vcgtq_u32(self.data, Self::ZEROES.data).into() }
    }

    /// select -> self ? true : false
    #[inline(always)]
    pub fn select(self, true_: Self, false_: Self) -> Self {
        unsafe { vbslq_u32(self.data, true_.data, false_.data).into() }
    }

    /// select -> cond ? true : false, same argument order as SimdF32::select
    #[inline(always)]
    pub fn select_bits(condition_mask: Self, if_true: Self, if_false: Self) -> Self {
        condition_mask.select(if_true, if_false)
    }

    #[inline(always)]
    pub fn has_none_active(self) -> bool {
        unsafe { vmaxvq_u32(self.data) == 0 }
    }

    #[inline(always)]
    pub fn has_active(self) -> bool {
        unsafe { vmaxvq_u32(self.data) != 0 }
    }

    #[inline(always)]
    pub fn has_all_active(self) -> bool {
        unsafe { vminvq_u32(self.data) == u32::MAX }
    }
}

impl Add for SimdU32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl From<uint32x4_t> for SimdU32 {
    #[inline(always)]
    fn from(value: uint32x4_t) -> Self {
        SimdU32 { data: value }
    }
}

impl From<[u32; 4]> for SimdU32 {
    #[inline(always)]
    fn from(value: [u32; 4]) -> Self {
        unsafe {
            Self {
                data: vld1q_u32(value.as_ptr()),
            }
        }
    }
}

impl BitAnd for SimdU32 {
    type Output = SimdU32;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}
impl BitOr for SimdU32 {
    type Output = SimdU32;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl From<SimdU32> for [u32; 4] {
    #[inline(always)]
    fn from(value: SimdU32) -> Self {
        value.to_array()
    }
}

impl From<u32> for SimdU32 {
    fn from(value: u32) -> Self {
        unsafe {
            Self {
                data: vdupq_n_u32(value),
            }
        }
    }
}

impl From<SimdF32> for SimdU32 {
    fn from(value: SimdF32) -> Self {
        unsafe {
            Self {
                data: vcvtq_u32_f32(value.data),
            }
        }
    }
}
impl std::ops::Not for SimdU32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe {
            Self {
                data: vmvnq_u32(self.data),
            }
        }
    }
}
