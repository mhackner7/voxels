use std::arch::aarch64::*;

use crate::simd::arm::{simdi32::SimdI32, simdu32::SimdU32};

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[repr(C)]
#[repr(align(16))] // just in case manually align to 16 bytes (or 128 bits)
#[derive(Debug, Clone, Copy)]
pub struct SimdF32 {
    pub data: float32x4_t,
}

impl SimdF32 {
    /// four zeroes
    pub const ZEROES: Self = unsafe { std::mem::transmute([0.0f32; 4]) };

    /// four identical tiny f32 values for raytracing
    pub const TINY: Self = unsafe { std::mem::transmute([1e-8f32; 4]) };

    /// four 1.0s
    pub const ONES: Self = unsafe { std::mem::transmute([1.0f32; 4]) };

    /// four 2.0s
    pub const TWOS: Self = unsafe { std::mem::transmute([2.0f32; 4]) };

    /// four -1.0s
    pub const NEGS: Self = unsafe { std::mem::transmute([-1.0f32; 4]) };

    /// four F32::INFINITY
    pub const INFY: Self = unsafe { std::mem::transmute([f32::INFINITY; 4]) };

    #[inline(always)]
    pub const fn from_float(value: f32) -> Self {
        unsafe {
            SimdF32 {
                data: std::mem::transmute([value; 4]),
            }
        }
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        unsafe {
            let mut a = [0.0f32; 4];
            vst1q_f32(a.as_mut_ptr(), self.data);
            return a;
        }
    }

    #[inline(always)]
    /// a + (b * c) -> add + (mul1 * mul2)
    pub fn mul_add(add: Self, mul1: Self, mul2: Self) -> Self {
        unsafe { vfmaq_f32(add.data, mul1.data, mul2.data).into() }
    }

    #[inline(always)]
    /// a + (b * c) -> self + (mul1 * mul2)
    pub fn add_to_mul(self, mul1: Self, mul2: Self) -> Self {
        unsafe { vfmaq_f32(self.data, mul1.data, mul2.data).into() }
    }

    #[inline(always)]
    /// a + (b * c) -> add + (self * mul2)
    pub fn mul_then_add(self, mul2: Self, add: Self) -> Self {
        unsafe { vfmaq_f32(add.data, self.data, mul2.data).into() }
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        unsafe { vnegq_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe { vaddq_f32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        unsafe { vmulq_f32(self.data, other.data).into() }
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        unsafe { vsubq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        unsafe { vminq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        unsafe { vmaxq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    /// select -> cond ? true : false
    pub fn select(condition_mask: SimdU32, if_true: Self, if_false: Self) -> Self {
        unsafe { vbslq_f32(condition_mask.data, if_true.data, if_false.data).into() }
    }

    #[inline(always)]
    pub fn is_eq_to(self, other: Self) -> SimdU32 {
        unsafe { vceqq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn is_less_than(self, other: Self) -> SimdU32 {
        unsafe { vcltq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn is_greater_than(self, other: Self) -> SimdU32 {
        unsafe { vcgtq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn is_less_or_equal(self, other: Self) -> SimdU32 {
        unsafe { vcleq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn is_greater_or_equal(self, other: Self) -> SimdU32 {
        unsafe { vcgeq_f32(self.data, other.data).into() }
    }

    #[inline(always)]
    pub fn reinterpret_as_i32(self) -> SimdI32 {
        unsafe { vreinterpretq_s32_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn reinterpret_as_u32(self) -> SimdU32 {
        unsafe { vreinterpretq_u32_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn to_i32(self) -> SimdI32 {
        unsafe { vcvtq_s32_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        unsafe { vrndmq_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe { vabsq_f32(self.data).into() }
    }

    #[inline(always)]
    pub fn clamp_to(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    #[inline(always)]
    pub fn squared(self) -> Self {
        self * self
    }

    #[inline(always)]
    pub fn recip(self) -> Self {
        unsafe {
            let estimate = vrecpeq_f32(self.data);

            let step1 = vrecpsq_f32(self.data, estimate);
            let estimate = vmulq_f32(estimate, step1);

            let step2 = vrecpsq_f32(self.data, estimate);

            vmulq_f32(estimate, step2).into()
        }
    }

    #[inline(always)]
    /// get the sign values as 1.0 for positive and -1.0 for negative
    pub fn signs(self) -> Self {
        static NEGS: SimdF32 = SimdF32::from_float(-1.0f32);

        static ONES: SimdF32 = SimdF32::from_float(1.0f32);
        let mask = self.is_greater_or_equal(Self::ZEROES);
        Self::select(mask, ONES, NEGS)
    }

    #[inline(always)]
    pub fn inv_sqrt(self) -> Self {
        unsafe {
            let estimate = vrsqrteq_f32(self.data);

            let step1 = vrsqrtsq_f32(vmulq_f32(estimate, estimate), self.data);
            let estimate = vmulq_f32(estimate, step1);

            let step2 = vrsqrtsq_f32(vmulq_f32(estimate, estimate), self.data);

            vmulq_f32(estimate, step2).into()
        }
    }
}

impl From<[f32; 4]> for SimdF32 {
    #[inline(always)]
    fn from(value: [f32; 4]) -> Self {
        unsafe {
            SimdF32 {
                data: vld1q_f32(value.as_ptr()),
            }
        }
    }
}

impl From<SimdF32> for [f32; 4] {
    #[inline(always)]
    fn from(value: SimdF32) -> Self {
        value.to_array()
    }
}

impl From<f32> for SimdF32 {
    #[inline(always)]
    fn from(value: f32) -> Self {
        unsafe {
            SimdF32 {
                data: vdupq_n_f32(value),
            }
        }
    }
}

impl From<float32x4_t> for SimdF32 {
    #[inline(always)]
    fn from(value: float32x4_t) -> Self {
        SimdF32 { data: value }
    }
}

impl Add<Self> for SimdF32 {
    type Output = SimdF32;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { vaddq_f32(self.data, rhs.data).into() }
    }
}

impl Sub<SimdF32> for SimdF32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: SimdF32) -> Self::Output {
        unsafe { vsubq_f32(self.data, rhs.data).into() }
    }
}

impl Mul<SimdF32> for SimdF32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: SimdF32) -> Self::Output {
        unsafe { vmulq_f32(self.data, rhs.data).into() }
    }
}

impl Neg for SimdF32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.neg()
    }
}
