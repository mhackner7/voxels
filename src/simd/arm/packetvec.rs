use std::ops::{Add, Mul, Sub};

use crate::math::vec3::Vec3;
use crate::simd::arm::simdf32::SimdF32;
use crate::simd::arm::simdu32::SimdU32;

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub struct Packed_Vec3_x4 {
    pub xs: SimdF32,
    pub ys: SimdF32,
    pub zs: SimdF32,
}

impl Packed_Vec3_x4 {
    pub const ZEROES: Self = Self {
        xs: SimdF32::ZEROES,
        ys: SimdF32::ZEROES,
        zs: SimdF32::ZEROES,
    };

    pub const ONES: Self = Self {
        xs: SimdF32::ONES,
        ys: SimdF32::ONES,
        zs: SimdF32::ONES,
    };

    #[inline(always)]
    pub fn new(xs: [f32; 4], ys: [f32; 4], zs: [f32; 4]) -> Self {
        Self {
            xs: xs.into(),
            ys: ys.into(),
            zs: zs.into(),
        }
    }

    #[inline(always)]
    pub fn from_single(v: Vec3) -> Self {
        Self {
            xs: SimdF32::from(v.data[0]),
            ys: SimdF32::from(v.data[1]),
            zs: SimdF32::from(v.data[2]),
        }
    }
    #[inline(always)]
    pub const fn from_single_const(v: Vec3) -> Self {
        Self {
            xs: SimdF32::from_float(v.data[0]),
            ys: SimdF32::from_float(v.data[1]),
            zs: SimdF32::from_float(v.data[2]),
        }
    }

    /// [xs, ys, zs]
    #[inline(always)]
    pub fn axes(self) -> [SimdF32; 3] {
        [self.xs, self.ys, self.zs]
    }

    #[inline(always)]
    pub fn from_axes([xs, ys, zs]: [SimdF32; 3]) -> Self {
        Self { xs, ys, zs }
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> SimdF32 {
        self.xs
            .mul_then_add(other.xs, self.ys.mul_then_add(other.ys, self.zs * other.zs))
    }

    #[inline(always)]
    /// every lane's vector times that lane's scalar
    pub fn scale(self, s: SimdF32) -> Self {
        Self {
            xs: self.xs * s,
            ys: self.ys * s,
            zs: self.zs * s,
        }
    }

    #[inline(always)]
    /// self + other * s, per lane
    pub fn add_scaled(self, other: Self, s: SimdF32) -> Self {
        Self {
            xs: other.xs.mul_then_add(s, self.xs),
            ys: other.ys.mul_then_add(s, self.ys),
            zs: other.zs.mul_then_add(s, self.zs),
        }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        self.scale(self.dot(self).inv_sqrt())
    }

    #[inline(always)]
    /// select -> cond ? true : false, per lane
    pub fn select(condition_mask: SimdU32, if_true: Self, if_false: Self) -> Self {
        Self {
            xs: SimdF32::select(condition_mask, if_true.xs, if_false.xs),
            ys: SimdF32::select(condition_mask, if_true.ys, if_false.ys),
            zs: SimdF32::select(condition_mask, if_true.zs, if_false.zs),
        }
    }
}

impl Add<Self> for Packed_Vec3_x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            xs: self.xs + rhs.xs,
            ys: self.ys + rhs.ys,
            zs: self.zs + rhs.zs,
        }
    }
}

impl Sub<Self> for Packed_Vec3_x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            xs: self.xs - rhs.xs,
            ys: self.ys - rhs.ys,
            zs: self.zs - rhs.zs,
        }
    }
}

impl Mul<Self> for Packed_Vec3_x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            xs: self.xs * rhs.xs,
            ys: self.ys * rhs.ys,
            zs: self.zs * rhs.zs,
        }
    }
}

impl From<Vec3> for Packed_Vec3_x4 {
    fn from(value: Vec3) -> Self {
        Self::from_single(value)
    }
}

impl From<f32> for Packed_Vec3_x4 {
    fn from(value: f32) -> Self {
        Self {
            xs: SimdF32::from(value),
            ys: SimdF32::from(value),
            zs: SimdF32::from(value),
        }
    }
}
