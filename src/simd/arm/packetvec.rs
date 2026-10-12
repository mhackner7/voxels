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

    pub const NEGS: Self = Self {
        xs: SimdF32::NEGS,
        ys: SimdF32::NEGS,
        zs: SimdF32::NEGS,
    };

    #[inline(always)]
    /// makes a new Packed_Vec3_x4 from one SimdF32 in all slots
    pub const fn stretched(vec: SimdF32) -> Self {
        Self {
            xs: vec,
            ys: vec,
            zs: vec,
        }
    }

    #[inline(always)]
    pub const fn float(val: f32) -> Self {
        Self {
            xs: SimdF32::from_float(val),
            ys: SimdF32::from_float(val),
            zs: SimdF32::from_float(val),
        }
    }

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

    #[inline(always)]
    /// get 1.0 for positive and -1.0 for negative
    pub fn signs(self) -> Self {
        Self {
            xs: self.xs.signs(),
            ys: self.ys.signs(),
            zs: self.zs.signs(),
        }
    }

    #[inline(always)]
    /// get absolute values for each
    pub fn abs(self) -> Self {
        Self {
            xs: self.xs.abs(),
            ys: self.ys.abs(),
            zs: self.zs.abs(),
        }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self {
            xs: self.xs.floor(),
            ys: self.ys.floor(),
            zs: self.zs.floor(),
        }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            xs: self.xs.min(other.xs),
            ys: self.ys.min(other.ys),
            zs: self.zs.min(other.zs),
        }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            xs: self.xs.max(other.xs),
            ys: self.ys.max(other.ys),
            zs: self.zs.max(other.zs),
        }
    }

    #[inline(always)]
    pub fn clamp_to(self, min: Self, max: Self) -> Self {
        Self {
            xs: self.xs.clamp_to(min.xs, max.xs),
            ys: self.ys.clamp_to(min.ys, max.ys),
            zs: self.zs.clamp_to(min.zs, max.zs),
        }
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
    pub fn neg(self) -> Self {
        Self {
            xs: self.xs.neg(),
            ys: self.ys.neg(),
            zs: self.zs.neg(),
        }
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

    #[inline(always)]
    /// (mul1 * mul2) + add
    fn mul_add(mul1: Self, mul2: Self, add: Self) -> Self {
        Self {
            xs: SimdF32::mul_add(add.xs, mul1.xs, mul2.xs),
            ys: SimdF32::mul_add(add.ys, mul1.ys, mul2.ys),
            zs: SimdF32::mul_add(add.zs, mul1.zs, mul2.zs),
        }
    }

    #[inline(always)]
    /// self + (mul1 * mul2)
    pub fn add_to_product(self, mul1: Self, mul2: Self) -> Self {
        Self::mul_add(mul1, mul2, self)
    }

    #[inline(always)]
    /// (self * mul2) + add
    pub fn mul_then_add(self, mul: Self, add: Self) -> Self {
        Self::mul_add(self, mul, add)
    }
}

impl Add<Self> for Packed_Vec3_x4 {
    type Output = Self;
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
