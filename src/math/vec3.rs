use crate::math::ivec3::IVec3;
use fastrand::Rng;
use std::mem::{size_of, transmute};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub __padd: f32,
}

pub type Color = Vec3;
pub type Point = Vec3;

pub fn dot(lhs: Vec3, rhs: Vec3) -> f32 {
    lhs.dot(rhs)
}

pub fn cross(lhs: Vec3, rhs: Vec3) -> Vec3 {
    lhs.cross(rhs)
}
pub const fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}
pub const fn color(r: f32, g: f32, b: f32) -> Color {
    Color::new(r, g, b)
}

impl Vec3 {
    const SIZEOF: usize = size_of::<Self>();

    const AXIS_MASKS: [Self; 3] = [
        Self::new(1.0, 0.0, 0.0),
        Self::new(0.0, 1.0, 0.0),
        Self::new(0.0, 0.0, 1.0),
    ];

    #[inline(always)]
    pub const fn axis_mask(axis: usize) -> Self {
        Self::AXIS_MASKS[axis]
    }

    pub const DEFAULT: Vec3 = Self::new(0.0, 0.0, 0.0);
    pub const X_POS_NORMAL: Vec3 = Self::new(1.0, 0.0, 0.0);
    pub const X_NEG_NORMAL: Vec3 = Self::new(-1.0, 0.0, 0.0);
    pub const Y_POS_NORMAL: Vec3 = Self::new(0.0, 1.0, 0.0);
    pub const Y_NEG_NORMAL: Vec3 = Self::new(0.0, -1.0, 0.0);
    pub const Z_POS_NORMAL: Vec3 = Self::new(0.0, 0.0, 1.0);
    pub const Z_NEG_NORMAL: Vec3 = Self::new(0.0, 0.0, -1.0);

    #[inline(always)]
    pub fn get_face_normal(axis: usize, step: i32) -> Vec3 {
        const NORMALS: [[Vec3; 2]; 3] = [
            [Vec3::X_NEG_NORMAL, Vec3::X_POS_NORMAL],
            [Vec3::Y_NEG_NORMAL, Vec3::Y_POS_NORMAL],
            [Vec3::Z_NEG_NORMAL, Vec3::Z_POS_NORMAL],
        ];

        let sign = (step + 1) as usize >> 1;
        NORMALS[axis][sign]
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        self / len
    }

    #[inline(always)]
    pub fn at(self, axis: usize) -> f32 {
        debug_assert!(axis < 3, "invalid indexing");
        unsafe { *(&self.x as *const f32).add(axis) }
    }

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            __padd: 1.0,
        }
    }

    #[inline(always)]
    pub const fn new_filled(f: f32) -> Self {
        Self {
            x: f,
            y: f,
            z: f,
            __padd: 1.0,
        }
    }

    #[inline(always)]
    pub fn clamp_to(self, min: f32, max: f32) -> Self {
        Self::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
        )
    }

    #[inline(always)]
    pub fn signs(self) -> Self {
        Self::new(self.x.signum(), self.y.signum(), self.z.signum())
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    #[inline(always)]
    pub fn max(self) -> f32 {
        self.x.max(self.y.max(self.z))
    }

    #[inline(always)]
    pub fn min(self) -> f32 {
        self.x.min(self.y.min(self.z))
    }

    #[inline(always)]
    pub fn max_axis(self) -> usize {
        if self.x > self.y {
            if self.x > self.z { 0 } else { 2 }
        } else {
            if self.y > self.z { 1 } else { 2 }
        }
    }
    #[inline(always)]
    pub fn min_axis(self) -> usize {
        if self.x < self.y {
            if self.x < self.z { 0 } else { 2 }
        } else {
            if self.y < self.z { 1 } else { 2 }
        }
    }

    #[inline(always)]
    pub fn split(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn unit(self) -> Self {
        self / self.length()
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    pub fn tmax_from(step: IVec3, entry_pos: Vec3, inverse_dir: Vec3) -> Vec3 {
        let sx = (step.x > 0) as i32 as f32;
        let sy = (step.y > 0) as i32 as f32;
        let sz = (step.z > 0) as i32 as f32;

        Vec3::new(
            (sx - entry_pos.x.fract()) * inverse_dir.x,
            (sy - entry_pos.y.fract()) * inverse_dir.y,
            (sz - entry_pos.z.fract()) * inverse_dir.z,
        )
    }

    #[inline(always)]
    pub const fn serialize(self) -> [u8; Self::SIZEOF] {
        unsafe { transmute(self) }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl Mul<Vec3> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn div(self, rhs: Vec3) -> Self::Output {
        rhs / self
    }
}
impl From<IVec3> for Vec3 {
    fn from(value: IVec3) -> Self {
        Self::new(value.x as f32, value.y as f32, value.z as f32)
    }
}
impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.__padd,
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.__padd,
        }
    }
}
