use crate::math::ivec3::IVec3;
use std::f32::{INFINITY, NAN};
use std::mem::{size_of, transmute};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub data: [f32; 3],
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
        self / self.length()
    }

    #[inline(always)]
    pub fn at(self, axis: usize) -> f32 {
        self.data[axis]
    }

    #[inline(always)]
    pub fn fix_nan(mut self) -> Self {
        for value in &mut self.data {
            if value.is_nan() {
                *value = INFINITY;
            }
        }

        self
    }

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { data: [x, y, z] }
    }

    #[inline(always)]
    pub const fn new_filled(f: f32) -> Self {
        Self { data: [f; 3] }
    }

    #[inline(always)]
    pub fn clamp_to(self, min: f32, max: f32) -> Self {
        Self {
            data: [
                self.data[0].clamp(min, max),
                self.data[1].clamp(min, max),
                self.data[2].clamp(min, max),
            ],
        }
    }

    #[inline(always)]
    pub fn signs(self) -> Self {
        Self {
            data: [
                self.data[0].signum(),
                self.data[1].signum(),
                self.data[2].signum(),
            ],
        }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self {
            data: [
                self.data[0].floor(),
                self.data[1].floor(),
                self.data[2].floor(),
            ],
        }
    }

    #[inline(always)]
    pub fn max(self) -> f32 {
        self.data[0].max(self.data[1].max(self.data[2]))
    }

    #[inline(always)]
    pub fn min(self) -> f32 {
        self.data[0].min(self.data[1].min(self.data[2]))
    }

    #[inline(always)]
    pub fn max_axis(self) -> usize {
        if self.data[0] > self.data[1] {
            if self.data[0] > self.data[2] { 0 } else { 2 }
        } else if self.data[1] > self.data[2] {
            1
        } else {
            2
        }
    }

    #[inline(always)]
    pub fn min_axis(self) -> usize {
        if self.data[0] < self.data[1] {
            if self.data[0] < self.data[2] { 0 } else { 2 }
        } else if self.data[1] < self.data[2] {
            1
        } else {
            2
        }
    }

    #[inline(always)]
    pub fn split(self) -> (f32, f32, f32) {
        (self.data[0], self.data[1], self.data[2])
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.data[0] * self.data[0] + self.data[1] * self.data[1] + self.data[2] * self.data[2]
    }

    #[inline(always)]
    pub fn unit(self) -> Self {
        self / self.length()
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.data[0] * rhs.data[0] + self.data[1] * rhs.data[1] + self.data[2] * rhs.data[2]
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            data: [
                self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
                self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
                self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
            ],
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self {
            data: [self.data[0].abs(), self.data[1].abs(), self.data[2].abs()],
        }
    }

    #[inline]
    pub fn tmax_from(step: IVec3, entry_pos: Vec3, inverse_dir: Vec3) -> Vec3 {
        let sx = (step[0] > 0) as i32 as f32;
        let sy = (step[1] > 0) as i32 as f32;
        let sz = (step[2] > 0) as i32 as f32;

        Self {
            data: [
                (sx - entry_pos.data[0].fract()) * inverse_dir.data[0],
                (sy - entry_pos.data[1].fract()) * inverse_dir.data[1],
                (sz - entry_pos.data[2].fract()) * inverse_dir.data[2],
            ],
        }
    }

    #[inline(always)]
    pub const fn serialize(self) -> [u8; Self::SIZEOF] {
        unsafe { transmute(self) }
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Neg for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            data: [-self.data[0], -self.data[1], -self.data[2]],
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                self.data[0] + rhs.data[0],
                self.data[1] + rhs.data[1],
                self.data[2] + rhs.data[2],
            ],
        }
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: f32) -> Self::Output {
        Self {
            data: [self.data[0] + rhs, self.data[1] + rhs, self.data[2] + rhs],
        }
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
        Self {
            data: [
                self.data[0] - rhs.data[0],
                self.data[1] - rhs.data[1],
                self.data[2] - rhs.data[2],
            ],
        }
    }
}

impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            data: [self.data[0] * rhs, self.data[1] * rhs, self.data[2] * rhs],
        }
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Self::Output {
        Self {
            data: [
                self.data[0] * rhs.data[0],
                self.data[1] * rhs.data[1],
                self.data[2] * rhs.data[2],
            ],
        }
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
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
        Self {
            data: [self.data[0] / rhs, self.data[1] / rhs, self.data[2] / rhs],
        }
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, rhs: Vec3) -> Vec3 {
        rhs / self
    }
}

impl From<IVec3> for Vec3 {
    #[inline(always)]
    fn from(value: IVec3) -> Self {
        Self::new(value[0] as f32, value[1] as f32, value[2] as f32)
    }
}
