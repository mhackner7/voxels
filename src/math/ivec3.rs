use crate::math::vec3::Vec3;
use std::mem::{size_of, transmute};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct IVec3 {
    pub data: [i32; 4],
}

impl IVec3 {
    const SIZEOF: usize = size_of::<Self>();

    const AXIS_MASKS: [Self; 3] = [Self::new(1, 0, 0), Self::new(0, 1, 0), Self::new(0, 0, 1)];

    #[inline(always)]
    pub const fn axis_mask(axis: usize) -> Self {
        Self::AXIS_MASKS[axis]
    }

    #[inline(always)]
    pub fn at(self, axis: usize) -> i32 {
        debug_assert!(axis < 3, "invalid indexing");
        self.data[axis]
    }

    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { data: [x, y, z, 0] }
    }

    #[inline(always)]
    pub fn split(self) -> (i32, i32, i32) {
        (self.data[0], self.data[1], self.data[2])
    }

    #[inline(always)]
    pub fn clamp_to(self, min: i32, max: i32) -> Self {
        Self::new(
            self.data[0].clamp(min, max),
            self.data[1].clamp(min, max),
            self.data[2].clamp(min, max),
        )
    }

    #[inline(always)]
    pub const fn serialize(self) -> [u8; Self::SIZEOF] {
        unsafe { transmute(self) }
    }
}

impl Neg for IVec3 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.data[0], -self.data[1], -self.data[2])
    }
}

impl Add for IVec3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data[0] + rhs.data[0],
            self.data[1] + rhs.data[1],
            self.data[2] + rhs.data[2],
        )
    }
}

impl Add<i32> for IVec3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: i32) -> Self::Output {
        Self::new(self.data[0] + rhs, self.data[1] + rhs, self.data[2] + rhs)
    }
}

impl AddAssign for IVec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for IVec3 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data[0] - rhs.data[0],
            self.data[1] - rhs.data[1],
            self.data[2] - rhs.data[2],
        )
    }
}

impl SubAssign for IVec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<IVec3> for IVec3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: IVec3) -> Self::Output {
        Self::new(
            self.data[0] * rhs.data[0],
            self.data[1] * rhs.data[1],
            self.data[2] * rhs.data[2],
        )
    }
}

impl From<Vec3> for IVec3 {
    #[inline(always)]
    fn from(value: Vec3) -> Self {
        Self::new(
            value[0].floor() as i32,
            value[1].floor() as i32,
            value[2].floor() as i32,
        )
    }
}

impl Index<usize> for IVec3 {
    type Output = i32;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for IVec3 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
