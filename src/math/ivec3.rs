use crate::math::vec3::Vec3;
use std::mem::{size_of, transmute};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    _padd: i32,
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
        unsafe { *(&self.x as *const i32).add(axis) }
    }

    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z, _padd: 0 }
    }

    #[inline(always)]
    pub fn split(self) -> (i32, i32, i32) {
        (self.x, self.y, self.z)
    }

    #[inline(always)]
    pub fn clamp_to(self, min: i32, max: i32) -> Self {
        Self::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
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
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for IVec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<i32> for IVec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
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
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for IVec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl From<Vec3> for IVec3 {
    fn from(value: Vec3) -> Self {
        Self::new(
            value.x.floor() as i32,
            value.y.floor() as i32,
            value.z.floor() as i32,
        )
    }
}

impl Index<usize> for IVec3 {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self._padd,
        }
    }
}

impl IndexMut<usize> for IVec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self._padd,
        }
    }
}
impl Mul<IVec3> for IVec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: IVec3) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}
