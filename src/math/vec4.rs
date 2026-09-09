use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub const fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    #[inline(always)]
    pub const fn vec3(self) {}

    #[inline(always)]
    pub fn dot(self, other: Vec4) -> f32 {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z) + (self.w * other.w)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub const fn split(self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    pub unsafe fn at(&self, index: usize) -> f32 {
        debug_assert!(index <= 3, "Invalid Index");
        unsafe { *(self as *const Vec4 as *const f32).add(index) }
    }

    #[inline(always)]
    pub unsafe fn at_mut(&mut self, index: usize) -> &f32 {
        debug_assert!(index <= 3, "Invalid Index");
        unsafe { &*(self as *mut Vec4 as *mut f32).add(index) }
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Add<Vec4> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Vec4) -> Self::Output {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub<Vec4> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Vec4) -> Self::Output {
        self + -rhs
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, rhs: Vec4) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Div<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn div(self, rhs: Vec4) -> Self::Output {
        rhs / self
    }
}

impl AddAssign<Vec4> for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vec4) {
        *self = *self + rhs
    }
}

impl SubAssign<Vec4> for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vec4) {
        *self = *self - rhs
    }
}
