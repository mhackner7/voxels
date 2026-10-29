use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub data: [f32; 3],
}

impl Vec3 {
    pub const DEFAULT: Vec3 = Self::new(0.0, 0.0, 0.0);
    pub const Y_POS_NORMAL: Vec3 = Self::new(0.0, 1.0, 0.0);

    #[inline(always)]
    pub fn normalize(self) -> Self {
        self / self.length()
    }

    #[inline(always)]
    pub fn at(self, axis: usize) -> f32 {
        self.data[axis]
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
