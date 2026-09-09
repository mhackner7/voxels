use crate::math::vec3::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    #[inline(always)]
    pub const fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    pub const DEFAULT: Ray = Ray {
        origin: Vec3::DEFAULT,
        direction: Vec3::DEFAULT,
    };

    #[inline(always)]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + (self.direction * t)
    }

    #[inline(always)]
    pub fn split(self) -> (Vec3, Vec3) {
        (self.origin, self.direction)
    }

    #[inline(always)]
    /// ray built from one origin vector to a point
    pub fn from(origin: Vec3, to: Vec3) -> Self {
        Self {
            origin,
            direction: (to - origin).normalize(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RayI {
    pub origin: Vec3,
    pub direction: Vec3,
    pub inv_dir: Vec3,
}

impl RayI {
    #[inline(always)]
    pub fn new(origin: Vec3, direction: Vec3) -> RayI {
        RayI {
            origin,
            direction,
            inv_dir: 1.0 / direction,
        }
    }

    #[inline(always)]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + (self.direction * t)
    }

    #[inline(always)]
    pub fn split(self) -> (Vec3, Vec3, Vec3) {
        (self.origin, self.direction, self.inv_dir)
    }
}

impl From<Ray> for RayI {
    fn from(value: Ray) -> Self {
        RayI {
            origin: value.origin,
            direction: value.direction,
            inv_dir: 1.0 / value.direction,
        }
    }
}
