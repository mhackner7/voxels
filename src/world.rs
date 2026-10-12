use crate::world::sun::Sun;
use crate::world::tree64::Tree64;

pub mod build;
pub mod collision;
pub mod leaf;
pub mod sun;
pub mod terrain;
pub mod tree64;
pub mod voxel;

pub struct World {
    pub tree: Tree64,
    /// the one sun, everything that gets lit reads it from here
    pub sun: Sun,
}

impl World {
    pub fn new() -> Self {
        println!("[World] Initializing voxel world with procedural seed 42...");
        World {
            tree: Tree64::build(42),
            sun: Sun::default(),
        }
    }

    /// put the sun somewhere else (degrees), angles get wrapped / clamped by Sun::new
    pub fn set_sun(&mut self, azimuth: f32, elevation: f32) {
        self.sun = Sun::new(azimuth, elevation);
    }

    pub fn update(&mut self, _dt: f32) {}
}
