use crate::world::voxel::Material;
use crate::{math::hit::simd_aabb, simd::packetray::Packet, world::testgrid::TestGrid};

pub mod testgrid;
pub mod voxel;

pub struct World {
    pub grid: TestGrid,
    pub rebuild_time: f32,
}

impl World {
    pub fn new() -> Self {
        World {
            grid: TestGrid::new(),
            rebuild_time: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.rebuild_time += dt;
        if self.rebuild_time >= 5.0 {
            self.rebuild_time = 0.0;
            self.grid = TestGrid::new();
        }
    }

    pub fn trace(&self, packet: Packet) -> ([Material; 4], [f32; 4], [u32; 4]) {
        let (distances, hit_mask) = simd_aabb(
            packet.origins,
            packet.inverse_dirs,
            self.grid.bmin,
            self.grid.bmax,
        );
        self.grid.trace(packet, distances, hit_mask)
    }
}
