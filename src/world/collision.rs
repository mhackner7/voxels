use crate::math::vec3::Vec3;
use crate::world::World;
use crate::world::tree64::Tree64;
use crate::world::voxel::*;

/*
    COLLISION QUERIES

    the world doesn't know who's asking, it just answers for boxes given
    as world space corners in meters. the player owns its hitbox and decides
    what to do about walls (sliding, stepping up, landing)

    boxes get swept one axis at a time through every voxel layer the leading
    face enters, so no speed can tunnel through a 0.125m voxel.
    water and air never block, everything else does
*/

impl World {
    /// gap left between a box and the face that stopped it (meters)
    pub const SKIN: f32 = 1e-3;

    /// how far the box lo..hi (meters) gets moving d along one axis,
    /// and whether a voxel stopped it short. a box already touching never gets pushed back
    pub fn sweep(&self, lo: Vec3, hi: Vec3, axis: usize, d: f32) -> (f32, bool) {
        if d == 0.0 {
            return (0.0, false);
        }

        let lo = lo * VOXELS_PER_METER;
        let hi = hi * VOXELS_PER_METER;
        let dv = d * VOXELS_PER_METER;

        // the voxel layers the leading face runs into //

        let (first, last, dir) = if d > 0.0 {
            (hi[axis].ceil() as i32, (hi[axis] + dv).ceil() as i32 - 1, 1)
        } else {
            (
                lo[axis].floor() as i32 - 1,
                (lo[axis] + dv).floor() as i32,
                -1,
            )
        };

        let mut layer = first;

        while (layer - last) * dir <= 0 {
            if self.layer_blocked(lo, hi, axis, layer) {
                // stop a skin short of the layer that's in the way //

                let travel = if dir > 0 {
                    (layer as f32 * VOXEL_SIZE - Self::SKIN) - hi[axis] * VOXEL_SIZE
                } else {
                    ((layer + 1) as f32 * VOXEL_SIZE + Self::SKIN) - lo[axis] * VOXEL_SIZE
                };

                // already touching (give or take float noise) means no move at all,
                // otherwise a box resting on the ground would creep every frame
                let travel = if travel * dir as f32 <= Self::SKIN * 0.5 {
                    0.0
                } else {
                    travel
                };

                return (travel, true);
            }

            layer += dir;
        }

        (d, false)
    }

    /// any solid voxel inside the box lo..hi (meters)
    pub fn box_blocked(&self, lo: Vec3, hi: Vec3) -> bool {
        let lo = lo * VOXELS_PER_METER;
        let hi = hi * VOXELS_PER_METER;

        (lo[1].floor() as i32..hi[1].ceil() as i32).any(|y| self.layer_blocked(lo, hi, 1, y))
    }

    /// the voxel around a point (meters) is water
    pub fn water_at(&self, p: Vec3) -> bool {
        let v = p * VOXELS_PER_METER;
        self.voxel([0, 1, 2].map(|a| v[a].floor() as i32)) == Some(Material::Water)
    }

    /// any solid voxel in one layer of the box, lo / hi are box corners in voxels
    fn layer_blocked(&self, lo: Vec3, hi: Vec3, axis: usize, layer: i32) -> bool {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);

        for u in lo[a].floor() as i32..hi[a].ceil() as i32 {
            for v in lo[b].floor() as i32..hi[b].ceil() as i32 {
                let mut voxel = [0; 3];
                voxel[axis] = layer;
                voxel[a] = u;
                voxel[b] = v;

                if self.voxel(voxel).is_some_and(|m| m != Material::Water) {
                    return true;
                }
            }
        }

        false
    }

    /// material at integer voxel coords, anything outside the volume is air
    #[inline(always)]
    fn voxel(&self, [x, y, z]: [i32; 3]) -> Option<Material> {
        // a negative coord or any bit above 2047 lands outside //
        if (x | y | z) & !(Tree64::VOXELS_PER_AXIS - 1) != 0 {
            return None;
        }

        self.tree.voxel_at(x as u32, y as u32, z as u32)
    }
}
