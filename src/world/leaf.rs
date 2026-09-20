use crate::math::vec3::*;
use crate::simd::packetray::Packet;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdi32::SimdI32;
use crate::simd::simdu32::SimdU32;
use crate::world::voxel::*;

pub struct Leaf {
    pub grid: [Material; GRID_SIZE],
}

const GRID_SIZE: usize = Leaf::VOXELS_PER_AXIS.pow(3);

impl Leaf {
    pub const VOXELS_PER_AXIS: usize = 8;
    pub const VOXELS_PER_AXIS_F32: f32 = Leaf::VOXELS_PER_AXIS as f32;

    /// not coords, just use for volume measurements
    pub const BMIN: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const BMAX: Vec3 = Vec3::new_filled(Leaf::VOXELS_PER_AXIS_F32 * VoxelInfo::VOXEL_SIZE);

    pub const BMIN_SIMD: Packed_Vec3_x4 = Packed_Vec3_x4::from_single_const(Leaf::BMIN);
    pub const BMAX_SIMD: Packed_Vec3_x4 = Packed_Vec3_x4::from_single_const(Leaf::BMAX);

    pub fn new() -> Self {
        let min = Vec3::default();
        let max = v3(
            VoxelInfo::VOXEL_SIZE * Leaf::VOXELS_PER_AXIS_F32,
            VoxelInfo::VOXEL_SIZE * Leaf::VOXELS_PER_AXIS_F32,
            VoxelInfo::VOXEL_SIZE * Leaf::VOXELS_PER_AXIS_F32,
        );

        //generate a default volume
        //one in three voxels will be air
        //the non-air blocks will be generated

        let mut grid = [Material::Air; 8 * 8 * 8];

        for voxel in grid.iter_mut() {
            if fastrand::u16(0..4) == 0 {
                *voxel = fastrand::u8(1..Material::__COUNT as u8).into();
            }
        }

        Self { grid }
    }

    /// trace the packet into the volume, return the 4 color values
    ///
    pub fn trace(
        &self,
        packet: Packet,
        hit_distances: SimdF32,
        active_rays: SimdU32,
    ) -> ([Material; 4], [f32; 4], [u32; 4]) {
        let origins = packet.origins;
        let directions = packet.directions;
        let inverse_dirs = packet.inverse_dirs;

        let step_xs = directions.xs.signs();
        let step_ys = directions.ys.signs();
        let step_zs = directions.zs.signs();

        let ida = inverse_dirs.abs();

        let voxel_size = SimdF32::from_float(VoxelInfo::VOXEL_SIZE);
        let voxels_per_unit = SimdF32::from_float(VoxelInfo::VOXELS_PER_UNIT);

        let deltas_xs = ida.xs * voxel_size;
        let deltas_ys = ida.ys * voxel_size;
        let deltas_zs = ida.zs * voxel_size;

        let entry_xs = (directions
            .xs
            .mul_then_add(hit_distances.add(0.0001.into()), origins.xs)
            - Self::BMIN_SIMD.xs)
            * voxels_per_unit;

        let entry_ys = (directions
            .ys
            .mul_then_add(hit_distances.add(0.0001.into()), origins.ys)
            - Self::BMIN_SIMD.ys)
            * voxels_per_unit;

        let entry_zs = (directions
            .zs
            .mul_then_add(hit_distances.add(0.0001.into()), origins.zs)
            - Self::BMIN_SIMD.zs)
            * voxels_per_unit;

        let upper_bound = SimdF32::from_float(7.9999);

        // FINALLY THE FIX TO THE WEIRD GODLY PILLARS FUCK MATH
        let current_xs = entry_xs.max(SimdF32::ZEROES).min(upper_bound).floor();
        let current_ys = entry_ys.max(SimdF32::ZEROES).min(upper_bound).floor();
        let current_zs = entry_zs.max(SimdF32::ZEROES).min(upper_bound).floor();

        let mut tmax_xs = hit_distances
            + (current_xs.floor() - entry_xs + step_xs.max(SimdF32::ZEROES))
                * inverse_dirs.xs
                * voxel_size;

        let mut tmax_ys = hit_distances
            + (current_ys.floor() - entry_ys + step_ys.max(SimdF32::ZEROES))
                * inverse_dirs.ys
                * voxel_size;

        let mut tmax_zs = hit_distances
            + (current_zs.floor() - entry_zs + step_zs.max(SimdF32::ZEROES))
                * inverse_dirs.zs
                * voxel_size;

        tmax_xs = SimdF32::select(tmax_xs.is_ne_to(tmax_xs), SimdF32::INFY, tmax_xs);

        tmax_ys = SimdF32::select(tmax_ys.is_ne_to(tmax_ys), SimdF32::INFY, tmax_ys);

        tmax_zs = SimdF32::select(tmax_zs.is_ne_to(tmax_zs), SimdF32::INFY, tmax_zs);

        let step_xs = step_xs.to_i32();
        let step_ys = step_ys.to_i32();
        let step_zs = step_zs.to_i32();

        let mut current_xs = current_xs.to_i32();
        let mut current_ys = current_ys.to_i32();
        let mut current_zs = current_zs.to_i32();

        let min_curr = SimdI32::ZEROES;
        let max_curr = SimdI32::from(7);

        let mut result_voxels = [Material::Air; 4];
        let mut result_distances = hit_distances.to_array();
        let mut result_hits = [0u32; 4];

        let mut result_mins = hit_distances.to_array();

        let mut active_rays = active_rays; // the rays that are still active
        let mut valid_rays = active_rays; // the rays that are still valid for the given volume 

        for _ in 0..(8 + 8 + 8) {
            // fetch voxel of current position for active rays //
            let xcurr = current_xs.to_array();
            let ycurr = current_ys.to_array();
            let zcurr = current_zs.to_array();
            let mut active = active_rays.to_array();
            let mut valid = valid_rays.to_array();

            // split up packets for individual ray processing:
            // validity checks, hit checks, etc
            for i in 0..4 {
                if active[i] == 0 || valid[i] == 0 {
                    active[i] = 0;
                    valid[i] = 0;
                    continue;
                }

                let x = xcurr[i];
                let y = ycurr[i];
                let z = zcurr[i];

                let index = x + (y * 8) + (z * 8 * 8);

                let voxel = self.grid[index as usize];

                if voxel != Material::Air {
                    active[i] = 0;
                    valid[i] = 0;
                    result_voxels[i] = voxel;
                    result_distances[i] = result_mins[i];
                    result_hits[i] = 1;
                }
            }

            active_rays = active.into();
            valid_rays = valid.into();

            if active_rays.has_none_active() {
                break;
            }

            // get the mask to move next //
            let mins = tmax_xs.min(tmax_ys.min(tmax_zs));

            let mask_xs = mins.is_eq_to(tmax_xs);
            let mask_ys = mins.is_eq_to(tmax_ys);
            let mask_zs = mins.is_eq_to(tmax_zs);

            // update tmax and current position //

            tmax_xs = SimdF32::select(mask_xs, tmax_xs + deltas_xs, tmax_xs);
            tmax_ys = SimdF32::select(mask_ys, tmax_ys + deltas_ys, tmax_ys);
            tmax_zs = SimdF32::select(mask_zs, tmax_zs + deltas_zs, tmax_zs);

            current_xs = current_xs + SimdI32::select(mask_xs, step_xs, SimdI32::ZEROES);
            current_ys = current_ys + SimdI32::select(mask_ys, step_ys, SimdI32::ZEROES);
            current_zs = current_zs + SimdI32::select(mask_zs, step_zs, SimdI32::ZEROES);

            let x_invalid = current_xs.outside_range(min_curr, max_curr);
            let y_invalid = current_ys.outside_range(min_curr, max_curr);
            let z_invalid = current_zs.outside_range(min_curr, max_curr);

            valid_rays = valid_rays & (x_invalid | y_invalid | z_invalid).not();

            result_mins = mins.to_array();
        }

        (result_voxels, result_distances, result_hits)
    }
}
