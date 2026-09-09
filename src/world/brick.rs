use crate::math::hit::*;
use crate::math::ivec3::IVec3;
use crate::math::ray::RayI;
use crate::math::vec3::Vec3;
use crate::simd::packetray::Packet;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdi32::SimdI32;
use crate::simd::simdu32::SimdU32;
use crate::world::voxel::VoxelInfo;

type Material = u32;

#[derive(Debug, Copy, Clone)]
pub struct Brick {
    pub voxels: [Material; Brick::BRICK_BUFFER_SIZE],
}

impl Brick {
    pub const BRICK_WIDTH_VOXELS: usize = 8;
    pub const BRICK_UNIT_WIDTH: f32 = Self::BRICK_WIDTH_VOXELS as f32 * VoxelInfo::VOXEL_SIZE;

    pub const BRICK_BUFFER_SIZE: usize =
        Self::BRICK_WIDTH_VOXELS * Self::BRICK_WIDTH_VOXELS * Self::BRICK_WIDTH_VOXELS;
    pub const BRICK_MAX: Vec3 = Vec3::new(
        Self::BRICK_WIDTH_VOXELS as f32,
        Self::BRICK_WIDTH_VOXELS as f32,
        Self::BRICK_WIDTH_VOXELS as f32,
    );

    pub fn dda(
        &self,
        rays: Packet,
        mut active_rays: SimdU32,
        entry_distances: SimdF32,
        base_coords: Packed_Vec3_x4,
        mut hit_distances: SimdF32,
        mut current_normals: Packed_Vec3_x4,
        mut hit_normals: Packed_Vec3_x4,
        mut hit_colors: SimdU32,
        ignore_water_mask: SimdU32
    ) -> (SimdU32, SimdF32, Packed_Vec3_x4, SimdU32) {
        let cell_size = SimdF32::from_float(VoxelInfo::VOXEL_SIZE);
        let cell_size_recip = cell_size.recip();

        let steps = rays.directions.signs();
        let step_xs = steps.xs.to_i32();
        let step_ys = steps.ys.to_i32();
        let step_zs = steps.zs.to_i32();

        let deltas = rays.inverse_dirs.abs();
        let delta_xs = deltas.xs * cell_size;
        let delta_ys = deltas.ys * cell_size;
        let delta_zs = deltas.zs * cell_size;

        const NUDGE: SimdF32 = SimdF32::TINY;

        let entry_pos_xs = (rays.directions.xs.mul_then_add(entry_distances + NUDGE, rays.origins.xs) - base_coords.xs) * cell_size_recip;
        let entry_pos_ys = (rays.directions.ys.mul_then_add(entry_distances + NUDGE, rays.origins.ys) - base_coords.ys) * cell_size_recip;
        let entry_pos_zs = (rays.directions.zs.mul_then_add(entry_distances + NUDGE, rays.origins.zs) - base_coords.zs) * cell_size_recip;

        let local_xs_f = entry_pos_xs.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(7.0));
        let local_ys_f = entry_pos_ys.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(7.0));
        let local_zs_f = entry_pos_zs.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(7.0));

        let mut tmax_xs = (local_xs_f - entry_pos_xs + steps.xs.max(SimdF32::ZEROES)) * cell_size * rays.inverse_dirs.xs + entry_distances + NUDGE;
        let mut tmax_ys = (local_ys_f - entry_pos_ys + steps.ys.max(SimdF32::ZEROES)) * cell_size * rays.inverse_dirs.ys + entry_distances + NUDGE;
        let mut tmax_zs = (local_zs_f - entry_pos_zs + steps.zs.max(SimdF32::ZEROES)) * cell_size * rays.inverse_dirs.zs + entry_distances + NUDGE;

        let mut local_xs = local_xs_f.to_i32();
        let mut local_ys = local_ys_f.to_i32();
        let mut local_zs = local_zs_f.to_i32();

        let mut mins = entry_distances + NUDGE;
        
        let mut alive_for_multigrid = active_rays;

        for _ in 0..24 {
            let in_bounds = (local_xs.great_or_eq(SimdI32::ZEROES) & local_xs.less_than(SimdI32::from(8)))
                & (local_ys.great_or_eq(SimdI32::ZEROES) & local_ys.less_than(SimdI32::from(8)))
                & (local_zs.great_or_eq(SimdI32::ZEROES) & local_zs.less_than(SimdI32::from(8)));

            active_rays = active_rays & in_bounds;

            if active_rays.has_none_active() {
                break;
            }

            let indices_simd = local_xs.add(local_ys.shll::<3>()).add(local_zs.shll::<6>());
            let indices = indices_simd.to_array();
            let active_arr = active_rays.to_array();
            
            let mut voxels = [0u32; 4];
            for i in 0..4 {
                if active_arr[i] != 0 {
                    voxels[i] = self.voxels[indices[i] as usize];
                }
            }
            let colors_simd = SimdU32::from(voxels);
            
            // Extract material byte (top byte of color)
            let materials = colors_simd.shrl::<24>();
            let is_water = materials.is_eq_to(SimdU32::from(crate::world::voxel::MAT_WATER as u32));
            let ignore_this = ignore_water_mask & is_water;
            
            let hit_mask = colors_simd.not_zero() & !ignore_this;
            let hits = active_rays & hit_mask;

            if hits.has_active() {
                hit_distances = SimdF32::select(hits, mins, hit_distances);
                
                hit_normals.xs = SimdF32::select(hits, current_normals.xs, hit_normals.xs);
                hit_normals.ys = SimdF32::select(hits, current_normals.ys, hit_normals.ys);
                hit_normals.zs = SimdF32::select(hits, current_normals.zs, hit_normals.zs);
                
                hit_colors = hits.select(colors_simd, hit_colors);
                
                alive_for_multigrid = hits.select(SimdU32::ZEROES, alive_for_multigrid);
                active_rays = hits.select(SimdU32::ZEROES, active_rays);
            }

            if active_rays.has_none_active() {
                break;
            }

            mins = tmax_xs.min(tmax_ys).min(tmax_zs);
            let x_delta_mask = mins.is_eq_to(tmax_xs);
            let y_delta_mask = mins.is_eq_to(tmax_ys);
            let z_delta_mask = mins.is_eq_to(tmax_zs);

            tmax_xs = SimdF32::select(x_delta_mask, tmax_xs + delta_xs, tmax_xs);
            tmax_ys = SimdF32::select(y_delta_mask, tmax_ys + delta_ys, tmax_ys);
            tmax_zs = SimdF32::select(z_delta_mask, tmax_zs + delta_zs, tmax_zs);

            local_xs = SimdI32::select(x_delta_mask, local_xs + step_xs, local_xs);
            local_ys = SimdI32::select(y_delta_mask, local_ys + step_ys, local_ys);
            local_zs = SimdI32::select(z_delta_mask, local_zs + step_zs, local_zs);
            
            let neg_step_xs = step_xs.to_f32() * SimdF32::NEGS;
            let neg_step_ys = step_ys.to_f32() * SimdF32::NEGS;
            let neg_step_zs = step_zs.to_f32() * SimdF32::NEGS;
            
            current_normals.xs = SimdF32::select(x_delta_mask, neg_step_xs, SimdF32::ZEROES);
            current_normals.ys = SimdF32::select(y_delta_mask, neg_step_ys, SimdF32::ZEROES);
            current_normals.zs = SimdF32::select(z_delta_mask, neg_step_zs, SimdF32::ZEROES);
        }

        (alive_for_multigrid, hit_distances, hit_normals, hit_colors)
    }

    pub fn traverse(&self, rayi: RayI, entry_pos: Vec3, entry_distance: f32) -> HitInfo {
        const WIDTH: i32 = Brick::BRICK_WIDTH_VOXELS as i32;
        const MAX_DEPTH: usize = Brick::BRICK_WIDTH_VOXELS * 3;

        let step: IVec3 = rayi.direction.signs().into();

        let delta = rayi.inv_dir.abs();

        let mut distance_to_next = Vec3::tmax_from(step, entry_pos, rayi.inv_dir);

        let mut position = IVec3::clamp_to(IVec3::from(entry_pos), 0, WIDTH - 1);

        let mut axis = entry_pos.max_axis();

        for _ in 0..MAX_DEPTH {
            let flat_index: usize =
                (position.x + (position.y * WIDTH) + (position.z * WIDTH * WIDTH)) as usize;

            let voxel = self.voxels[flat_index];

            if voxel != 0 {
                return HitInfo::new(
                    Vec3::get_face_normal(axis, -step[axis]),
                    entry_distance + distance_to_next[axis] - delta[axis],
                    voxel,
                    true,
                );
            }

            axis = distance_to_next.min_axis();

            position[axis] += step[axis];

            if position[axis] < 0 || WIDTH <= position[axis] {
                return HitInfo::exit_volume_at(
                    entry_distance + distance_to_next[axis] - delta[axis],
                );
            }

            distance_to_next[axis] += delta[axis];
        }

        return HitInfo::DUD;
    }
}
