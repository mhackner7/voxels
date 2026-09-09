use crate::math::hit::*;
use crate::math::vec3::Vec3;
use crate::simd::packetray::Packet;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdi32::SimdI32;
use crate::simd::simdu32::*;
use crate::world::brick::Brick;
use crate::world::mask::Mask;
use crate::world::voxel::*;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub mask: Mask,
    pub child_ptr: u32,
    pub is_leaf: bool, // true if points to bricks, false if points to nodes
}

pub struct Tree64 {
    pub nodes: Vec<Node>,
    pub bricks: Vec<Brick>,
    pub base_coords: Vec3,
    pub base_packed: Packed_Vec3_x4,
}

impl Tree64 {
    pub const BRICKS_PER_AXIS: usize = 256;
    pub const BMAX: Vec3 = Vec3::new(256.0, 256.0, 256.0);
    
    pub fn trace_packet(grid: &Tree64, rays: Packet, current_max_distances: SimdF32, ignore_water_mask: SimdU32) -> (SimdF32, Packed_Vec3_x4, SimdU32) {
        let (distances, hit_mask) = simd_aabb(
            rays.origins,
            rays.inverse_dirs,
            grid.base_coords,
            grid.base_coords + Tree64::BMAX,
        );

        let active_rays = hit_mask.not_zero() & distances.is_less_than(current_max_distances);
        
        if active_rays.has_none_active() {
            return (current_max_distances, Packed_Vec3_x4::ZEROES, SimdU32::ZEROES);
        }

        let (_, hit_distances, hit_normals, hit_colors) = grid.dda_node(
            0, // root node is at index 0
            64.0, // root cell size
            rays,
            active_rays,
            distances,
            grid.base_packed,
            current_max_distances, // Start with max distance
            Packed_Vec3_x4::ZEROES,
            Packed_Vec3_x4::ZEROES,
            SimdU32::ZEROES,
            ignore_water_mask
        );

        (hit_distances, hit_normals, hit_colors)
    }

    pub fn dda_node(
        &self,
        node_idx: usize,
        cell_size: f32,
        rays: Packet,
        mut active_rays: SimdU32,
        entry_distances: SimdF32,
        node_base_packed: Packed_Vec3_x4,
        mut hit_distances: SimdF32,
        mut current_normals: Packed_Vec3_x4,
        mut hit_normals: Packed_Vec3_x4,
        mut hit_colors: SimdU32,
        ignore_water_mask: SimdU32
    ) -> (SimdU32, SimdF32, Packed_Vec3_x4, SimdU32) {
        let steps = rays.directions.signs();
        let step_xs = steps.xs.to_i32();
        let step_ys = steps.ys.to_i32();
        let step_zs = steps.zs.to_i32();

        let deltas = rays.inverse_dirs.abs();

        let cell_size_simd = SimdF32::from_float(cell_size);
        let delta_xs = deltas.xs * cell_size_simd;
        let delta_ys = deltas.ys * cell_size_simd;
        let delta_zs = deltas.zs * cell_size_simd;

        const NUDGE: SimdF32 = SimdF32::TINY;
        let cell_size_recip = cell_size_simd.recip();

        let entry_pos_xs = (rays.directions.xs.mul_then_add(entry_distances + NUDGE, rays.origins.xs) - node_base_packed.xs) * cell_size_recip;
        let entry_pos_ys = (rays.directions.ys.mul_then_add(entry_distances + NUDGE, rays.origins.ys) - node_base_packed.ys) * cell_size_recip;
        let entry_pos_zs = (rays.directions.zs.mul_then_add(entry_distances + NUDGE, rays.origins.zs) - node_base_packed.zs) * cell_size_recip;

        let local_xs = entry_pos_xs.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(3.0));
        let local_ys = entry_pos_ys.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(3.0));
        let local_zs = entry_pos_zs.floor().clamp_to(SimdF32::ZEROES, SimdF32::from_float(3.0));

        let mut tmax_xs = (local_xs - entry_pos_xs + steps.xs.max(SimdF32::ZEROES)) * cell_size_simd * rays.inverse_dirs.xs + entry_distances + NUDGE;
        let mut tmax_ys = (local_ys - entry_pos_ys + steps.ys.max(SimdF32::ZEROES)) * cell_size_simd * rays.inverse_dirs.ys + entry_distances + NUDGE;
        let mut tmax_zs = (local_zs - entry_pos_zs + steps.zs.max(SimdF32::ZEROES)) * cell_size_simd * rays.inverse_dirs.zs + entry_distances + NUDGE;

        let mut local_xs = local_xs.to_i32();
        let mut local_ys = local_ys.to_i32();
        let mut local_zs = local_zs.to_i32();

        let mut mins = entry_distances + NUDGE;
        
        let mut alive_for_multigrid = active_rays;

        let node = &self.nodes[node_idx];
        let mask_bits = node.mask.bits;
        
        let hi_mask = SimdU32::from((mask_bits >> 32) as u32);
        let lo_mask = SimdU32::from(mask_bits as u32);

        let ones = SimdI32::from(1);
        let twos = SimdI32::from(2);

        for _ in 0..12 {
            let in_bounds = (local_xs.great_or_eq(SimdI32::ZEROES)
                & local_xs.less_than(SimdI32::FOURS))
                & (local_ys.great_or_eq(SimdI32::ZEROES) & local_ys.less_than(SimdI32::FOURS))
                & (local_zs.great_or_eq(SimdI32::ZEROES) & local_zs.less_than(SimdI32::FOURS));

            active_rays = active_rays & in_bounds;

            if active_rays.has_none_active() {
                return (alive_for_multigrid, hit_distances, hit_normals, hit_colors);
            }

            let indices = local_xs.add(local_ys.shll::<2>()).add(local_zs.shll::<4>()).reinterpret_as_u32();

            let mask_selecter = indices.great_or_eq(SimdU32::THIRTY_TWO);
            let target_mask = mask_selecter.select(hi_mask, lo_mask);
            let bitmask = SimdU32::ONES.shllv(indices & SimdU32::THIRTY_ONE);
            
            let hit_cells = target_mask & bitmask;
            let rays_to_trace = active_rays & hit_cells.not_zero();

            if rays_to_trace.has_active() {
                let mut processed_lanes = 0u32;
                let mut new_active_rays = active_rays;
                
                let rays_to_trace_arr = rays_to_trace.to_array();
                let indices_arr = indices.to_array();

                for i in 0..4 {
                    if rays_to_trace_arr[i] != 0 && (processed_lanes & (1 << i)) == 0 {
                        let inner_idx = node.mask.vector_index_for(indices_arr[i]);

                        let mut group_mask = 0u32;
                        for j in i..4 {
                            if rays_to_trace_arr[j] != 0 && node.mask.vector_index_for(indices_arr[j]) == inner_idx {
                                group_mask |= 1 << j;
                            }
                        }
                        processed_lanes |= group_mask;

                        let mut group_arr = [0u32; 4];
                        for j in 0..4 {
                            if (group_mask & (1 << j)) != 0 {
                                group_arr[j] = u32::MAX;
                            }
                        }
                        let simd_group_mask = SimdU32::from(group_arr);

                        let inner_base_xs = node_base_packed.xs + (local_xs.to_f32() * cell_size_simd);
                        let inner_base_ys = node_base_packed.ys + (local_ys.to_f32() * cell_size_simd);
                        let inner_base_zs = node_base_packed.zs + (local_zs.to_f32() * cell_size_simd);
                        let inner_base = Packed_Vec3_x4 { xs: inner_base_xs, ys: inner_base_ys, zs: inner_base_zs };

                        let (updated_active, updated_distances, updated_normals, updated_colors) = if node.child_ptr == u32::MAX {
                            // Completely solid node (usually underground stone)
                            let hits = simd_group_mask;
                            let stone_color = SimdU32::from(VoxelInfo::ARRAY[MAT_STONE as usize].color);
                            (
                                !hits, // rays that missed (none)
                                SimdF32::select(hits, mins, hit_distances),
                                current_normals, // normals are just the current cell entry normals
                                hits.select(stone_color, hit_colors)
                            )
                        } else if node.is_leaf {
                            let brick_idx = node.child_ptr as usize + inner_idx;
                            let brick = &self.bricks[brick_idx];
                            brick.dda(
                                rays, simd_group_mask, mins, inner_base, hit_distances, current_normals, hit_normals, hit_colors, ignore_water_mask
                            )
                        } else {
                            let child_node_idx = node.child_ptr as usize + inner_idx;
                            self.dda_node(
                                child_node_idx,
                                cell_size / 4.0,
                                rays, simd_group_mask, mins, inner_base, hit_distances, current_normals, hit_normals, hit_colors, ignore_water_mask
                            )
                        };
                        
                        hit_distances = SimdF32::select(simd_group_mask, updated_distances, hit_distances);
                        hit_normals.xs = SimdF32::select(simd_group_mask, updated_normals.xs, hit_normals.xs);
                        hit_normals.ys = SimdF32::select(simd_group_mask, updated_normals.ys, hit_normals.ys);
                        hit_normals.zs = SimdF32::select(simd_group_mask, updated_normals.zs, hit_normals.zs);
                        hit_colors = simd_group_mask.select(updated_colors, hit_colors);
                        
                        let hits_in_child = simd_group_mask & !updated_active;
                        alive_for_multigrid = hits_in_child.select(SimdU32::ZEROES, alive_for_multigrid);
                        new_active_rays = hits_in_child.select(SimdU32::ZEROES, new_active_rays);
                    }
                }
                active_rays = new_active_rays;
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
}
