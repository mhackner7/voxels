use crate::math::hit::simd_aabb;
use crate::math::vec3::Vec3;
use crate::simd::packetray::*;
use crate::simd::packetvec::*;
use crate::simd::simdf32::*;
use crate::simd::simdi32::*;
use crate::simd::simdu32::*;
use crate::world::leaf::Leaf;
use crate::world::voxel::*;

pub struct Tree64 {
    /// pool of branch nodes
    /// the first node is the root node
    branches: Vec<Branch>,

    /// pool of leaf bricks
    leaves: Vec<Leaf>,

    coords: Vec3,
    coords_simd: Packed_Vec3_x4,
}

impl Tree64 {
    /// however many times it can be subdivided
    pub const MAX_DEPTH: usize = 4;
    pub const MAX_DEPTH_F: f32 = Self::MAX_DEPTH as f32;

    /// cell width at a depth of 1.0, useful for calculations
    pub const BASE_CELL_WIDTH: f32 = Tree64::AXIS_WIDTH.pow(Tree64::MAX_DEPTH as u32) as f32
        * Leaf::VOXELS_PER_AXIS_F32
        * VoxelInfo::VOXEL_SIZE
        / Tree64::AXIS_WIDTH_F;

    /// how many cells are in one axis of subdivision
    pub const AXIS_WIDTH: usize = 4;
    pub const AXIS_WIDTH_F: f32 = 4.0;

    /// how many total subdivisions there are per layer (same for all)
    pub const MAX_SUBDIVISIONS: usize = Tree64::AXIS_WIDTH.pow(3);
    pub const MAX_SUBDIVISIONS_F: f32 = Tree64::AXIS_WIDTH.pow(3) as f32;

    /// not coords, just use for volume measurements
    pub const BMIN: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const BMAX: Vec3 = Vec3::new_filled(Tree64::BASE_CELL_WIDTH * Tree64::AXIS_WIDTH_F);

    pub const BMIN_SIMD: Packed_Vec3_x4 = Packed_Vec3_x4::from_single_const(Tree64::BMIN);
    pub const BMAX_SIMD: Packed_Vec3_x4 = Packed_Vec3_x4::from_single_const(Tree64::BMAX);

    /// trace the packet within the volume,
    /// return the 4 color values needed for the pixel buffer
    pub fn trace(packet: Packet) /* -> [u32;4] */
    {
        // idk if i need ts to be HERE vs the function, disregard for now
        let mut depth = 0i32;

        // simd aabb intersection test with volume
        // get the intersection entry distances and active mask
        let (intersection_ts, active_rays) = simd_aabb(
            packet.origins,
            packet.inverse_dirs,
            Self::BMIN_SIMD,
            Self::BMAX_SIMD,
        );
    }

    /// should return:
    ///     -the next set of masks to traverse
    ///     -the distances of the rays
    ///     -active rays
    pub fn trace_branch_simd(
        &self,
        packet: Packet,
        hit_distances: SimdF32,
        active_rays: SimdU32,
        branches: [Branch; 4],
        depth: i32,
    ) -> ([Material; 4], [f32; 4], [u32; 4]) {
        // split up the packet //
        let origins = packet.origins;
        let directions = packet.directions;
        let inverse_dirs = packet.inverse_dirs;

        // set up branch masks //
        // no 64 bit simd, so uses 2 32 bit registers
        let lo = SimdU32::from([
            branches[0].mask as u32,
            branches[1].mask as u32,
            branches[2].mask as u32,
            branches[3].mask as u32,
        ]);
        let hi = SimdU32::from([
            (branches[0].mask >> 32) as u32,
            (branches[1].mask >> 32) as u32,
            (branches[2].mask >> 32) as u32,
            (branches[3].mask >> 32) as u32,
        ]);
        // selects which bitmask to pick
        // 32 for use with gre function
        let mask_selector = SimdU32::from(32);
        // used to subtract the indices
        let mask_clamp = SimdU32::from(31);

        // the step amount is just +1 if the sign is positive and -1 if negative //
        let step_xs = directions.xs.signs();
        let step_ys = directions.ys.signs();
        let step_zs = directions.zs.signs();

        /*
            cell size = size of the unit of traversal of the current node
            given a 512^3 volume of 0.125m sized voxels:
            - at a depth of _, each cell is ___ wide
                            1      -->      16m
                            2      -->       4m
                            3      -->       1m

            for dda, the cell size is used to figure out the movement delta while tracing

            the inverse cell size (just to avoid dividing by the cell size)
            is used to convert from world space into local tree space
        */
        let cell_size = SimdF32::from_float(Tree64::BASE_CELL_WIDTH / 4.0f32.powi(depth));
        let inv_cell_size = cell_size.recip();

        let ida = inverse_dirs.abs();

        let deltas_xs = ida.xs * cell_size;
        let deltas_ys = ida.ys * cell_size;
        let deltas_zs = ida.zs * cell_size;

        // the entry point into the volume, nudged in, then scaled to cell size, used for tmax and current trackers //

        let entry_xs = (directions
            .xs
            .mul_then_add(hit_distances.add(0.0001.into()), origins.xs)
            - Tree64::BMIN_SIMD.xs)
            * inv_cell_size;

        let entry_ys = (directions
            .ys
            .mul_then_add(hit_distances.add(0.0001.into()), origins.ys)
            - Tree64::BMIN_SIMD.ys)
            * inv_cell_size;

        let entry_zs = (directions
            .zs
            .mul_then_add(hit_distances.add(0.0001.into()), origins.zs)
            - Tree64::BMIN_SIMD.zs)
            * inv_cell_size;

        let upper_bound = SimdF32::from_float(3.9999);

        // FINALLY THE FIX TO THE WEIRD GODLY PILLARS FUCK MATH
        let current_xs = entry_xs.max(SimdF32::ZEROES).min(upper_bound).floor();
        let current_ys = entry_ys.max(SimdF32::ZEROES).min(upper_bound).floor();
        let current_zs = entry_zs.max(SimdF32::ZEROES).min(upper_bound).floor();

        // the distance to the next voxel volume, primary dda driver //

        let mut tmax_xs = hit_distances
            + (current_xs.floor() - entry_xs + step_xs.max(SimdF32::ZEROES))
                * inverse_dirs.xs
                * cell_size;

        let mut tmax_ys = hit_distances
            + (current_ys.floor() - entry_ys + step_ys.max(SimdF32::ZEROES))
                * inverse_dirs.ys
                * cell_size;

        let mut tmax_zs = hit_distances
            + (current_zs.floor() - entry_zs + step_zs.max(SimdF32::ZEROES))
                * inverse_dirs.zs
                * cell_size;

        // used to fix nan errors //

        tmax_xs = SimdF32::select(tmax_xs.is_ne_to(tmax_xs), SimdF32::INFY, tmax_xs);

        tmax_ys = SimdF32::select(tmax_ys.is_ne_to(tmax_ys), SimdF32::INFY, tmax_ys);

        tmax_zs = SimdF32::select(tmax_zs.is_ne_to(tmax_zs), SimdF32::INFY, tmax_zs);

        // step and current are +/- 1 voxel, i32 suits better //

        let step_xs = step_xs.to_i32();
        let step_ys = step_ys.to_i32();
        let step_zs = step_zs.to_i32();

        let mut current_xs = current_xs.to_i32();
        let mut current_ys = current_ys.to_i32();
        let mut current_zs = current_zs.to_i32();

        // to clamp the current vectors to only valid array spots
        let min_curr = SimdI32::ZEROES;
        let max_curr = SimdI32::from(3);

        let mut result_voxels = [Material::Air; 4];
        let mut result_distances = hit_distances.to_array();
        let mut result_hits = [0u32; 4];

        let mut result_mins = hit_distances.to_array();

        let mut remaining = active_rays; // the rays that are still valid for the given volume

        for _ in 0..(4 + 4 + 4) {
            if remaining.has_none_active() {
                break;
            }

            // the indices for the bitmask of current packet
            let indices = current_xs
                .or(current_ys.shll::<2>())
                .or(current_zs.shll::<4>())
                .reinterpret_as_u32();

            // get the rays that hit active volumes //

            // get the proper mask half
            let hi_rays = indices.great_or_eq(mask_selector);

            let bitmask = hi_rays.select(hi, lo);

            let clamped_indices = indices & mask_clamp;

            // test mask with rays
            let hit_rays = (bitmask & SimdU32::ONES.shllv(clamped_indices)).not_zero() & remaining;

            let hits = hit_rays.to_array();
            let active = active_rays.to_array();

            // perform hit logic
            for i in 0..4 {
                // trace the leaf
                if branches[i].has_leaves {
                }
                // trace the next branch
                else {
                }
            }

            // trace hit rays
            if hit_rays.has_active() {}
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

            let exit_rays = remaining & (x_invalid | y_invalid | z_invalid).not();

            if exit_rays.has_active() {
                distances = SimdF32::select(exit_rays, mins, hit_distances);
            }
        }

        (result_voxels, result_distances, result_hits)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Branch {
    /// 4x4x4 occupancy mask, 1 = a node or leaf is in this subcell (depends on depth)
    mask: u64,

    /// pointer to start of children in given pool
    child_pointer: u32,

    /// if the children of this branch are leaves or not
    has_leaves: bool,
}

impl Branch {}

#[derive(Clone, Copy, Debug)]
pub struct DDAinfo {
    pub descended: SimdU32,
    pub distances: SimdF32,
    pub exited: SimdU32,
    pub indices: [u32; 4],
}
