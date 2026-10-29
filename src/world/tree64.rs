use crate::math::hit::simd_aabb;
use crate::math::vec3::Vec3;
use crate::simd::packetray::*;
use crate::simd::packetvec::*;
use crate::simd::simdf32::*;
use crate::simd::simdi32::*;
use crate::simd::simdu32::*;
use crate::world::leaf::{Brick, Leaf};
use crate::world::voxel::*;

pub struct Tree64 {
    /// pool of branch nodes
    /// the first node is the root node
    pub branches: Vec<Branch>,

    /// pool of leaf occupancy grids, children of depth 3 branches
    pub leaves: Vec<Leaf>,

    /// pool of leaf materials, bricks[i] belongs to leaves[i]
    pub bricks: Vec<Brick>,

    /// what each 2x2x2 block looks like from far away, for lod hits, leaf_block_materials[i]
    /// belongs to leaves[i]. low nibble = material of the block's highest voxel,
    /// high nibble = highest non water voxel (for rays looking through water)
    pub leaf_block_materials: Vec<[u8; 64]>,

    /// which 2x2x2 blocks of each leaf have anything in them (4x4x4 of them, one u64),
    /// leaf_blocks[i] belongs to leaves[i], empty blocks get stepped over in one go
    /// block index = (x >> 1) | ((y >> 1) << 2) | ((z >> 1) << 4)
    /// [0] = any voxel, [1] = any non water voxel, so rays looking through water skip water only blocks too
    pub leaf_blocks: Vec<[u64; 2]>,

    /// skip map, per 1m cell: chebyshev distance (in meters, capped at SKIP_MAX) to the
    /// nearest meter with anything in it. a meter at distance d sits inside an empty cube
    /// reaching d - 1 meters each way, so rays can jump the whole cube.
    /// [0] counts water as something, [1] doesn't (for rays looking through water)
    pub skip: Vec<[u8; 2]>,

    /// one bit per leaf, set when the leaf holds any water voxel
    /// lets water skipping rays avoid the brick read on dry leaves
    pub water_leaves: Vec<u64>,

    /// world space max corner of everything that is occupied
    /// rays above the highest voxel never enter the traversal loop
    pub bmax: Packed_Vec3_x4,
}

/// result of tracing one packet
#[derive(Clone, Copy, Debug)]
pub struct PacketHit {
    /// lanes that hit something
    pub hit: SimdU32,
    /// distance to where the ray entered the hit cell
    pub distances: SimdF32,
    /// axis aligned face normal of the hit cell, zero for misses
    pub normals: Packed_Vec3_x4,
    /// Air for misses and any hit traces
    pub materials: [Material; 4],
}

impl Tree64 {
    /// depth of the leaf level (0.25m blocks), the 4 branch levels (64m, 16m, 4m, 1m cells) sit above it
    /// node_stack[LEAF_DEPTH] holds the index of the current leaf
    pub const LEAF_DEPTH: usize = 4;

    /// depth of the single voxels inside a 2x2x2 block, node_stack[VOXEL_DEPTH] is just the leaf again
    pub const VOXEL_DEPTH: usize = 5;

    /// voxels along one axis of the volume (2^11)
    pub const VOXELS_PER_AXIS: i32 = 2048;

    /// log2 of the cell width in voxels at each depth
    /// depth 0 = 64m, 1 = 16m, 2 = 4m, 3 = 1m (leaf), 4 = 0.25m (block), 5 = 0.125m (voxel)
    pub const CELL_SHIFT: [u32; Self::VOXEL_DEPTH + 1] = [9, 7, 5, 3, 1, 0];

    /// meters along one axis of the volume, the skip map's size
    pub const METERS_PER_AXIS: usize = (Self::VOXELS_PER_AXIS >> 3) as usize;

    /// skip map distances stop counting here (meters)
    pub const SKIP_MAX: u8 = 15;

    /// skip map index of a meter
    #[inline(always)]
    pub fn skip_index([x, y, z]: [usize; 3]) -> usize {
        (z * Self::METERS_PER_AXIS + y) * Self::METERS_PER_AXIS + x
    }

    /// width of the volume in meters (256m)
    pub const WIDTH: f32 = Self::VOXELS_PER_AXIS as f32 * VOXEL_SIZE;

    /// world space min corner of the volume
    pub const BMIN_SIMD: Packed_Vec3_x4 = Packed_Vec3_x4::from_single_const(Vec3::new_filled(0.0));

    /// safety net for the traversal loop
    /// every step advances one axis by at least one voxel, so a ray needs
    /// at most 3 * 2048 voxel steps plus the coarser steps to leave the volume
    const MAX_STEPS: usize = 8192;

    /*
        trace the packet within the volume

        ANY_HIT    = shadow rays, stop at the first occupied cell, no material reads
        SKIP_WATER = water voxels / cells count as air (shadow, bounce, refraction)
        active     = lanes that need a ray at all
        max_ts     = lanes are dropped (miss) once they get past this distance
        cone       = lod, how many voxels wide the ray gets per meter it travels (0 = exact).
                     once a 2x2x2 block is no wider than the ray it's just hit as a whole,
                     shadow rays (ANY_HIT) do the same with any bigger cell too
    */
    #[inline(always)]
    pub fn trace<const ANY_HIT: bool, const SKIP_WATER: bool>(
        &self,
        packet: Packet,
        active: SimdU32,
        max_ts: SimdF32,
        cone: f32,
    ) -> PacketHit {
        self.trace_from::<ANY_HIT, SKIP_WATER>(packet, active, max_ts, cone, 0.0)
    }

    /// trace starting at start (meters), the caller promises nothing's in the way before that
    /// and every lane is still inside the volume (see Engine::beam_start)
    #[inline(always)]
    pub fn trace_from<const ANY_HIT: bool, const SKIP_WATER: bool>(
        &self,
        packet: Packet,
        active: SimdU32,
        max_ts: SimdF32,
        cone: f32,
        start: f32,
    ) -> PacketHit {
        let mut walk = PacketWalk::begin(self, packet, active, max_ts, cone, start);

        // occupancy skips dead lanes, so a packet that starts out empty just falls right through //

        for _ in 0..Self::MAX_STEPS {
            walk.occupancy::<ANY_HIT, SKIP_WATER>(self);

            if walk.active.has_none_active() {
                break;
            }

            walk.step();
        }

        walk.finish()
    }

    /// scalar point query at integer voxel coords (0..2047), None for air
    /// ground truth for the traversal tests, collision reads it as well
    #[inline(always)]
    pub fn voxel_at(&self, x: u32, y: u32, z: u32) -> Option<Material> {
        let mut node = 0usize;

        for depth in 0..Self::LEAF_DEPTH {
            let shift = Self::CELL_SHIFT[depth];

            let cell = ((x >> shift) & 3) | (((y >> shift) & 3) << 2) | (((z >> shift) & 3) << 4);
            let bit = 1u64 << cell;
            let branch = &self.branches[node];

            if branch.solid & bit != 0 {
                return Some(SOLID_MATERIAL);
            }

            if branch.water & bit != 0 {
                return Some(Material::Water);
            }

            if branch.children & bit == 0 {
                return None;
            }

            node = (branch.child_pointer + (branch.children & (bit - 1)).count_ones()) as usize;
        }

        let voxel = ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6)) as usize;

        if (self.leaves[node].voxels[voxel >> 6] >> (voxel & 63)) & 1 != 0 {
            Some(self.bricks[node].get(voxel))
        } else {
            None
        }
    }
}

/// traversal state of one packet, see Tree64::trace
struct PacketWalk {
    // results //
    materials: [Material; 4],
    distances: [f32; 4],
    hit_flags: [u32; 4],

    /// lanes still walking
    active: SimdU32,
    /// ray width in voxels per meter traveled, for lod
    cone: f32,
    /// distance at which each ray entered its current cell
    ts: SimdF32,
    max_ts: SimdF32,

    /// integer voxel coords (0..2047) per axis, always inside the current cell
    voxels: [SimdI32; 3],

    /// current depth per lane
    depths: SimdI32,

    /// log2 width (voxels) of the empty cell each lane steps out of next
    shifts: [i32; 4],
    /// voxels of empty space around the lane's meter from the skip map, 0 = just step the cell
    reaches: [i32; 4],

    /// per ray traversal stack, [depth] = node index at that depth, 0 is the root
    node_stack: [[u32; Tree64::VOXEL_DEPTH + 1]; 4],

    /// per axis: all ones in the lanes that step towards +axis
    pos: [SimdU32; 3],

    /// per axis: crossed by the last step, that's the hit normal
    last: [SimdU32; 3],

    /// origins / directions in voxel space, exit voxel = floor(o_v + d_v * t)
    voxel_origins: [SimdF32; 3],
    voxel_dirs: [SimdF32; 3],

    /// VOXEL_SIZE * inv_dir
    scaled_invs: [SimdF32; 3],
}

impl PacketWalk {
    #[inline(always)]
    fn begin(
        tree: &Tree64,
        packet: Packet,
        active: SimdU32,
        max_ts: SimdF32,
        cone: f32,
        start: f32,
    ) -> Self {
        // split up the packet //

        let origins = packet.origins;
        let directions = packet.directions;
        let inverse_dirs = packet.inverse_dirs;

        // entry distance into the occupied volume, misses never get traced //

        let (entry_ts, in_volume) = simd_aabb(origins, inverse_dirs, Tree64::BMIN_SIMD, tree.bmax);

        let active = active & in_volume & entry_ts.is_less_or_equal(max_ts);

        // the axis of the slab the ray entered through is the first normal //

        let (o, inv, max) = (origins.axes(), inverse_dirs.axes(), tree.bmax.axes());
        let near = per_axis(|a| (o[a].neg() * inv[a]).min((max[a] - o[a]) * inv[a]));

        let last_x = near[0].is_greater_or_equal(near[1]) & near[0].is_greater_or_equal(near[2]);
        let last_y = near[1].is_greater_or_equal(near[2]) & last_x.not();
        let last_z = (last_x | last_y).not();

        /*
            the ray position is tracked as integer voxel coords (0..2047)
            that are always inside the current cell, floats are only used
            to compare boundary distances, so no epsilon nudging is needed
        */

        let ts = SimdF32::select(
            active,
            entry_ts.max(SimdF32::from_float(start)),
            SimdF32::ZEROES,
        );

        let vpm = SimdF32::from_float(VOXELS_PER_METER);
        let max_voxel = SimdI32::from(Tree64::VOXELS_PER_AXIS - 1);

        let voxel_origins = origins.scale(vpm).axes();
        let voxel_dirs = directions.scale(vpm).axes();

        /* ENTRY VOXEL */

        let entry = |d: SimdF32, o: SimdF32| {
            d.mul_then_add(ts, o)
                .floor_to_i32()
                .clamp_to(SimdI32::ZEROES, max_voxel)
        };

        Self {
            materials: [Material::Air; 4],
            distances: [f32::INFINITY; 4],
            hit_flags: [0; 4],
            active,
            cone,
            ts,
            max_ts,
            voxels: per_axis(|a| entry(voxel_dirs[a], voxel_origins[a])),
            depths: SimdI32::ZEROES,
            shifts: [Tree64::CELL_SHIFT[0] as i32; 4],
            reaches: [0; 4],
            node_stack: [[0; Tree64::VOXEL_DEPTH + 1]; 4],
            pos: per_axis(|a| directions.axes()[a].is_greater_or_equal(SimdF32::ZEROES)),
            last: [last_x, last_y, last_z],
            voxel_origins,
            voxel_dirs,
            // boundary distance = (voxel - voxel_origin) * (VOXEL_SIZE * inv_dir)
            // the subtraction is exact near the origin, rays starting on a boundary tie exactly
            scaled_invs: inverse_dirs.scale(SimdF32::from_float(VOXEL_SIZE)).axes(),
        }
    }

    /* OCCUPANCY + DESCEND */

    // every ray descends as deep as its current cell is occupied,
    // so each iteration ends in a hit or a step for every lane
    #[inline(always)]
    fn occupancy<const ANY_HIT: bool, const SKIP_WATER: bool>(&mut self, tree: &Tree64) {
        // cells that end a ray at branch level //

        let water_hits = if SKIP_WATER { 0 } else { u64::MAX };

        let [x_array, y_array, z_array] = per_axis(|a| self.voxels[a].to_array());
        let t_array = self.ts.to_array();
        let mut depth_array = self.depths.to_array();
        let mut active_array = self.active.to_array();

        // the last lane that went all the way down: its lane, where it ended up and how //
        let mut resolved: Option<(usize, u32, bool, i32)> = None;

        for i in 0..4 {
            if active_array[i] == 0 {
                continue;
            }

            let (x, y, z) = (x_array[i] as u32, y_array[i] as u32, z_array[i] as u32);
            let mut depth = depth_array[i] as usize;
            let mut group = false;

            // log2 of how many voxels wide the ray is by now, cells that size or smaller //
            // don't get looked inside anymore
            let lod = lod_shift(t_array[i] * self.cone);

            // ok so camera rays of a packet mostly sit in the same cell, a lane in the very //
            // same cell as the one before it (same lod too) just copies its answer and stack,
            // same spot means same ancestors, so that's exactly what it would have found

            if let Some((j, shift, hit, j_lod)) = resolved {
                let apart =
                    (x ^ x_array[j] as u32) | (y ^ y_array[j] as u32) | (z ^ z_array[j] as u32);

                if j_lod == lod && apart >> shift == 0 {
                    self.node_stack[i] = self.node_stack[j];
                    depth_array[i] = depth_array[j];
                    self.shifts[i] = self.shifts[j];
                    self.reaches[i] = self.reaches[j];

                    if hit {
                        active_array[i] = 0;
                        self.hit_flags[i] = u32::MAX;
                        self.distances[i] = t_array[i];
                        self.materials[i] = self.materials[j];
                    }

                    continue;
                }
            }

            // SAFETY: node indices come from child_pointer + popcount of the build,
            // they always point into the pool of their depth, and depth never passes VOXEL_DEPTH
            let hit = loop {
                let node = unsafe { *self.node_stack[i].get_unchecked(depth) } as usize;

                if depth == Tree64::LEAF_DEPTH {
                    let block = ((x >> 1) & 3) | (((y >> 1) & 3) << 2) | (((z >> 1) & 3) << 4);

                    // empty (or water only) 2x2x2 block, hop right over it, //
                    // or over 4x4x4 voxels if its whole group of blocks is empty too

                    let blocks =
                        unsafe { tree.leaf_blocks.get_unchecked(node) }[SKIP_WATER as usize];

                    // the ray covers a whole 4x4x4 group of blocks by now, so any block makes it solid //

                    if lod >= 2 {
                        let base = block & 0b101010;
                        let in_group = blocks & (GROUP << base);

                        if in_group == 0 {
                            group = true;
                            break false;
                        }

                        if !ANY_HIT {
                            let summary = unsafe { tree.leaf_block_materials.get_unchecked(node) };
                            self.materials[i] = top_material(in_group, summary, SKIP_WATER);
                        }

                        break true;
                    }

                    if (blocks >> block) & 1 == 0 {
                        group = (blocks >> (block & 0b101010)) & GROUP == 0;
                        break false;
                    }

                    // the ray is as wide as the block by now, so the block just counts as solid //

                    if lod >= 1 {
                        if !ANY_HIT {
                            let summary = unsafe { tree.leaf_block_materials.get_unchecked(node) };
                            self.materials[i] = block_material(summary, block as usize, SKIP_WATER);
                        }

                        break true;
                    }

                    unsafe { *self.node_stack[i].get_unchecked_mut(depth + 1) = node as u32 };
                    depth += 1;
                    continue;
                }

                if depth == Tree64::VOXEL_DEPTH {
                    let voxel = ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6)) as usize;

                    if (unsafe { tree.leaves.get_unchecked(node) }.voxels[voxel >> 6]
                        >> (voxel & 63))
                        & 1
                        == 0
                    {
                        break false;
                    }

                    // dry leaves never touch their brick when skipping water //

                    if SKIP_WATER
                        && (unsafe { tree.water_leaves.get_unchecked(node >> 6) } >> (node & 63))
                            & 1
                            != 0
                        && unsafe { tree.bricks.get_unchecked(node) }.get(voxel) == Material::Water
                    {
                        break false;
                    }

                    if !ANY_HIT {
                        self.materials[i] = unsafe { tree.bricks.get_unchecked(node) }.get(voxel);
                    }

                    break true;
                }

                let shift = unsafe { *Tree64::CELL_SHIFT.get_unchecked(depth) };

                let cell =
                    ((x >> shift) & 3) | (((y >> shift) & 3) << 2) | (((z >> shift) & 3) << 4);

                let bit = 1u64 << cell;
                let branch = unsafe { tree.branches.get_unchecked(node) };

                // uniform cells end the ray at any depth //

                if (branch.solid | (branch.water & water_hits)) & bit != 0 {
                    if !ANY_HIT {
                        self.materials[i] = if branch.solid & bit != 0 {
                            SOLID_MATERIAL
                        } else {
                            Material::Water
                        };
                    }

                    break true;
                }

                // empty cell (or skipped water), step over it, //
                // the whole 2x2x2 group of cells in one go if that's empty too

                if branch.children & bit == 0 {
                    let occupied = branch.children | branch.solid | (branch.water & water_hits);
                    group = (occupied >> (cell & 0b101010)) & GROUP == 0;
                    break false;
                }

                // shadow rays don't care what they hit, so any cell no wider than them is solid //

                if ANY_HIT && shift as i32 <= lod {
                    break true;
                }

                // occupancy masks are compressed, so:
                // child_pointer + number of set bits before current cell

                let child = branch.child_pointer + (branch.children & (bit - 1)).count_ones();

                // the ray is a whole meter wide by now, any non empty leaf is one solid cell //
                // (a water only leaf is nothing to a ray looking through water, it gets stepped)

                if !ANY_HIT && depth == Tree64::LEAF_DEPTH - 1 && lod >= 3 {
                    let blocks = unsafe { tree.leaf_blocks.get_unchecked(child as usize) }
                        [SKIP_WATER as usize];

                    if blocks == 0 {
                        break false;
                    }

                    let summary =
                        unsafe { tree.leaf_block_materials.get_unchecked(child as usize) };
                    self.materials[i] = top_material(blocks, summary, SKIP_WATER);
                    break true;
                }

                unsafe {
                    *self.node_stack[i].get_unchecked_mut(depth + 1) = child;
                }

                depth += 1;
            };

            // the hit distance is where the ray entered the hit cell //

            if hit {
                active_array[i] = 0;
                self.hit_flags[i] = u32::MAX;
                self.distances[i] = t_array[i];
            }

            depth_array[i] = depth as i32;

            let shift = Tree64::CELL_SHIFT[depth] + group as u32;
            self.shifts[i] = shift as i32;
            self.reaches[i] = 0;

            // ok so this is the sneaky part, an empty 16m, 4m or 1m cell might sit in a much bigger //
            // empty cube, the skip map knows how many empty meters surround each meter

            if !hit && shift >= 3 {
                let meter = [x, y, z].map(|c| (c >> 3) as usize);
                let empty = unsafe { tree.skip.get_unchecked(Tree64::skip_index(meter)) }
                    [SKIP_WATER as usize] as i32;
                let reach = (empty - 1) * 8;

                if 2 * reach + 8 > 1 << shift {
                    self.reaches[i] = reach;
                }
            }

            // a hit is only shared within the cell it happened in, an empty step within the box it took //
            let shared = if hit {
                Tree64::CELL_SHIFT[depth]
            } else {
                shift
            };
            resolved = Some((i, shared, hit, lod));
        }

        self.depths = SimdI32::from(depth_array);
        self.active = SimdU32::from(active_array);
    }

    /* STEP + ASCEND */

    #[inline(always)]
    fn step(&mut self) {
        let voxels = self.voxels;

        // box bounds in voxels, the far face is the one the ray exits through //
        // either the lane's own cell, or the empty cube around its meter. a cube can poke
        // out of the volume, but leaving through there means leaving the volume for good

        let sizes = SimdI32::ONES.shlv(SimdI32::from(self.shifts));
        let reaches = SimdI32::from(self.reaches);
        let cubes = reaches.reinterpret_as_u32().not_zero();
        let cube_sizes = reaches + reaches + SimdI32::from(8);

        let lo = per_axis(|a| {
            let c = voxels[a];
            SimdI32::select(cubes, (c & SimdI32::from(!7)) - reaches, c & sizes.neg())
        });
        let sizes = SimdI32::select(cubes, cube_sizes, sizes);
        let hi = per_axis(|a| lo[a] + sizes - SimdI32::ONES);

        let tmax = per_axis(|a| {
            let face = SimdI32::select(self.pos[a], hi[a] + SimdI32::ONES, lo[a]);
            (face.to_f32() - self.voxel_origins[a]) * self.scaled_invs[a]
        });

        let t_exit = tmax[0].min(tmax[1].min(tmax[2]));

        // only one axis crosses per step (x > y > z on exact ties),
        // the tied axis crosses on the next step at the same distance,
        // so rays through exact edges / corners never slip between voxels

        let mask_x = tmax[0].is_eq_to(t_exit);
        let mask_y = tmax[1].is_eq_to(t_exit) & mask_x.not();
        let masks = [mask_x, mask_y, (mask_x | mask_y).not()];

        // crossed axes land on the first voxel past the box,
        // the other axes are clamped into the box at the exit point

        let next = per_axis(|a| {
            let past = SimdI32::select(self.pos[a], hi[a] + SimdI32::ONES, lo[a] - SimdI32::ONES);
            let inside = self.voxel_dirs[a]
                .mul_then_add(t_exit, self.voxel_origins[a])
                .floor_to_i32()
                .clamp_to(lo[a], hi[a]);

            SimdI32::select(masks[a], past, inside)
        });

        /* ASCEND */

        // any bit above 2047 (or a negative coord) means the ray left the root, //
        // past max_ts the ray is done as well

        let outside = (next[0] | next[1] | next[2]) & SimdI32::from(!(Tree64::VOXELS_PER_AXIS - 1));

        self.active = self.active
            & outside.reinterpret_as_u32().not_zero().not()
            & t_exit.is_less_or_equal(self.max_ts);

        /*
            the highest bit that changed tells which ancestor still contains the ray
            node spans per depth are 2^11, 2^9, 2^7, 2^5, 2^3, 2^1 voxels, so for the
            highest changed bit h the deepest valid node is (10 - h) / 2,
            which is (leading_zeros - 21) / 2, stack entries above it stay valid
        */

        let changed = next[0].xor(voxels[0]) | next[1].xor(voxels[1]) | next[2].xor(voxels[2]);

        let pop_depths = (changed
            .reinterpret_as_u32()
            .leading_zeros()
            .reinterpret_as_i32()
            - SimdI32::from(21))
        .shr::<1>();

        self.depths = self.depths.min(pop_depths);

        // remember the crossed axis of lanes that are still moving, it is the hit normal //

        self.last = per_axis(|a| SimdU32::select_bits(self.active, masks[a], self.last[a]));
        self.voxels = next;
        self.ts = t_exit;
    }

    /// pack the per lane results, normal = -step direction on the last crossed axis
    #[inline(always)]
    fn finish(self) -> PacketHit {
        let hit = SimdU32::from(self.hit_flags);

        let normals = per_axis(|a| {
            let away = SimdF32::select(self.pos[a], SimdF32::NEGS, SimdF32::ONES);
            SimdF32::select(self.last[a] & hit, away, SimdF32::ZEROES)
        });

        PacketHit {
            hit,
            distances: SimdF32::from(self.distances),
            normals: Packed_Vec3_x4::from_axes(normals),
            materials: self.materials,
        }
    }
}

/// [f(0), f(1), f(2)], spelled out because [0, 1, 2].map pushed the simd values
/// out of registers and cost the traversal ~10%
#[inline(always)]
fn per_axis<T>(mut f: impl FnMut(usize) -> T) -> [T; 3] {
    [f(0), f(1), f(2)]
}

/// all 8 cells of a 2x2x2 group at the low corner of a 4x4x4 node, shifted by
/// (cell & 0b101010) it's the group of any cell (index = x | y << 2 | z << 4).
/// a whole empty group gets stepped over as one cell twice the size, that trick's
/// straight from dubiousconst282's 64-tree write up
const GROUP: u64 = 0x0033_0033;

/// floor(log2) of a footprint in voxels, straight off the float exponent,
/// comes out hugely negative for anything under one voxel (or 0)
#[inline(always)]
fn lod_shift(footprint: f32) -> i32 {
    // just the exponent bits, so a -0.0 footprint doesn't sneak in through the sign //
    ((footprint.to_bits() >> 23) & 0xFF) as i32 - 127
}

/// how a lumped bunch of blocks looks from afar: the summary material of its highest block
#[inline(always)]
fn top_material(blocks: u64, summary: &[u8; 64], skip_water: bool) -> Material {
    // the 16 blocks of one y layer, block index = x | (y << 2) | (z << 4) //
    const LAYER: u64 = 0x000F_000F_000F_000F;

    let mut y = 3;
    while y > 0 && blocks & (LAYER << (4 * y)) == 0 {
        y -= 1;
    }

    let block = (blocks & (LAYER << (4 * y))).trailing_zeros() as usize;
    block_material(summary, block, skip_water)
}

/// summary material of one 2x2x2 block, the water skipping one is the high nibble
#[inline(always)]
fn block_material(summary: &[u8; 64], block: usize, skip_water: bool) -> Material {
    let nibble = (summary[block] >> (4 * skip_water as u8)) & 15;

    // SAFETY: the build only ever writes Material values in here
    unsafe { std::mem::transmute::<u8, Material>(nibble) }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Branch {
    /// 4x4x4 cells that have a child node (a branch, or a leaf at depth 3)
    /// cell index = x | (y << 2) | (z << 4)
    pub children: u64,

    /// 4x4x4 cells that are completely solid, no child is stored for them
    pub solid: u64,

    /// 4x4x4 cells that are completely water, no child is stored for them
    pub water: u64,

    /// index of the first child in the branch or leaf pool
    /// children are contiguous, offset = popcount(children & cells before)
    pub child_pointer: u32,

    /// unused, spells out the padding so the world cache can write branches raw
    pub spare: u32,
}

impl Branch {
    pub const EMPTY: Self = Self {
        children: 0,
        solid: 0,
        water: 0,
        child_pointer: 0,
        spare: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;
    use crate::world::terrain::Terrain;

    fn primary(tree: &Tree64, packet: Packet) -> PacketHit {
        tree.trace::<false, false>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, 0.0)
    }

    #[test]
    fn test_trace_packet() {
        let tree = Tree64::build(1337);
        let ray_orig = Vec3::new(128.0, 160.0, -20.0);
        let ray_dir = (Vec3::new(128.0, 60.0, 128.0) - ray_orig).normalize();
        let packet = Packet::from_vecs([ray_orig; 4], [ray_dir; 4]);
        let result = primary(&tree, packet);
        println!("Materials: {:?}", result.materials);
        println!("Distances: {:?}", result.distances.to_array());

        assert_eq!(result.hit.to_array(), [u32::MAX; 4]);

        // benchmark ray traversal speed
        let iterations = 25000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(primary(&tree, std::hint::black_box(packet)));
        }
        let elapsed = start.elapsed();
        let total_rays = iterations * 4;
        let mrps = (total_rays as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
        println!(
            "Benchmark: {} rays traversed in {:.2}ms ({:.2} Million Rays/sec per single CPU core)",
            total_rays,
            elapsed.as_secs_f64() * 1000.0,
            mrps
        );
    }

    #[test]
    fn test_trace_sky() {
        let tree = Tree64::build(1337);
        // Ray shooting horizontally high above terrain (y = 240, world bounds y = 0..256)
        let ray_orig = Vec3::new(128.0, 240.0, -10.0);
        let ray_dir = Vec3::new(0.0, 0.0, 1.0);
        let packet = Packet::from_vecs([ray_orig; 4], [ray_dir; 4]);
        let result = primary(&tree, packet);
        assert_eq!(result.hit.to_array(), [0; 4]);
    }

    /// what the reference walk found: material, distance, crossed axis
    type Reference = Option<(Material, f64, usize)>;

    /// scalar f64 amanatides & woo walk over the full 2048^3 grid, the ground truth
    fn reference_trace(tree: &Tree64, o: Vec3, d: Vec3, skip_water: bool) -> Reference {
        let o = [o[0] as f64, o[1] as f64, o[2] as f64];
        let d = [d[0] as f64, d[1] as f64, d[2] as f64];
        let width = Tree64::WIDTH as f64;
        let voxel = VOXEL_SIZE as f64;

        // volume entry, the axis of the last slab entered is the entry normal //

        let mut t0 = 0.0f64;
        let mut t1 = f64::INFINITY;
        let mut axis = 0;
        let mut nears = [f64::NEG_INFINITY; 3];

        for a in 0..3 {
            let inv = 1.0 / d[a];
            let (n, f) = ((0.0 - o[a]) * inv, (width - o[a]) * inv);
            let (n, f) = if n.is_nan() || f.is_nan() {
                if o[a] < 0.0 || o[a] > width {
                    return None;
                }
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                (n.min(f), n.max(f))
            };
            nears[a] = n;
            t0 = t0.max(n);
            t1 = t1.min(f);
        }

        if t1 < t0 {
            return None;
        }

        if nears[1] > nears[axis] {
            axis = 1;
        }
        if nears[2] > nears[axis] {
            axis = 2;
        }

        let mut cell = [0i64; 3];
        let mut step = [0i64; 3];

        for a in 0..3 {
            cell[a] = (((o[a] + d[a] * t0) / voxel).floor() as i64).clamp(0, 2047);
            step[a] = if d[a] >= 0.0 { 1 } else { -1 };
        }

        let boundary = |a: usize, c: i64| {
            let face = (c + (step[a] > 0) as i64) as f64 * voxel;
            let t = (face - o[a]) / d[a];
            if t.is_nan() { f64::INFINITY } else { t }
        };

        let mut t = t0;

        loop {
            if let Some(m) = tree.voxel_at(cell[0] as u32, cell[1] as u32, cell[2] as u32)
                && !(skip_water && m == Material::Water)
            {
                return Some((m, t, axis));
            }

            let tmax = [
                boundary(0, cell[0]),
                boundary(1, cell[1]),
                boundary(2, cell[2]),
            ];
            axis = if tmax[0] <= tmax[1] && tmax[0] <= tmax[2] {
                0
            } else if tmax[1] <= tmax[2] {
                1
            } else {
                2
            };

            t = tmax[axis];
            cell[axis] += step[axis];

            if cell[axis] < 0 || cell[axis] > 2047 {
                return None;
            }
        }
    }

    /// camera like, grazing, boundary aligned and outside rays
    fn test_rays() -> Vec<(Vec3, Vec3)> {
        // xorshift, deterministic without pulling in a crate //

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };

        let mut rays: Vec<(Vec3, Vec3)> = Vec::new();

        for _ in 0..6000 {
            let o = Vec3::new(rand() * 256.0, 60.0 + rand() * 140.0, rand() * 256.0);
            let d = Vec3::new(rand() - 0.5, -rand() * 0.6, rand() - 0.5).normalize();
            rays.push((o, d));
        }

        // grazing rays skimming the surface //

        for _ in 0..2000 {
            let o = Vec3::new(rand() * 256.0, 44.0 + rand() * 30.0, rand() * 256.0);
            let d = Vec3::new(rand() - 0.5, -rand() * 0.05, rand() - 0.5).normalize();
            rays.push((o, d));
        }

        // origins exactly on 64m / 16m / 4m / 1m / voxel boundaries, axis aligned too //

        let boundaries = [64.0, 128.0, 16.0, 48.0, 4.0, 1.0, 0.125, 0.0];
        let axis_dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            // diagonals are nudged off the lattice, an exact corner crossing ties
            // every axis and f32 / f64 rounding may pick different (valid) paths
            Vec3::new(1.0, -1.003, 0.0).normalize(),
            Vec3::new(1.0, -0.997, 1.002).normalize(),
            Vec3::new(-1.0, -1.001, -0.998).normalize(),
        ];

        for &bx in &boundaries {
            for &bz in &boundaries {
                for &dir in &axis_dirs {
                    for y in [48.0, 64.0, 72.0, 96.0] {
                        rays.push((Vec3::new(bx + 64.0, y, bz + 64.0), dir));
                    }
                }
            }
        }

        // secondary like rays, upwards from low down //

        for _ in 0..2000 {
            let o = Vec3::new(rand() * 256.0, 40.0 + rand() * 40.0, rand() * 256.0);
            let d = Vec3::new(rand() - 0.5, rand(), rand() - 0.5).normalize();
            rays.push((o, d));
        }

        // rays from outside the volume //

        for _ in 0..1000 {
            let o = Vec3::new(-20.0 + rand() * 10.0, 120.0 + rand() * 60.0, rand() * 256.0);
            let d = Vec3::new(1.0, -0.2 - rand() * 0.3, rand() - 0.5).normalize();
            rays.push((o, d));
        }

        rays
    }

    /// run every flavor of trace against the reference, returns mismatches
    fn compare<const ANY_HIT: bool, const SKIP_WATER: bool>(
        tree: &Tree64,
        rays: &[(Vec3, Vec3)],
    ) -> usize {
        let mut mismatches = 0;

        for chunk in rays.chunks(4) {
            let mut origins = [chunk[0].0; 4];
            let mut dirs = [chunk[0].1; 4];

            for (i, &(o, d)) in chunk.iter().enumerate() {
                origins[i] = o;
                dirs[i] = d;
            }

            let packet = Packet::from_vecs(origins, dirs);
            let result =
                tree.trace::<ANY_HIT, SKIP_WATER>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, 0.0);

            let hits = result.hit.to_array();
            let distances = result.distances.to_array();
            let normals = [
                result.normals.xs.to_array(),
                result.normals.ys.to_array(),
                result.normals.zs.to_array(),
            ];

            for i in 0..chunk.len() {
                let expected = reference_trace(tree, origins[i], dirs[i], SKIP_WATER);

                let ok = match expected {
                    None => hits[i] == 0,
                    Some(_) if ANY_HIT => hits[i] != 0,
                    Some((m, t, axis)) => {
                        let tolerance = 1e-3 + t * 1e-5;
                        let sign = if dirs[i][axis] >= 0.0 { -1.0 } else { 1.0 };

                        hits[i] != 0
                            && result.materials[i] == m
                            && (distances[i] as f64 - t).abs() <= tolerance
                            // rays starting inside a solid voxel have no crossed face
                            && (t == 0.0 || normals[axis][i] == sign)
                    }
                };

                if !ok {
                    mismatches += 1;
                    if mismatches <= 10 {
                        println!(
                            "o {:?} d {:?} expected {:?} got {} {:?} {} n [{} {} {}]",
                            origins[i],
                            dirs[i],
                            expected,
                            hits[i],
                            result.materials[i],
                            distances[i],
                            normals[0][i],
                            normals[1][i],
                            normals[2][i]
                        );
                    }
                }
            }
        }

        println!(
            "any_hit {} skip_water {}: {} rays, {} mismatches",
            ANY_HIT,
            SKIP_WATER,
            rays.len(),
            mismatches
        );
        mismatches
    }

    #[test]
    fn test_trace_matches_reference() {
        let tree = Tree64::build(1337);
        let rays = test_rays();

        let mismatches = compare::<false, false>(&tree, &rays)
            + compare::<false, true>(&tree, &rays)
            + compare::<true, true>(&tree, &rays)
            + compare::<true, false>(&tree, &rays);

        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_trace_lod() {
        let tree = Tree64::build(1337);
        let mut checked = 0;

        for chunk in test_rays().chunks(4).take(2000) {
            let origins = [0, 1, 2, 3].map(|i| chunk[i.min(chunk.len() - 1)].0);
            let dirs = [0, 1, 2, 3].map(|i| chunk[i.min(chunk.len() - 1)].1);
            let packet = Packet::from_vecs(origins, dirs);

            let exact = primary(&tree, packet);
            let wide = tree.trace::<false, false>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, 1.0);
            let shadow = tree.trace::<true, true>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, 1.0);
            let exact_shadow =
                tree.trace::<true, true>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, 0.0);

            let (e, w) = (exact.distances.to_array(), wide.distances.to_array());

            for lane in 0..4 {
                // lumping cells together only ever makes things solid sooner, never later //

                if exact.hit.to_array()[lane] != 0 {
                    assert!(wide.hit.to_array()[lane] != 0);
                    assert_ne!(
                        wide.materials[lane],
                        Material::Air,
                        "lod hits keep a material"
                    );
                    assert!(
                        w[lane] <= e[lane] + 1e-3,
                        "lod hit {} past exact {}",
                        w[lane],
                        e[lane]
                    );
                    checked += 1;
                }

                // and a shadow that's there without lod is still there with it //

                if exact_shadow.hit.to_array()[lane] != 0 {
                    assert!(shadow.hit.to_array()[lane] != 0);
                }
            }
        }

        assert!(checked > 1000);
    }

    #[test]
    fn test_trace_max_distance() {
        let tree = Tree64::build(1337);
        let mut checked = 0;

        for chunk in test_rays().chunks(4).take(1000) {
            let origins = [0, 1, 2, 3].map(|i| chunk[i.min(chunk.len() - 1)].0);
            let dirs = [0, 1, 2, 3].map(|i| chunk[i.min(chunk.len() - 1)].1);
            let packet = Packet::from_vecs(origins, dirs);
            let full = primary(&tree, packet);

            if full.hit.has_none_active() {
                continue;
            }

            // a limit just short of the hit retires the ray, just past keeps the hit //

            let short = full.distances - SimdF32::from_float(0.2);
            let long = full.distances + SimdF32::from_float(0.2);

            let cut = tree.trace::<false, false>(packet, full.hit, short, 0.0);
            let kept = tree.trace::<false, false>(packet, full.hit, long, 0.0);

            assert!((cut.hit & full.hit).has_none_active());
            assert_eq!(kept.hit.to_array(), full.hit.to_array());
            checked += 1;
        }

        assert!(checked > 100);
    }

    #[test]
    fn test_edits() {
        use crate::world::World;
        use crate::world::sun::Sun;

        let mut world = World {
            tree: Tree64::build(1337),
            sun: Sun::default(),
        };

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };

        let materials = [
            Material::Dirt,
            Material::Grass,
            Material::Stone,
            Material::Wood,
            Material::Leaves,
            Material::Snow,
            Material::Water,
        ];

        let mut brushes = Vec::new();

        for _ in 0..600 {
            let o = Vec3::new(rand() * 256.0, 200.0, rand() * 256.0);
            let d = Vec3::new(rand() - 0.5, -2.0, rand() - 0.5).normalize();

            if let Some(target) = world.raycast(o, d, 250.0) {
                let diameter = 1 + (rand() * 8.0) as u32 % 8;
                let place = rand() < 0.5;
                let material = if place {
                    materials[(rand() * 7.0) as usize % 7]
                } else {
                    Material::Air
                };

                brushes.push((
                    World::brush_center(&target, diameter, place),
                    diameter,
                    material,
                ));
            }
        }

        brushes.push((Vec3::new(1024.0, 16.0, 1024.0), 8, Material::Air));
        brushes.push((Vec3::new(800.0, 1840.0, 800.0), 4, Material::Wood));

        let mut rays = test_rays();

        for &(center, diameter, material) in &brushes {
            let carve = material == Material::Air;
            let lo = [0, 1, 2].map(|a| center[a].floor() as i32 - 5);

            let mut before = Vec::new();
            for z in lo[2]..lo[2] + 11 {
                for y in lo[1]..lo[1] + 11 {
                    for x in lo[0]..lo[0] + 11 {
                        before.push(([x, y, z], world.voxel([x, y, z])));
                    }
                }
            }

            world.edit(center, diameter, material, None);

            for (voxel, old) in before {
                let now = world.voxel(voxel);
                let old_material = old.unwrap_or(Material::Air);
                let open = old_material == Material::Air || old_material == Material::Water;
                let inside = World::in_brush(voxel, center, diameter)
                    && voxel
                        .iter()
                        .all(|&c| (0..Tree64::VOXELS_PER_AXIS).contains(&c));

                let ok = match (inside, carve, open) {
                    (true, true, false) => now.is_none() || now == Some(Material::Water),
                    (true, false, true) => now == Some(material),
                    _ => now == old,
                };

                assert!(
                    ok,
                    "{voxel:?} was {old:?} now {now:?}, brush {center:?} {diameter} {material:?}"
                );
            }

            // rays straight at the edit, from above, sideways and from its center. //
            // an exact tie at the center can go either (valid) way

            let m = center * VOXEL_SIZE;
            let dirs = [
                Vec3::new(0.0113, -1.0, 0.0217),
                Vec3::new(1.0, -0.0531, 0.0347),
                Vec3::new(0.3119, 0.2063, 1.0),
            ];

            rays.push((m + Vec3::new(0.03, 20.0, 0.07), dirs[0].normalize()));
            rays.push((m + Vec3::new(-3.0, 0.02, 0.01), dirs[1].normalize()));
            rays.push((m, dirs[2].normalize()));
        }

        let packet = Packet::from_vecs(
            [Vec3::new(100.06, 250.0, 100.06); 4],
            [Vec3::new(0.0, -1.0, 0.0); 4],
        );
        let hit = primary(&world.tree, packet);
        assert_eq!(hit.materials[0], Material::Wood);
        assert!((hit.distances.to_array()[0] - 19.75).abs() < 1e-3);

        let mismatches = compare::<false, false>(&world.tree, &rays)
            + compare::<false, true>(&world.tree, &rays)
            + compare::<true, true>(&world.tree, &rays)
            + compare::<true, false>(&world.tree, &rays);

        assert_eq!(mismatches, 0);

        // the skip map is never bigger than a fresh one, too small is fine //

        let fresh = world.tree.build_skip();
        let over = (0..fresh.len())
            .filter(|&i| (0..2).any(|k| world.tree.skip[i][k] > fresh[i][k]))
            .count();

        println!(
            "{} edits, {} leaves",
            brushes.len(),
            world.tree.leaves.len()
        );
        assert_eq!(over, 0);
    }

    #[test]
    fn test_terrain_water_and_trees() {
        let terrain = Terrain::generate(1337);
        const N: usize = Tree64::VOXELS_PER_AXIS as usize;

        // water stands at most one voxel above the top of each neighbor, no walls / spikes //

        let top = |c: usize| terrain.ground[c].max(terrain.water[c]);
        let mut wet = 0;

        for z in 0..N {
            for x in 0..N {
                let c = z * N + x;
                if terrain.water[c] <= terrain.ground[c] {
                    continue;
                }
                wet += 1;

                for (nx, nz) in [
                    (x.wrapping_sub(1), z),
                    (x + 1, z),
                    (x, z.wrapping_sub(1)),
                    (x, z + 1),
                ] {
                    if nx < N && nz < N {
                        assert!(
                            terrain.water[c] <= top(nz * N + nx) + 1,
                            "water wall at {x} {z}: {} vs {}",
                            terrain.water[c],
                            top(nz * N + nx)
                        );
                    }
                }
            }
        }

        // lakes stay full: flat and watertight, every neighbor is the same lake or a higher bank //

        for &c in &terrain.lakes {
            let c = c as usize;
            let (x, z) = (c % N, c / N);
            assert!(
                terrain.water[c] > terrain.ground[c],
                "drained lake at {x} {z}"
            );

            for n in [c.wrapping_sub(1), c + 1, c.wrapping_sub(N), c + N] {
                if n < N * N {
                    assert!(
                        terrain.water[n] == terrain.water[c]
                            || terrain.ground[n].max(terrain.water[n]) >= terrain.water[c],
                        "lake leaks at {x} {z}"
                    );
                }
            }
        }

        // trees grow from grass, snow only high up //

        for tree in &terrain.trees {
            let c = tree.z as usize * N + tree.x as usize;
            assert_eq!(terrain.surface[c], Material::Grass);
        }

        let mut snow = 0;
        for c in 0..N * N {
            if terrain.surface[c] == Material::Snow {
                snow += 1;
                assert!(terrain.ground[c] as f32 * VOXEL_SIZE > 79.0);
            }
        }

        let sea = (Terrain::SEA_LEVEL * VOXELS_PER_METER) as u16;
        let ocean = (0..N * N)
            .filter(|&c| terrain.water[c] > terrain.ground[c] && terrain.water[c] <= sea + 1)
            .count();
        let max_ground = terrain.ground.iter().max().unwrap();

        println!(
            "wet columns {:.1}% (sea {:.1}%), snow {:.1}%, trees {}, highest ground {:.1}m",
            100.0 * wet as f64 / (N * N) as f64,
            100.0 * ocean as f64 / (N * N) as f64,
            100.0 * snow as f64 / (N * N) as f64,
            terrain.trees.len(),
            *max_ground as f32 * VOXEL_SIZE
        );

        assert!(wet > ocean, "rivers / lakes above the sea should exist");
        assert!(!terrain.trees.is_empty());
    }

    #[test]
    fn test_render_one_frame() {
        let (width, height) = (640, 352);
        let mut engine = crate::engine::Engine::new(width, height, World::new());
        let mut buffer = vec![0u32; width * height];

        engine.frame(buffer.as_mut_ptr() as usize);

        // a moving camera resets the accumulation every frame //

        let frames = 10;
        let start = std::time::Instant::now();
        for _ in 0..frames {
            engine.frame_accumulation_count = 0;
            engine.frame(buffer.as_mut_ptr() as usize);
        }
        let frame_ms = start.elapsed().as_secs_f64() * 1000.0 / frames as f64;

        // top left pixel looks above the terrain: light blue sky //

        let [r, g, b, _] = buffer[0].to_le_bytes();
        println!(
            "Frame {}x{}: {:.2}ms / frame, sky pixel rgb {} {} {}",
            width, height, frame_ms, r, g, b
        );
        assert!(b > 180 && b > r && g > r, "sky should be light blue");
    }

    /// dev only: writes a top down map and accumulated views as ppm files
    /// PREVIEW_DIR=/some/dir cargo test --release render_preview -- --ignored
    #[test]
    #[ignore]
    fn render_preview() {
        use std::io::Write;

        let dir = std::env::var("PREVIEW_DIR").unwrap_or_else(|_| ".".into());
        let write_ppm = |name: &str, w: usize, h: usize, rgb: &[u8]| {
            let mut f = std::fs::File::create(format!("{dir}/{name}.ppm")).unwrap();
            writeln!(f, "P6 {w} {h} 255").unwrap();
            f.write_all(rgb).unwrap();
        };

        // top down map, 1 pixel per 4 columns //

        let terrain = Terrain::generate(42);
        const N: usize = Tree64::VOXELS_PER_AXIS as usize;
        let size = N / 4;
        let mut map = vec![0u8; size * size * 3];

        for z in 0..size {
            for x in 0..size {
                let c = (z * 4) * N + x * 4;
                let g = terrain.ground[c] as f32 / 1600.0;
                let color = if terrain.water[c] > terrain.ground[c] {
                    [20.0, 70.0 + 80.0 * g, 160.0]
                } else {
                    let a = Material::ALBEDO[terrain.surface[c] as usize];
                    [
                        a[0] * 600.0 * (0.5 + g),
                        a[1] * 600.0 * (0.5 + g),
                        a[2] * 600.0 * (0.5 + g),
                    ]
                };
                map[(z * size + x) * 3..][..3].copy_from_slice(&color.map(|v| v.min(255.0) as u8));
            }
        }

        for tree in &terrain.trees {
            let (x, z) = (tree.x as usize / 4, tree.z as usize / 4);
            map[(z * size + x) * 3..][..3].copy_from_slice(&[255, 0, 0]);
        }

        write_ppm("map", size, size, &map);

        // accumulated views //

        let (width, height) = (960, 544);
        let mut engine = crate::engine::Engine::new(width, height, World::new());

        // whole frames, every one of the 64 renders is a full sample //
        engine.fps_target = 2;

        let views = [
            (
                Vec3::new(128.0, 160.0, -20.0),
                Vec3::new(128.0, 90.0, 128.0),
            ),
            (Vec3::new(40.0, 90.0, 40.0), Vec3::new(140.0, 70.0, 140.0)),
            (
                Vec3::new(250.0, 120.0, 250.0),
                Vec3::new(150.0, 60.0, 150.0),
            ),
            (Vec3::new(95.0, 92.0, 96.0), Vec3::new(88.0, 62.0, 150.0)),
        ];

        for (index, (position, target)) in views.into_iter().enumerate() {
            engine.camera.center = position;
            engine.camera.look_at(target);
            engine.frame_accumulation_count = 0;

            let mut buffer = vec![0u32; width * height];
            let start = std::time::Instant::now();
            for _ in 0..64 {
                engine.render(buffer.as_mut_ptr() as usize);
            }
            println!(
                "view {index}: 64 samples in {:.0}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );

            let rgb: Vec<u8> = buffer
                .iter()
                .flat_map(|p| {
                    let b = p.to_le_bytes();
                    [b[0], b[1], b[2]]
                })
                .collect();
            write_ppm(&format!("view{index}"), width, height, &rgb);
        }
    }

    /// dev only: moving frame times (ms) at 720p, the perf gate
    /// cargo test --release bench_views -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_views() {
        let (width, height) = (1280, 720);
        let mut engine = crate::engine::Engine::new(width, height, World::new());
        let mut buffer = vec![0u32; width * height];

        engine.fps_target = 2;

        let views = [
            (
                Vec3::new(128.0, 160.0, -20.0),
                Vec3::new(128.0, 90.0, 128.0),
            ),
            (Vec3::new(40.0, 90.0, 40.0), Vec3::new(140.0, 70.0, 140.0)),
            (
                Vec3::new(250.0, 120.0, 250.0),
                Vec3::new(150.0, 60.0, 150.0),
            ),
            (Vec3::new(95.0, 92.0, 96.0), Vec3::new(88.0, 62.0, 150.0)),
        ];

        for (index, (position, target)) in views.into_iter().enumerate() {
            engine.camera.center = position;
            engine.camera.look_at(target);

            let mut times: Vec<f64> = (0..45)
                .map(|_| {
                    engine.frame_accumulation_count = 0;
                    let start = std::time::Instant::now();
                    engine.render(buffer.as_mut_ptr() as usize);
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .skip(5)
                .collect();
            times.sort_by(f64::total_cmp);

            // the min is the steady one, a laptop's median wanders with its temperature //
            println!(
                "view {index}: min {:.2}ms  median {:.2}ms  p90 {:.2}ms",
                times[0],
                times[times.len() / 2],
                times[times.len() * 9 / 10]
            );
        }
    }
}
