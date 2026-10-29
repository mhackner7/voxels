use crate::math::vec3::Vec3;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::world::World;
use crate::world::leaf::{Brick, Leaf};
use crate::world::tree64::{Branch, Tree64};
use crate::world::voxel::*;

// a new child copies its whole sibling family to the end of the pool, unless it's already there
// the skip map only ever shrinks, a distance that's too small is still correct

/// what the crosshair is on
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub voxel: [i32; 3],
    /// the empty voxel in front is voxel + normal, all zero when the ray started inside
    pub normal: [i32; 3],
    pub point: Vec3,
    pub material: Material,
}

impl World {
    /// biggest brush diameter in voxels (1m)
    pub const BRUSH_MAX: u32 = 8;

    /// first non water voxel along a ray within reach (meters), plain voxel dda
    pub fn raycast(&self, origin: Vec3, dir: Vec3, reach: f32) -> Option<Target> {
        let o = origin * VOXELS_PER_METER;
        let mut voxel = [0, 1, 2].map(|a| o[a].floor() as i32);
        let mut normal = [0; 3];

        // voxel space t, dir is normalized so t counts voxels traveled //

        let step = [0, 1, 2].map(|a| if dir[a] < 0.0 { -1 } else { 1 });
        let delta = [0, 1, 2].map(|a| (1.0 / dir[a]).abs());
        let mut next = [0, 1, 2].map(|a| {
            let edge = voxel[a] + (step[a] > 0) as i32;

            if dir[a] == 0.0 {
                f32::INFINITY
            } else {
                (edge as f32 - o[a]) / dir[a]
            }
        });

        let max_t = reach * VOXELS_PER_METER;
        let mut t = 0.0;

        while t <= max_t {
            if let Some(material) = self.voxel(voxel)
                && material != Material::Water
            {
                return Some(Target {
                    voxel,
                    normal,
                    point: o + dir * t,
                    material,
                });
            }

            let axis = if next[0] < next[1] && next[0] < next[2] {
                0
            } else if next[1] < next[2] {
                1
            } else {
                2
            };

            t = next[axis];
            next[axis] += delta[axis];
            voxel[axis] += step[axis];
            normal = [0; 3];
            normal[axis] = -step[axis];
        }

        None
    }

    /// brush center on the target (voxel space), placing sits on the face, breaking sinks in as deep
    pub fn brush_center(target: &Target, diameter: u32, place: bool) -> Vec3 {
        let half = diameter as f32 * 0.5;
        let side = if place { 1.0 } else { -1.0 };
        let mut center = Vec3::DEFAULT;

        for a in 0..3 {
            let n = target.normal[a];

            center[a] = if n != 0 {
                let face = (target.voxel[a] + (n > 0) as i32) as f32;
                face + n as f32 * side * half
            } else if diameter & 1 == 1 {
                // odd brushes center on a voxel //
                target.voxel[a] as f32 + 0.5
            } else {
                // even ones on whichever corner of the voxel the hit is closest to //
                target.point[a].round()
            };
        }

        center
    }

    /// a voxel counts by its center
    #[inline(always)]
    pub fn in_brush(voxel: [i32; 3], center: Vec3, diameter: u32) -> bool {
        let half = diameter as f32 * 0.5;
        let offset = Vec3::new(
            voxel[0] as f32 + 0.5,
            voxel[1] as f32 + 0.5,
            voxel[2] as f32 + 0.5,
        ) - center;

        offset.length_squared() <= half * half + 1e-3
    }

    // Air carves (the hole floods from water above or beside), anything else fills air and water
    // but never keep_out (the player). returns whether anything changed
    pub fn edit(
        &mut self,
        center: Vec3,
        diameter: u32,
        material: Material,
        keep_out: Option<(Vec3, Vec3)>,
    ) -> bool {
        let diameter = diameter.clamp(1, Self::BRUSH_MAX);
        let half = diameter as f32 * 0.5;
        let lo = [0, 1, 2].map(|a| (center[a] - half).floor() as i32);
        let hi = [0, 1, 2].map(|a| (center[a] + half).ceil() as i32);
        let carve = material == Material::Air;

        let keep_out = keep_out.map(|(lo, hi)| (lo * VOXELS_PER_METER, hi * VOXELS_PER_METER));

        let mut changed =
            [[0i32; 3]; (Self::BRUSH_MAX * Self::BRUSH_MAX * Self::BRUSH_MAX) as usize];
        let mut count = 0;
        let mut pending = None;

        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let voxel = [x, y, z];

                    if !Self::in_brush(voxel, center, diameter) {
                        continue;
                    }

                    if (x | y | z) & !(Tree64::VOXELS_PER_AXIS - 1) != 0 {
                        continue;
                    }

                    // carving takes anything solid, filling takes air and water //

                    let old = self.voxel(voxel).unwrap_or(Material::Air);
                    let open = old == Material::Air || old == Material::Water;

                    if carve == open || old == material {
                        continue;
                    }

                    if let Some((box_lo, box_hi)) = keep_out
                        && (0..3).all(|a| {
                            (voxel[a] as f32) < box_hi[a] && (voxel[a] + 1) as f32 > box_lo[a]
                        })
                    {
                        continue;
                    }

                    self.set(voxel, material, &mut pending);
                    changed[count] = voxel;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return false;
        }

        // water pours into the hole: from above, or sideways at the same height //

        if carve {
            let mut flooded = true;

            while flooded {
                flooded = false;

                for &[x, y, z] in &changed[..count] {
                    if self.voxel([x, y, z]).is_some() {
                        continue;
                    }

                    let wet = [
                        [x, y + 1, z],
                        [x - 1, y, z],
                        [x + 1, y, z],
                        [x, y, z - 1],
                        [x, y, z + 1],
                    ]
                    .into_iter()
                    .any(|n| self.voxel(n) == Some(Material::Water));

                    if wet {
                        self.set([x, y, z], Material::Water, &mut pending);
                        flooded = true;
                    }
                }
            }
        }

        if let Some(leaf) = pending {
            self.tree.refresh_leaf(leaf);
        }

        if !carve {
            let (mut lo, mut hi) = ([usize::MAX; 3], [0; 3]);

            for voxel in &changed[..count] {
                for a in 0..3 {
                    lo[a] = lo[a].min((voxel[a] >> 3) as usize);
                    hi[a] = hi[a].max((voxel[a] >> 3) as usize);
                }
            }

            self.tree.shrink_skip(lo, hi, material != Material::Water);
            self.tree.raise_top((hi[1] as u32 + 1) << 3);
        }

        true
    }

    fn set(&mut self, [x, y, z]: [i32; 3], material: Material, pending: &mut Option<usize>) {
        let leaf = self
            .tree
            .set_voxel([x as u32, y as u32, z as u32], material);

        if *pending != Some(leaf)
            && let Some(last) = pending.replace(leaf)
        {
            self.tree.refresh_leaf(last);
        }
    }
}

impl Tree64 {
    /// set one voxel, Air clears it. returns its leaf, which still needs a refresh_leaf
    pub fn set_voxel(&mut self, [x, y, z]: [u32; 3], material: Material) -> usize {
        let leaf = self.open_leaf([x, y, z]);
        let voxel = ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6)) as usize;
        let bit = 1u64 << (voxel & 63);

        if material == Material::Air {
            self.leaves[leaf].voxels[voxel >> 6] &= !bit;
        } else {
            self.leaves[leaf].voxels[voxel >> 6] |= bit;
        }

        self.bricks[leaf].set(voxel, material);
        leaf
    }

    /// leaf holding a voxel, opening up cells on the way down
    fn open_leaf(&mut self, [x, y, z]: [u32; 3]) -> usize {
        let mut node = 0usize;

        for depth in 0..Self::LEAF_DEPTH {
            let shift = Self::CELL_SHIFT[depth];
            let cell = ((x >> shift) & 3) | (((y >> shift) & 3) << 2) | (((z >> shift) & 3) << 4);
            let bit = 1u64 << cell;

            if self.branches[node].children & bit == 0 {
                self.adopt(node, depth, bit);
            }

            let branch = &self.branches[node];
            node = (branch.child_pointer + (branch.children & (bit - 1)).count_ones()) as usize;
        }

        node
    }

    /// give a branch a child at the cell bit, just as full as the cell was
    fn adopt(&mut self, node: usize, depth: usize, bit: u64) {
        let mut branch = self.branches[node];

        let fill = if branch.solid & bit != 0 {
            SOLID_MATERIAL
        } else if branch.water & bit != 0 {
            Material::Water
        } else {
            Material::Air
        };

        let start = branch.child_pointer as usize;
        let count = branch.children.count_ones() as usize;
        let rank = (branch.children & (bit - 1)).count_ones() as usize;

        branch.child_pointer = if depth == Self::LEAF_DEPTH - 1 {
            self.adopt_leaf(start, count, rank, fill)
        } else {
            let child = Branch {
                solid: if fill == SOLID_MATERIAL { u64::MAX } else { 0 },
                water: if fill == Material::Water { u64::MAX } else { 0 },
                ..Branch::EMPTY
            };

            insert(&mut self.branches, start, count, rank, child)
        };

        branch.children |= bit;
        branch.solid &= !bit;
        branch.water &= !bit;
        self.branches[node] = branch;
    }

    /// adopt for a family of leaves (4 pools at once), returns where it lives now
    fn adopt_leaf(&mut self, start: usize, count: usize, rank: usize, fill: Material) -> u32 {
        let full = if fill == Material::Air { 0 } else { u64::MAX };
        let leaf = Leaf { voxels: [full; 8] };
        let brick = Brick {
            grid: [fill as u8 * 0x11; Leaf::VOXEL_COUNT / 2],
        };

        let moved = insert(&mut self.leaves, start, count, rank, leaf);
        insert(&mut self.bricks, start, count, rank, brick);
        insert(&mut self.leaf_blocks, start, count, rank, [0; 2]);
        insert(&mut self.leaf_block_materials, start, count, rank, [0; 64]);

        self.water_leaves.resize(self.leaves.len().div_ceil(64), 0);

        for leaf in moved as usize..=moved as usize + count {
            self.refresh_leaf(leaf);
        }

        moved
    }

    /// meters lo..=hi got filled, shrink the skip map around them (dry = the water ignoring one too)
    pub fn shrink_skip(&mut self, lo: [usize; 3], hi: [usize; 3], dry: bool) {
        let reach = Self::SKIP_MAX as usize;
        let last = Self::METERS_PER_AXIS - 1;

        let around = |a: usize| lo[a].saturating_sub(reach)..=(hi[a] + reach).min(last);

        for z in around(2) {
            for y in around(1) {
                for x in around(0) {
                    let meter = [x, y, z];

                    let gap = (0..3)
                        .map(|a| {
                            lo[a]
                                .saturating_sub(meter[a])
                                .max(meter[a].saturating_sub(hi[a]))
                        })
                        .max()
                        .unwrap_or(0) as u8;

                    let skip = &mut self.skip[Self::skip_index(meter)];
                    skip[0] = skip[0].min(gap);

                    if dry {
                        skip[1] = skip[1].min(gap);
                    }
                }
            }
        }
    }

    pub fn raise_top(&mut self, top: u32) {
        let top = (top as f32 * VOXEL_SIZE).min(Self::WIDTH);

        if top > self.bmax.ys.to_array()[0] {
            self.bmax = Packed_Vec3_x4::from_single(Vec3::new(Self::WIDTH, top, Self::WIDTH));
        }
    }
}

/// slot item into the family at rank, returns its new start.
/// a family at the end of the pool grows in place, anything else moves to the end first
fn insert<T: Copy>(pool: &mut Vec<T>, start: usize, count: usize, rank: usize, item: T) -> u32 {
    let start = if start + count == pool.len() {
        start
    } else {
        pool.extend_from_within(start..start + count);
        pool.len() - count
    };

    pool.insert(start + rank, item);
    start as u32
}
