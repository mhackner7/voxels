use crate::math::vec3::Vec3;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::world::leaf::{Brick, Leaf};
use crate::world::terrain::Terrain;
use crate::world::tree64::{Branch, Tree64};
use crate::world::voxel::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// what a cell of the tree turned into after generation
enum Cell {
    /// no voxels, nothing stored
    Empty,
    /// every voxel is SOLID_MATERIAL, stored as a bit in the parent
    Solid,
    /// every voxel is water, stored as a bit in the parent
    Water,
    /// mixed contents, stored as a child node
    Mixed(Branch),
}

impl Tree64 {
    pub fn build(seed: u64) -> Self {
        let start_time = std::time::Instant::now();

        println!("------------------------------------------------------------");
        println!(
            "[Tree64] Building 64-tree: {}m volume, {}^3 voxels of {:.3}m, depth {} + 8^3 leaves",
            Self::WIDTH,
            Self::VOXELS_PER_AXIS,
            VOXEL_SIZE,
            Self::LEAF_DEPTH
        );

        let terrain = Terrain::generate(seed);
        let terrain_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // the highest occupied voxel bounds every ray, the sky above it is never traversed //

        let top = terrain.top_max.get(11, 0, 0) as f32 * VOXEL_SIZE;

        let mut tree = Self {
            branches: vec![Branch::EMPTY],
            leaves: Vec::new(),
            bricks: Vec::new(),
            leaf_blocks: Vec::new(),
            leaf_block_materials: Vec::new(),
            skip: Vec::new(),
            water_leaves: Vec::new(),
            bmax: Packed_Vec3_x4::from_single(Vec3::new(
                Self::WIDTH,
                top.min(Self::WIDTH),
                Self::WIDTH,
            )),
        };

        let mut scratch = Vec::new();

        // the root is reserved at index 0, children are pushed behind it //

        tree.branches[0] = match tree.build_branch(&terrain, &mut scratch, 0, [0, 0, 0]) {
            Cell::Empty => Branch::EMPTY,
            Cell::Solid => Branch {
                solid: u64::MAX,
                ..Branch::EMPTY
            },
            Cell::Water => Branch {
                water: u64::MAX,
                ..Branch::EMPTY
            },
            Cell::Mixed(branch) => branch,
        };

        let skip_start = std::time::Instant::now();
        tree.skip = tree.build_skip();
        let skip_ms = skip_start.elapsed().as_secs_f64() * 1000.0;

        let branch_ram = tree.branches.len() * std::mem::size_of::<Branch>();
        let leaf_ram = tree.leaves.len() * std::mem::size_of::<Leaf>();
        let brick_ram = tree.bricks.len() * std::mem::size_of::<Brick>();
        const MB: f64 = 1024.0 * 1024.0;

        println!(
            "[Tree64] Terrain : {} trees, {} lake columns, top {:.1}m, generated in {:.2}ms",
            terrain.trees.len(),
            terrain.lakes.len(),
            top,
            terrain_ms
        );
        println!(
            "[Tree64] Branches: {:>8} ({:>6.2} MB @ {} B)",
            tree.branches.len(),
            branch_ram as f64 / MB,
            std::mem::size_of::<Branch>()
        );
        println!(
            "[Tree64] Leaves  : {:>8} ({:>6.2} MB @ {} B) + bricks ({:.2} MB @ {} B)",
            tree.leaves.len(),
            leaf_ram as f64 / MB,
            std::mem::size_of::<Leaf>(),
            brick_ram as f64 / MB,
            std::mem::size_of::<Brick>()
        );
        println!(
            "[Tree64] Skip maps: {:.2} MB, built in {:.2}ms",
            (tree.skip.len() * 2) as f64 / MB,
            skip_ms
        );
        println!(
            "[Tree64] Total {:.2} MB, built in {:.2}ms",
            (branch_ram + leaf_ram + brick_ram) as f64 / MB,
            start_time.elapsed().as_secs_f64() * 1000.0
        );
        println!("------------------------------------------------------------");

        tree
    }

    /// classify a cube of 2^level voxels from the terrain pyramids, None = mixed
    fn classify(terrain: &Terrain, min: [u32; 3], level: u32) -> Option<Cell> {
        let (x, y, z) = (min[0], min[1], min[2]);
        let end = y + (1 << level);

        if y >= terrain.top_max.get(level, x, z) as u32 {
            return Some(Cell::Empty);
        }

        if end <= terrain.stone_min.get(level, x, z) as u32 {
            return Some(Cell::Solid);
        }

        if y >= terrain.ground_max.get(level, x, z) as u32
            && end <= terrain.water_min.get(level, x, z) as u32
        {
            return Some(Cell::Water);
        }

        None
    }

    /// min is in voxels, the node spans 4 cells of (1 << CELL_SHIFT[depth]) voxels
    fn build_branch(
        &mut self,
        terrain: &Terrain,
        scratch: &mut Vec<u32>,
        depth: usize,
        min: [u32; 3],
    ) -> Cell {
        let shift = Self::CELL_SHIFT[depth];

        if let Some(cell) = Self::classify(terrain, min, shift + 2) {
            return cell;
        }

        let child_min = |cell: u32| {
            [
                min[0] + ((cell & 3) << shift),
                min[1] + (((cell >> 2) & 3) << shift),
                min[2] + (((cell >> 4) & 3) << shift),
            ]
        };

        let mut branch = Branch::EMPTY;

        // leaf branches push their leaves directly, nothing else is pushed in between //

        if depth == Self::LEAF_DEPTH - 1 {
            branch.child_pointer = self.leaves.len() as u32;

            for cell in 0..64 {
                let leaf_min = child_min(cell);

                let kind = match Self::classify(terrain, leaf_min, shift) {
                    Some(kind) => kind,
                    None => self.generate_leaf(terrain, scratch, leaf_min),
                };

                match kind {
                    Cell::Empty => {}
                    Cell::Solid => branch.solid |= 1 << cell,
                    Cell::Water => branch.water |= 1 << cell,
                    Cell::Mixed(_) => branch.children |= 1 << cell,
                }
            }

            return Self::finish(branch);
        }

        // regular branches build every child first,
        // then push them contiguously for popcount lookup

        let mut nodes = [Branch::EMPTY; 64];
        let mut count = 0;

        for cell in 0..64 {
            match self.build_branch(terrain, scratch, depth + 1, child_min(cell)) {
                Cell::Empty => {}
                Cell::Solid => branch.solid |= 1 << cell,
                Cell::Water => branch.water |= 1 << cell,
                Cell::Mixed(child) => {
                    nodes[count] = child;
                    branch.children |= 1 << cell;
                    count += 1;
                }
            }
        }

        branch.child_pointer = self.branches.len() as u32;
        self.branches.extend_from_slice(&nodes[..count]);

        Self::finish(branch)
    }

    /*
        SKIP MAP

        1. which meters have anything in them, asked straight from the tree,
           once with water counting and once without
        2. chebyshev distance to the nearest of them, one axis at a time:
           distance along x, then the max with y, then with z. that's exact for
           chebyshev and every pass only looks SKIP_MAX meters each way
    */

    fn build_skip(&self) -> Vec<[u8; 2]> {
        let wet = Self::chebyshev(|meter| self.meter_occupied(meter, false));
        let dry = Self::chebyshev(|meter| self.meter_occupied(meter, true));

        wet.into_iter().zip(dry).map(|(w, d)| [w, d]).collect()
    }

    /// chebyshev distance (meters, capped) from every meter to the nearest occupied one
    fn chebyshev(occupied: impl Fn([usize; 3]) -> bool + Sync) -> Vec<u8> {
        const M: usize = Tree64::METERS_PER_AXIS;
        let reach = Self::SKIP_MAX as usize;

        let mut skip: Vec<u8> = (0..M * M * M)
            .into_par_iter()
            .map(|i| {
                if occupied([i % M, (i / M) % M, i / (M * M)]) {
                    0
                } else {
                    Self::SKIP_MAX
                }
            })
            .collect();

        for stride in [1, M, M * M] {
            skip = (0..M * M * M)
                .into_par_iter()
                .map(|i| {
                    // position along this axis, so the window never wraps into the next row //
                    let along = (i / stride) % M;
                    let mut nearest = Self::SKIP_MAX;

                    for other in along.saturating_sub(reach)..(along + reach + 1).min(M) {
                        let gap = other.abs_diff(along) as u8;
                        let there = skip[i - along * stride + other * stride];
                        nearest = nearest.min(gap.max(there));
                    }

                    nearest
                })
                .collect();
        }

        skip
    }

    /// does the meter at [x, y, z] (meters) have any voxel in it,
    /// dry = water doesn't count
    fn meter_occupied(&self, meter: [usize; 3], dry: bool) -> bool {
        let [x, y, z] = meter.map(|m| (m << 3) as u32);
        let water = if dry { 0 } else { u64::MAX };
        let mut node = 0usize;

        for depth in 0..Self::LEAF_DEPTH {
            let shift = Self::CELL_SHIFT[depth];
            let cell = ((x >> shift) & 3) | (((y >> shift) & 3) << 2) | (((z >> shift) & 3) << 4);
            let bit = 1u64 << cell;
            let branch = &self.branches[node];

            if (branch.solid | (branch.water & water)) & bit != 0 {
                return true;
            }

            if branch.children & bit == 0 {
                return false;
            }

            node = (branch.child_pointer + (branch.children & (bit - 1)).count_ones()) as usize;
        }

        // the deepest branch's children are the meters themselves, a leaf of only water is dry //
        !dry || self.leaf_blocks[node][1] != 0
    }

    /// collapse fully empty / solid / water branches into their parent
    #[inline(always)]
    fn finish(branch: Branch) -> Cell {
        if branch.solid == u64::MAX {
            return Cell::Solid;
        }

        if branch.water == u64::MAX {
            return Cell::Water;
        }

        if branch.children | branch.solid | branch.water == 0 {
            return Cell::Empty;
        }

        Cell::Mixed(branch)
    }

    /// fill one 8x8x8 leaf at min (voxels), pushes it only if it is mixed
    /// returns Mixed(EMPTY) as the marker for a pushed leaf
    fn generate_leaf(&mut self, terrain: &Terrain, scratch: &mut Vec<u32>, min: [u32; 3]) -> Cell {
        terrain.trees_near(min, scratch);

        let mut leaf = Leaf { voxels: [0u64; 8] };
        let mut blocks = [0u64; 2];

        // the highest voxel of every block (and highest non water one) is how it looks from afar //
        let mut tops = [(-1i32, Material::Air); 64];
        let mut dry_tops = [(-1i32, Material::Air); 64];
        let mut brick = Brick::EMPTY;
        let mut stone = 0;
        let mut water = 0;

        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let material = terrain.voxel(min[0] + x, min[1] + y, min[2] + z, scratch);

                    if material == Material::Air {
                        continue;
                    }

                    let index = (x | (y << 3) | (z << 6)) as usize;
                    brick.set(index, material);
                    leaf.voxels[index >> 6] |= 1u64 << (index & 63);
                    let block_index = ((x >> 1) | ((y >> 1) << 2) | ((z >> 1) << 4)) as usize;
                    let block = 1u64 << block_index;
                    blocks[0] |= block;

                    if y as i32 >= tops[block_index].0 {
                        tops[block_index] = (y as i32, material);
                    }

                    if material != Material::Water {
                        blocks[1] |= block;

                        if y as i32 >= dry_tops[block_index].0 {
                            dry_tops[block_index] = (y as i32, material);
                        }
                    }

                    stone += (material == Material::Stone) as usize;
                    water += (material == Material::Water) as usize;
                }
            }
        }

        if leaf.voxels == [0; 8] {
            return Cell::Empty;
        }

        // only uniform leaves collapse, everything else keeps its materials //

        if stone == Leaf::VOXEL_COUNT {
            return Cell::Solid;
        }

        if water == Leaf::VOXEL_COUNT {
            return Cell::Water;
        }

        let index = self.leaves.len();

        if self.water_leaves.len() <= index >> 6 {
            self.water_leaves.push(0);
        }

        if water != 0 {
            self.water_leaves[index >> 6] |= 1 << (index & 63);
        }

        self.leaves.push(leaf);
        self.leaf_blocks.push(blocks);
        self.leaf_block_materials.push(std::array::from_fn(|b| {
            tops[b].1 as u8 | ((dry_tops[b].1 as u8) << 4)
        }));
        self.bricks.push(brick);

        Cell::Mixed(Branch::EMPTY)
    }
}
