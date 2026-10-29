use crate::world::tree64::Tree64;
use crate::world::voxel::*;
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/*
    BUILD TIME ONLY

    the terrain is generated as voxel column maps (2048 x 2048 columns):
    - ground  = y of the first air voxel above the ground
    - water   = y of the first non water voxel (no water when <= ground)
    - surface = top material of the column
    - soil    = depth of the non stone layers in voxels

    trees are stored as shapes and only rasterized when a leaf asks for them,
    min / max pyramids over the maps let the tree builder classify whole nodes
*/

/// columns along one axis of the world
const COLUMNS: usize = Tree64::VOXELS_PER_AXIS as usize;

/// samples per axis of the 0.5m noise grid (256m * 2 + far edge)
const GRID: usize = 513;

/// samples per axis of the 1m hydrology grid
const HYDRO: usize = 257;

/// grass / snow cap thickness in voxels
const TOP_LAYER: u16 = 2;

/// tree buckets are 8m (64 voxels) wide
const BUCKET_SHIFT: u32 = 6;
const BUCKETS: usize = COLUMNS >> BUCKET_SHIFT;

/// coarsest pyramid level is the whole world (2^11 columns)
const MIP_LEVELS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeKind {
    Oak,
    Spruce,
}

/// one tree, all coords in voxels
#[derive(Clone, Copy, Debug)]
pub struct TreeShape {
    pub kind: TreeKind,
    /// trunk center column
    pub x: i32,
    pub z: i32,
    /// first voxel above the ground at the trunk
    pub base: i32,
    /// trunk height above the base
    pub height: i32,
    /// widest canopy radius
    pub radius: i32,
    /// inclusive voxel bounds of the whole tree
    pub min: [i32; 3],
    pub max: [i32; 3],
}

/// min or max of a column map over aligned 2^level blocks
pub struct Mips {
    levels: Vec<Vec<u16>>,
}

impl Mips {
    fn build(columns: &[u16], take_max: bool) -> Self {
        let pick = |a: u16, b: u16| if take_max { a.max(b) } else { a.min(b) };

        let mut levels = vec![columns.to_vec()];

        for level in 1..MIP_LEVELS {
            let size = COLUMNS >> level;
            let prev = &levels[level - 1];
            let prev_size = size * 2;

            let mut next = vec![0u16; size * size];

            for z in 0..size {
                for x in 0..size {
                    let i = (z * 2) * prev_size + x * 2;
                    next[z * size + x] = pick(
                        pick(prev[i], prev[i + 1]),
                        pick(prev[i + prev_size], prev[i + prev_size + 1]),
                    );
                }
            }

            levels.push(next);
        }

        Self { levels }
    }

    /// value over the aligned block of 2^level columns that contains (x, z)
    #[inline(always)]
    pub fn get(&self, level: u32, x: u32, z: u32) -> u16 {
        let size = COLUMNS >> level;
        self.levels[level as usize][(z >> level) as usize * size + (x >> level) as usize]
    }
}

pub struct Terrain {
    pub ground: Vec<u16>,
    pub water: Vec<u16>,
    pub surface: Vec<Material>,
    pub soil: Vec<u16>,
    /// columns fill_lakes flooded, all flat and watertight at their spill height
    pub lakes: Vec<u32>,

    pub trees: Vec<TreeShape>,
    /// indices into trees for every 8m bucket a tree overlaps
    pub buckets: Vec<Vec<u32>>,

    /// max of ground, water and tree tops, anything at or above is air
    pub top_max: Mips,
    /// min of the stone top (ground - soil), anything below is stone
    pub stone_min: Mips,
    /// max ground, anything at or above it (and below water_min) is water
    pub ground_max: Mips,
    /// min water top
    pub water_min: Mips,
}

impl Terrain {
    /// sea surface in meters
    pub const SEA_LEVEL: f32 = 44.0;

    /// snow covers flat ground above this height (meters), only the highest tops
    const SNOW_LINE: f32 = 90.0;

    /// no trees grow above this height (meters)
    const TREE_LINE: f32 = 88.0;

    /// rise over run where the ground turns into bare stone
    const CLIFF_SLOPE: f32 = 1.1;

    /// water surface below the river banks (meters)
    const BANK_DROP: f32 = 0.2;

    /// flow accumulation (in rain weighted cells) where a stream starts
    const RIVER_THRESHOLD: f32 = 1400.0;

    /// a dip has to go this far below its spill height (meters) to count as a lake
    const LAKE_MIN_DEPTH: f32 = 0.4;

    /// steepest water surface drop (meters per meter) a stream is allowed,
    /// anything steeper would drain into 1 voxel stairs, so streams start below it
    const RIVER_MAX_SLOPE: f32 = 0.08;

    pub fn generate(seed: u64) -> Self {
        let seed = (seed ^ (seed >> 32)) as u32 as i32;

        /* HEIGHT FIELD */

        let (heights, detail) = Self::height_grid(seed);

        // ground in meters per column, bilinear on the 0.5m grid //

        let mut ground_m = vec![0.0f32; COLUMNS * COLUMNS];

        for cz in 0..COLUMNS {
            for cx in 0..COLUMNS {
                ground_m[cz * COLUMNS + cx] = Self::bilinear(&heights, cx, cz);
            }
        }

        let mut water_m = vec![0.0f32; COLUMNS * COLUMNS];
        let mut lakes_m = vec![0.0f32; COLUMNS * COLUMNS];

        /* WATER */

        Self::add_rivers_and_lakes(&heights, &mut ground_m, &mut water_m, &mut lakes_m);

        // quantize to voxels, voxel y is filled when its center is below the height //

        let quantize = |m: f32| ((m * VOXELS_PER_METER + 0.5).floor().max(0.0) as u16).min(2047);

        let mut ground: Vec<u16> = ground_m.iter().map(|&m| quantize(m).max(1)).collect();
        let mut water: Vec<u16> = water_m.iter().map(|&m| quantize(m)).collect();
        let lake_seeds: Vec<u32> = (0..COLUMNS * COLUMNS)
            .filter(|&c| quantize(lakes_m[c]) > ground[c])
            .map(|c| c as u32)
            .collect();

        for (w, &g) in water.iter_mut().zip(&ground) {
            if *w < g {
                *w = g;
            }
        }

        drop(ground_m);
        drop(water_m);
        drop(lakes_m);

        let lakes = Self::fill_lakes(&ground, &mut water, lake_seeds);

        Self::settle_water(&mut ground, &mut water);

        /* MATERIALS */

        let (surface, soil) = Self::paint(&ground, &water, &heights, &detail);

        /* TREES */

        let trees = Self::plant_trees(seed, &ground, &water, &surface, &detail);

        let mut buckets = vec![Vec::new(); BUCKETS * BUCKETS];
        let mut top: Vec<u16> = ground.iter().zip(&water).map(|(&g, &w)| g.max(w)).collect();

        for (index, tree) in trees.iter().enumerate() {
            let b0 = (tree.min[0].max(0) >> BUCKET_SHIFT) as usize;
            let b1 = ((tree.max[0].min(COLUMNS as i32 - 1)) >> BUCKET_SHIFT) as usize;
            let c0 = (tree.min[2].max(0) >> BUCKET_SHIFT) as usize;
            let c1 = ((tree.max[2].min(COLUMNS as i32 - 1)) >> BUCKET_SHIFT) as usize;

            for bz in c0..=c1 {
                for bx in b0..=b1 {
                    buckets[bz * BUCKETS + bx].push(index as u32);
                }
            }

            // conservative top, the whole bounding box counts as occupied //

            let tree_top = (tree.max[1] + 1).clamp(0, 2047) as u16;

            for z in tree.min[2].max(0)..=tree.max[2].min(COLUMNS as i32 - 1) {
                for x in tree.min[0].max(0)..=tree.max[0].min(COLUMNS as i32 - 1) {
                    let c = z as usize * COLUMNS + x as usize;
                    top[c] = top[c].max(tree_top);
                }
            }
        }

        /* PYRAMIDS */

        let stone: Vec<u16> = ground.iter().zip(&soil).map(|(&g, &s)| g - s).collect();

        let top_max = Mips::build(&top, true);
        let stone_min = Mips::build(&stone, false);
        let ground_max = Mips::build(&ground, true);
        let water_min = Mips::build(&water, false);

        Self {
            ground,
            water,
            surface,
            soil,
            lakes,
            trees,
            buckets,
            top_max,
            stone_min,
            ground_max,
            water_min,
        }
    }

    /// heights (meters) and a [-1, 1] detail noise on the 0.5m grid
    fn height_grid(seed: i32) -> (Vec<f32>, Vec<f32>) {
        let noise = |seed: i32, frequency: f32, octaves: i32, fractal: FractalType| {
            let mut noise = FastNoiseLite::with_seed(seed);
            noise.set_noise_type(Some(NoiseType::OpenSimplex2));
            noise.set_frequency(Some(frequency));
            noise.set_fractal_type(Some(fractal));
            noise.set_fractal_octaves(Some(octaves));
            noise.set_fractal_lacunarity(Some(2.0));
            noise.set_fractal_gain(Some(0.5));
            noise
        };

        let continent = noise(seed, 1.0 / 220.0, 3, FractalType::FBm);
        let swell = noise(seed.wrapping_add(1), 1.0 / 160.0, 2, FractalType::FBm);
        let hills = noise(seed.wrapping_add(2), 1.0 / 60.0, 4, FractalType::FBm);
        let plains = noise(seed.wrapping_add(3), 1.0 / 30.0, 3, FractalType::FBm);
        let fine = noise(seed.wrapping_add(4), 1.0 / 7.0, 2, FractalType::FBm);

        let mut heights = vec![0.0f32; GRID * GRID];
        let mut detail = vec![0.0f32; GRID * GRID];

        for gz in 0..GRID {
            for gx in 0..GRID {
                let x = gx as f32 * 0.5;
                let z = gz as f32 * 0.5;

                // radial falloff so the coast wraps the map and rivers reach the sea //

                let ex = x / 128.0 - 1.0;
                let ez = z / 128.0 - 1.0;
                let edge = (ex * ex + ez * ez).sqrt();

                let land =
                    continent.get_noise_2d(x, z) * 0.5 + 0.55 - smoothstep(0.95, 1.35, edge) * 1.4;

                let inland = smoothstep(0.0, 0.3, land);

                // one big soft swell makes the rounded mountains, hills roll a bit harder on top of it //

                let rise =
                    smoothstep(0.0, 0.8, swell.get_noise_2d(x, z)) * smoothstep(0.1, 0.5, land);
                let mountain = rise * rise * 30.0;
                let hill = (hills.get_noise_2d(x, z) * 0.5 + 0.5) * (4.0 + rise * 10.0);

                let height = Self::SEA_LEVEL
                    + land * 14.0
                    + plains.get_noise_2d(x, z) * 1.2
                    + (hill + mountain) * inland;

                heights[gz * GRID + gx] = height.clamp(2.0, 228.0);
                detail[gz * GRID + gx] = fine.get_noise_2d(x, z);
            }
        }

        Self::erode(&mut heights, &detail);

        (heights, detail)
    }

    /// thermal erosion: material slides down wherever the slope passes the talus angle
    /// turns noise needles into scree slopes and broad peaks
    fn erode(heights: &mut [f32], detail: &[f32]) {
        /// biggest height step allowed between 0.5m samples (rise over run ~0.76, 37 degrees)
        const TALUS: f32 = 0.38;

        for _ in 0..140 {
            for z in 1..GRID - 1 {
                for x in 1..GRID - 1 {
                    let c = z * GRID + x;

                    // rock strength varies, so slopes are not perfect cones //
                    let talus = TALUS * (1.0 + detail[c] * 0.45);

                    // steepest of the 4 neighbors //

                    let mut lowest = c;
                    for n in [c - 1, c + 1, c - GRID, c + GRID] {
                        if heights[n] < heights[lowest] {
                            lowest = n;
                        }
                    }

                    let drop = heights[c] - heights[lowest];

                    if drop > talus {
                        let moved = (drop - talus) * 0.5;
                        heights[c] -= moved;
                        heights[lowest] += moved;
                    }
                }
            }
        }
    }

    /// bilinear sample of a 0.5m grid at the center of a voxel column
    #[inline(always)]
    fn bilinear(grid: &[f32], cx: usize, cz: usize) -> f32 {
        // column centers in grid units (2 samples per meter)
        let gx = (cx as f32 + 0.5) * VOXEL_SIZE * 2.0;
        let gz = (cz as f32 + 0.5) * VOXEL_SIZE * 2.0;

        let x0 = (gx.floor() as usize).min(GRID - 2);
        let z0 = (gz.floor() as usize).min(GRID - 2);

        let tx = gx - x0 as f32;
        let tz = gz - z0 as f32;

        let h00 = grid[z0 * GRID + x0];
        let h10 = grid[z0 * GRID + x0 + 1];
        let h01 = grid[(z0 + 1) * GRID + x0];
        let h11 = grid[(z0 + 1) * GRID + x0 + 1];

        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;

        hx0 + (hx1 - hx0) * tz
    }

    /*
        LAKES + RIVERS

        1. priority flood the 1m grid from the sea and the world edges
           - without epsilon: depressions fill flat to their spill height (lakes)
           - with epsilon: every cell has a strictly lower neighbor (flow directions)
        2. d8 flow accumulation, rain weighted by height so mountains feed rivers
        3. rasterize every gentle stream segment into the column maps:
           levees first (so the channel is always contained), then the carved bed,
           the water surface falls linearly along each segment and is level across it
        4. lakes only mark their area, fill_lakes floods them at voxel resolution
    */

    fn add_rivers_and_lakes(
        heights: &[f32],
        ground: &mut [f32],
        water: &mut [f32],
        lakes: &mut [f32],
    ) {
        let h: Vec<f32> = (0..HYDRO * HYDRO)
            .map(|i| heights[(i / HYDRO) * 2 * GRID + (i % HYDRO) * 2])
            .collect();

        let lake_level = Self::priority_flood(&h, 0.0);
        let filled = Self::priority_flood(&h, 1e-3);

        // flow directions, seeds (sea / edges) drain nowhere //

        let mut downstream = vec![u32::MAX; HYDRO * HYDRO];

        for i in 0..HYDRO * HYDRO {
            let (x, z) = ((i % HYDRO) as i32, (i / HYDRO) as i32);

            if Self::is_outlet(&h, x, z) {
                continue;
            }

            let mut best = filled[i];

            for (nx, nz) in Self::neighbors8(x, z) {
                let n = nz as usize * HYDRO + nx as usize;
                if filled[n] < best {
                    best = filled[n];
                    downstream[i] = n as u32;
                }
            }
        }

        // flow accumulation, highest cells first //

        let mut order: Vec<u32> = (0..(HYDRO * HYDRO) as u32).collect();
        order.sort_unstable_by(|&a, &b| filled[b as usize].total_cmp(&filled[a as usize]));

        let mut flow: Vec<f32> = h
            .iter()
            .map(|&height| 1.0 + ((height - 60.0) / 25.0).max(0.0))
            .collect();

        for &i in &order {
            let d = downstream[i as usize];
            if d != u32::MAX {
                flow[d as usize] += flow[i as usize];
            }
        }

        // lakes, flat at their spill height, puddle sized dips just stay dry //

        let in_lake =
            |i: usize| lake_level[i] - h[i] > Self::LAKE_MIN_DEPTH && h[i] > Self::SEA_LEVEL;

        for (i, &level) in lake_level.iter().enumerate() {
            if !in_lake(i) {
                continue;
            }

            let (x, z) = ((i % HYDRO) as f32, (i / HYDRO) as f32);
            Self::rasterize_square(ground, lakes, x, z, level);
        }

        // streams, lowest cells first so a stream only shows up below a gentle path //
        // to the sea or a lake, which means every stream starts where its bed calms down

        let mut stream = vec![false; HYDRO * HYDRO];

        for &i in order.iter().rev() {
            let i = i as usize;
            let d = downstream[i];

            if d == u32::MAX || flow[i] < Self::RIVER_THRESHOLD || in_lake(i) {
                continue;
            }

            let d = d as usize;
            let run = if i % HYDRO != d % HYDRO && i / HYDRO != d / HYDRO {
                std::f32::consts::SQRT_2
            } else {
                1.0
            };

            let gentle = filled[i] - filled[d] <= Self::RIVER_MAX_SLOPE * run;
            let fed = downstream[d] == u32::MAX || in_lake(d) || stream[d];

            stream[i] = gentle && fed;
        }

        // lakes already hold the water of whatever streams run into them //

        let segments: Vec<(usize, usize)> = (0..HYDRO * HYDRO)
            .filter(|&i| stream[i] && !in_lake(downstream[i] as usize))
            .map(|i| (i, downstream[i] as usize))
            .collect();

        // rivers, two passes so a levee never fills another channel //

        let surface = |i: usize| (filled[i] - Self::BANK_DROP).max(Self::SEA_LEVEL);
        let radius =
            |i: usize| (0.5 + (flow[i] - Self::RIVER_THRESHOLD).max(0.0).sqrt() * 0.02).min(2.0);

        for carve in [false, true] {
            for &(a, b) in &segments {
                let pa = ((a % HYDRO) as f32, (a / HYDRO) as f32);
                let pb = ((b % HYDRO) as f32, (b / HYDRO) as f32);

                Self::rasterize_segment(
                    ground,
                    water,
                    carve,
                    [pa, pb],
                    [surface(a), surface(b).min(surface(a))],
                    [radius(a), radius(b).max(radius(a))],
                );
            }
        }

        // the sea //

        for (g, w) in ground.iter().zip(water.iter_mut()) {
            if *g < Self::SEA_LEVEL {
                *w = w.max(Self::SEA_LEVEL);
            }
        }
    }

    /// cells that drain out of the world: the edges and the sea
    #[inline(always)]
    fn is_outlet(h: &[f32], x: i32, z: i32) -> bool {
        x == 0
            || z == 0
            || x == HYDRO as i32 - 1
            || z == HYDRO as i32 - 1
            || h[z as usize * HYDRO + x as usize] <= Self::SEA_LEVEL
    }

    #[inline(always)]
    fn neighbors8(x: i32, z: i32) -> impl Iterator<Item = (i32, i32)> {
        const OFFSETS: [(i32, i32); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        OFFSETS
            .into_iter()
            .map(move |(dx, dz)| (x + dx, z + dz))
            .filter(|&(x, z)| x >= 0 && z >= 0 && x < HYDRO as i32 && z < HYDRO as i32)
    }

    /// barnes et al. 2014, heights must be positive (f32 bits sort like the values)
    fn priority_flood(h: &[f32], epsilon: f32) -> Vec<f32> {
        let mut filled = vec![f32::INFINITY; HYDRO * HYDRO];
        let mut open = BinaryHeap::new();

        for z in 0..HYDRO as i32 {
            for x in 0..HYDRO as i32 {
                if Self::is_outlet(h, x, z) {
                    let i = z as usize * HYDRO + x as usize;
                    filled[i] = h[i].max(Self::SEA_LEVEL);
                    open.push(Reverse((filled[i].to_bits(), i as u32)));
                }
            }
        }

        while let Some(Reverse((_, i))) = open.pop() {
            let i = i as usize;
            let (x, z) = ((i % HYDRO) as i32, (i / HYDRO) as i32);

            for (nx, nz) in Self::neighbors8(x, z) {
                let n = nz as usize * HYDRO + nx as usize;

                if filled[n] != f32::INFINITY {
                    continue;
                }

                filled[n] = h[n].max(filled[i] + epsilon);
                open.push(Reverse((filled[n].to_bits(), n as u32)));
            }
        }

        filled
    }

    /// mark the 1m square around a lake cell (meters) with its level
    fn rasterize_square(ground: &[f32], lakes: &mut [f32], x: f32, z: f32, level: f32) {
        let lo = |m: f32| ((m - 1.0) * VOXELS_PER_METER).max(0.0) as usize;
        let hi = |m: f32| (((m + 1.0) * VOXELS_PER_METER) as usize).min(COLUMNS - 1);

        for cz in lo(z)..=hi(z) {
            for cx in lo(x)..=hi(x) {
                let c = cz * COLUMNS + cx;
                if ground[c] < level {
                    lakes[c] = lakes[c].max(level);
                }
            }
        }
    }

    /// one river segment between two 1m cells (meters)
    fn rasterize_segment(
        ground: &mut [f32],
        water: &mut [f32],
        carve: bool,
        points: [(f32, f32); 2],
        surfaces: [f32; 2],
        radii: [f32; 2],
    ) {
        const LEVEE_WIDTH: f32 = 3.0;

        let [(ax, az), (bx, bz)] = points;
        let reach = radii[1] + LEVEE_WIDTH;

        let lo = |a: f32, b: f32| ((a.min(b) - reach) * VOXELS_PER_METER).max(0.0) as usize;
        let hi =
            |a: f32, b: f32| (((a.max(b) + reach) * VOXELS_PER_METER) as usize).min(COLUMNS - 1);

        let (dx, dz) = (bx - ax, bz - az);
        let length_sq = dx * dx + dz * dz;

        for cz in lo(az, bz)..=hi(az, bz) {
            for cx in lo(ax, bx)..=hi(ax, bx) {
                let px = (cx as f32 + 0.5) * VOXEL_SIZE;
                let pz = (cz as f32 + 0.5) * VOXEL_SIZE;

                // closest point on the segment //

                let s = (((px - ax) * dx + (pz - az) * dz) / length_sq).clamp(0.0, 1.0);
                let (qx, qz) = (ax + dx * s - px, az + dz * s - pz);
                let distance = (qx * qx + qz * qz).sqrt();

                let surface = surfaces[0] + (surfaces[1] - surfaces[0]) * s;
                let radius = radii[0] + (radii[1] - radii[0]) * s;
                let c = cz * COLUMNS + cx;

                if carve {
                    if distance < radius {
                        let depth = 0.25 + radius * 0.2;
                        let t = distance / radius;
                        let bed = surface - depth * (1.0 - t * t);

                        ground[c] = ground[c].min(bed);
                        water[c] = water[c].max(surface);
                    }
                } else if distance < radius + LEVEE_WIDTH {
                    // low banks poke just above the water and fade softly back into the land
                    let bank = surface + 0.15 - (distance - radius).max(0.0) * 0.25;
                    ground[c] = ground[c].max(bank);
                }
            }
        }
    }

    /*
        LAKE FILL

        ok so the 1m lake cells miss the little rim gaps and pits of the 0.125m ground,
        and those used to drain whole lakes into staircases. so lakes get flooded
        again, per column this time:
        1. spill height of every column, a priority flood over the ground (plus
           stream / sea water) starting from the edges and the sea. heights are just
           u16 voxels, so a bucket per height does the job of the heap
        2. every basin a lake cell touches floods flat up to its spill height,
           that's watertight by construction, so the settle pass never drains it
    */

    fn fill_lakes(ground: &[u16], water: &mut [u16], seeds: Vec<u32>) -> Vec<u32> {
        let sea = (Self::SEA_LEVEL * VOXELS_PER_METER) as u16;
        let top = |c: usize| ground[c].max(water[c]);

        let mut spill = vec![u16::MAX; COLUMNS * COLUMNS];
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); Tree64::VOXELS_PER_AXIS as usize];

        for z in 0..COLUMNS {
            for x in 0..COLUMNS {
                let c = z * COLUMNS + x;
                let edge = x == 0 || z == 0 || x == COLUMNS - 1 || z == COLUMNS - 1;

                if edge || ground[c] < sea {
                    spill[c] = top(c);
                    buckets[spill[c] as usize].push(c as u32);
                }
            }
        }

        for level in 0..buckets.len() {
            while let Some(c) = buckets[level].pop() {
                let c = c as usize;

                for n in Self::neighbors4(c) {
                    if spill[n] != u16::MAX {
                        continue;
                    }

                    spill[n] = top(n).max(level as u16);
                    buckets[spill[n] as usize].push(n as u32);
                }
            }
        }

        // flood every seeded basin, water < spill doubles as the not visited yet check //

        let mut stack = seeds;
        stack.retain(|&c| spill[c as usize] > ground[c as usize]);

        let mut lakes = Vec::new();

        while let Some(c) = stack.pop() {
            let c = c as usize;

            if water[c] >= spill[c] {
                continue;
            }

            water[c] = spill[c];
            lakes.push(c as u32);

            for n in Self::neighbors4(c) {
                if spill[n] > ground[n] && water[n] < spill[n] {
                    stack.push(n as u32);
                }
            }
        }

        lakes
    }

    /// the 4 neighbor columns of a column index, clipped to the world
    #[inline(always)]
    fn neighbors4(c: usize) -> impl Iterator<Item = usize> {
        let (x, z) = (c % COLUMNS, c / COLUMNS);

        [
            (x > 0).then(|| c - 1),
            (x + 1 < COLUMNS).then(|| c + 1),
            (z > 0).then(|| c - COLUMNS),
            (z + 1 < COLUMNS).then(|| c + COLUMNS),
        ]
        .into_iter()
        .flatten()
    }

    /// water may only stand 1 voxel above the top of each of its 4 neighbors,
    /// this removes every wall / spike the rasterization could leave behind
    fn settle_water(ground: &mut [u16], water: &mut [u16]) {
        let top = |ground: &[u16], water: &[u16], c: usize| ground[c].max(water[c]);

        let relax = |ground: &[u16], water: &mut [u16], x: usize, z: usize| -> bool {
            let c = z * COLUMNS + x;

            if water[c] <= ground[c] {
                return false;
            }

            let mut limit = u16::MAX;

            if x > 0 {
                limit = limit.min(top(ground, water, c - 1) + 1);
            }
            if x + 1 < COLUMNS {
                limit = limit.min(top(ground, water, c + 1) + 1);
            }
            if z > 0 {
                limit = limit.min(top(ground, water, c - COLUMNS) + 1);
            }
            if z + 1 < COLUMNS {
                limit = limit.min(top(ground, water, c + COLUMNS) + 1);
            }

            if water[c] > limit {
                water[c] = limit.max(ground[c]);
                return true;
            }

            false
        };

        // water only ever drops, alternating sweeps converge in a few passes //

        for _ in 0..256 {
            let mut changed = false;

            for z in 0..COLUMNS {
                for x in 0..COLUMNS {
                    changed |= relax(ground, water, x, z);
                }
            }

            for z in (0..COLUMNS).rev() {
                for x in (0..COLUMNS).rev() {
                    changed |= relax(ground, water, x, z);
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// top material and soil depth of every column
    fn paint(
        ground: &[u16],
        water: &[u16],
        heights: &[f32],
        detail: &[f32],
    ) -> (Vec<Material>, Vec<u16>) {
        let mut surface = vec![Material::Grass; COLUMNS * COLUMNS];
        let mut soil = vec![0u16; COLUMNS * COLUMNS];

        let sea = (Self::SEA_LEVEL * VOXELS_PER_METER) as u16;

        for cz in 0..COLUMNS {
            for cx in 0..COLUMNS {
                let c = cz * COLUMNS + cx;
                let g = ground[c];
                let jitter = Self::bilinear(detail, cx, cz);

                // slope of the smooth height field over one meter,
                // the quantized ground would band into contour stripes

                let at = |x: usize, z: usize| {
                    Self::bilinear(heights, x.min(COLUMNS - 1), z.min(COLUMNS - 1))
                };
                let slope_x = (at(cx + 4, cz) - at(cx.saturating_sub(4), cz)).abs();
                let slope_z = (at(cx, cz + 4) - at(cx, cz.saturating_sub(4))).abs();
                let slope = slope_x.max(slope_z) + jitter * 0.15;

                let height = g as f32 * VOXEL_SIZE;
                let wet = water[c] > g;
                let shore = g <= sea + 3 || Self::near_water(ground, water, cx, cz);

                let material = if wet {
                    if water[c] - g > 24 {
                        Material::Stone
                    } else {
                        Material::Dirt
                    }
                } else if slope > Self::CLIFF_SLOPE {
                    Material::Stone
                } else if height > Self::SNOW_LINE + jitter * 10.0 {
                    if slope < 1.2 {
                        Material::Snow
                    } else {
                        Material::Stone
                    }
                } else if height > Self::TREE_LINE + 8.0 + jitter * 6.0 {
                    Material::Stone
                } else if shore {
                    Material::Dirt
                } else {
                    Material::Grass
                };

                surface[c] = material;
                soil[c] = match material {
                    Material::Stone => 0,
                    Material::Snow => 3,
                    _ => (12.0 + jitter * 10.0) as u16,
                }
                .min(g);
            }
        }

        (surface, soil)
    }

    /// water within 2 columns (0.25m) on any axis, so grass gets right up to calm water
    #[inline(always)]
    fn near_water(ground: &[u16], water: &[u16], cx: usize, cz: usize) -> bool {
        const REACH: usize = 2;

        let wet = |x: usize, z: usize| {
            let c = z.min(COLUMNS - 1) * COLUMNS + x.min(COLUMNS - 1);
            water[c] > ground[c]
        };

        wet(cx + REACH, cz)
            || wet(cx.saturating_sub(REACH), cz)
            || wet(cx, cz + REACH)
            || wet(cx, cz.saturating_sub(REACH))
    }

    /// jittered 5m grid, density from biome and a forest mask
    fn plant_trees(
        seed: i32,
        ground: &[u16],
        water: &[u16],
        surface: &[Material],
        detail: &[f32],
    ) -> Vec<TreeShape> {
        const CELL: i32 = 40;

        let mut forest = FastNoiseLite::with_seed(seed.wrapping_add(7));
        forest.set_noise_type(Some(NoiseType::OpenSimplex2));
        forest.set_frequency(Some(1.0 / 70.0));

        let mut trees = Vec::new();
        let cells = COLUMNS as i32 / CELL;

        for gz in 0..cells {
            for gx in 0..cells {
                let h = hash3(gx as u32, gz as u32, seed as u32);

                let x = gx * CELL + 6 + (h % (CELL as u32 - 12)) as i32;
                let z = gz * CELL + 6 + ((h >> 8) % (CELL as u32 - 12)) as i32;
                let c = z as usize * COLUMNS + x as usize;

                if surface[c] != Material::Grass {
                    continue;
                }

                // stay off the water, the wet check covers 3m around the trunk //

                let dry = |dx: i32, dz: i32| {
                    let i = ((z + dz).clamp(0, 2047) as usize) * COLUMNS
                        + (x + dx).clamp(0, 2047) as usize;
                    water[i] <= ground[i]
                };

                if !(dry(0, 0) && dry(24, 0) && dry(-24, 0) && dry(0, 24) && dry(0, -24)) {
                    continue;
                }

                let height_m = ground[c] as f32 * VOXEL_SIZE;

                if height_m > Self::TREE_LINE {
                    continue;
                }

                // forest mask: dense patches on the hills, a few lone trees on the plains //

                let density =
                    (forest.get_noise_2d(x as f32 * VOXEL_SIZE, z as f32 * VOXEL_SIZE) * 0.5 + 0.5)
                        .powf(1.5)
                        + ((height_m - 52.0) / 60.0).clamp(0.0, 0.35);

                if unit(hash(h)) > density {
                    continue;
                }

                let r = hash3(h, 1, 2);
                let base = ground[c] as i32;
                let spruce =
                    height_m > 64.0 + Self::bilinear(detail, x as usize, z as usize) * 12.0;

                let tree = if spruce {
                    let height = 40 + (r % 32) as i32;
                    let radius = 14 + ((r >> 8) % 8) as i32;
                    TreeShape {
                        kind: TreeKind::Spruce,
                        x,
                        z,
                        base,
                        height,
                        radius,
                        min: [x - radius, base - 8, z - radius],
                        max: [x + radius, base + height + 8, z + radius],
                    }
                } else {
                    let height = 22 + (r % 20) as i32;
                    let radius = 16 + ((r >> 8) % 9) as i32;
                    let canopy_y = radius * 3 / 4;
                    TreeShape {
                        kind: TreeKind::Oak,
                        x,
                        z,
                        base,
                        height,
                        radius,
                        min: [x - radius, base - 8, z - radius],
                        max: [x + radius, base + height + canopy_y, z + radius],
                    }
                };

                trees.push(tree);
            }
        }

        trees
    }

    /// material of one voxel, ground first, then water, then trees
    #[inline(always)]
    pub fn voxel(&self, x: u32, y: u32, z: u32, trees: &[u32]) -> Material {
        let c = z as usize * COLUMNS + x as usize;
        let y16 = y as u16;
        let ground = self.ground[c];

        if y16 < ground {
            let depth = ground - 1 - y16;
            let surface = self.surface[c];

            return if surface == Material::Stone || depth >= self.soil[c] {
                Material::Stone
            } else if depth < TOP_LAYER || surface == Material::Snow {
                surface
            } else {
                Material::Dirt
            };
        }

        if y16 < self.water[c] {
            return Material::Water;
        }

        for &index in trees {
            let material = self.trees[index as usize].voxel(x as i32, y as i32, z as i32);
            if material != Material::Air {
                return material;
            }
        }

        Material::Air
    }

    /// trees in the bucket of an 8 voxel leaf that overlap it
    pub fn trees_near(&self, min: [u32; 3], out: &mut Vec<u32>) {
        out.clear();

        let bucket =
            (min[2] as usize >> BUCKET_SHIFT) * BUCKETS + (min[0] as usize >> BUCKET_SHIFT);
        let (lo, hi) = (min.map(|v| v as i32), min.map(|v| v as i32 + 7));

        for &index in &self.buckets[bucket] {
            let tree = &self.trees[index as usize];

            if (0..3).all(|a| tree.min[a] <= hi[a] && tree.max[a] >= lo[a]) {
                out.push(index);
            }
        }
    }
}

impl TreeShape {
    /// Wood, Leaves or Air at a voxel
    #[inline(always)]
    pub fn voxel(&self, x: i32, y: i32, z: i32) -> Material {
        if x < self.min[0] || y < self.min[1] || z < self.min[2] {
            return Material::Air;
        }
        if x > self.max[0] || y > self.max[1] || z > self.max[2] {
            return Material::Air;
        }

        let (dx, dy, dz) = (x - self.x, y - self.base, z - self.z);

        // 4x4 voxel trunk, sunk 1m into the ground so slopes never float it //

        if (-2..2).contains(&dx) && (-2..2).contains(&dz) && dy >= -8 && dy < self.height {
            return Material::Wood;
        }

        let (fx, fz) = (dx as f32 + 0.5, dz as f32 + 0.5);
        let r = self.radius as f32;

        let inside = match self.kind {
            TreeKind::Oak => {
                // ellipsoid around the trunk top //
                let fy = (dy - self.height) as f32 + 0.5;
                let v = (fx * fx + fz * fz) / (r * r) + (fy * fy) / (r * r * 0.5);
                v <= 1.0 && (v < 0.7 || !hash3(x as u32, y as u32, z as u32).is_multiple_of(3))
            }
            TreeKind::Spruce => {
                // stacked cones, wide at the bottom and pointed at the top //
                let start = self.height / 4;
                let span = (self.height + 8 - start) as f32;
                let t = (dy - start) as f32 / span;

                if !(0.0..=1.0).contains(&t) {
                    false
                } else {
                    let layer = ((dy - start) % 12) as f32 / 12.0;
                    let radius = r * (1.0 - t) * (0.55 + 0.45 * (1.0 - layer));
                    fx * fx + fz * fz <= radius * radius
                }
            }
        };

        if inside {
            Material::Leaves
        } else {
            Material::Air
        }
    }
}

#[inline(always)]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// lowbias32 integer hash
#[inline(always)]
pub fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[inline(always)]
pub fn hash3(a: u32, b: u32, c: u32) -> u32 {
    hash(a ^ hash(b ^ hash(c)))
}

/// top 24 bits as [0, 1)
#[inline(always)]
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}
