use super::lightcache::{Claim, LightCache};
use super::lighting::View;
use super::shading::*;
use super::*;
use crate::simd::packetray::{Packet, PacketBuilder};
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::sun::SUN_SIZE;
use crate::world::tree64::{PacketHit, Tree64};
use crate::world::voxel::{VOXEL_SIZE, VOXELS_PER_METER};

/// one light sample to take for a face
#[derive(Clone, Copy)]
struct Job {
    /// cache slot it goes into, or (PRIVATE | n) for the tile's own n-th private face
    target: u32,
    voxel: [u32; 3],
    face: u8,
    seed: u32,
}

impl Job {
    const EMPTY: Self = Self {
        target: 0,
        voxel: [0; 3],
        face: 0,
        seed: 0,
    };
}

const NO_WATER: u8 = u8::MAX;

/// marks a job whose light only this tile keeps (see Claim::Missing)
const PRIVATE: u32 = 1 << 31;

/// pixels along one side of a tile
const TILE: usize = Engine::PIXELS_PER_TILE_AXIS;

const TILE_PACKETS: usize = (TILE / 2) * (TILE / 2);

const BLOCK_PACKETS: usize = (Engine::BEAM_PIXELS / 2) * (Engine::BEAM_PIXELS / 2);

const TILE_JOBS: usize = TILE_PACKETS * 4;

/// most private faces a tile keeps, past that they just get ambient light for a frame
const TILE_PRIVATES: usize = 64;

impl Engine {
    pub const MAX_ACCUMULATED_FRAMES: u32 = 128;

    /// running mean window of a still pixel, that's the anti aliasing
    pub const ANTI_ALIAS_FRAMES: f32 = 16.0;

    /// bounce rays that get further than this (meters) just count as open sky
    pub const BOUNCE_MAX_DISTANCE: f32 = 48.0;

    /// shadow rays from bounce hits give up here (meters), far blockers barely matter for bounce light
    pub const BOUNCE_SHADOW_DISTANCE: f32 = 64.0;

    /// frame budgets for the fps target setting (ms), infinity = no target at all
    pub const FRAME_BUDGETS: [f32; 3] = [1000.0 / 60.0, 1000.0 / 30.0, f32::INFINITY];

    /// share of the budget a frame aims for, the rest goes to ui and present
    pub const BUDGET_SHARE: f32 = 0.85;

    pub const MAX_EVERY: u32 = 64;

    /// face samples a frame always shades, even over budget (slow frames are camera rays anyway)
    pub const LIT_FLOOR: u32 = 16384;

    // lod ladder: full res up to LOD_START, then 2x2x2 / 4x4x4 / 8x8x8 each time the distance doubles
    pub const LOD_START: f32 = 24.0;
    pub const CAMERA_CONE: f32 = 2.0 / Self::LOD_START;

    /// far water traces every nth pixel's reflection and blurs the rest together
    pub const REFLECTION_STEP: [usize; 4] = [1, 2, 4, 4];
    pub const REFLECTION_BLUR: [usize; 4] = [0, 2, 4, 6];

    /// max skip map hops per beam, and the shortest hop worth a lookup (meters)
    pub const BEAM_HOPS: usize = 16;
    pub const BEAM_MIN_HOP: f32 = 0.25;

    pub const BEAM_PIXELS: usize = 8;

    /// samples a brand new face gets right away, one sample per face looks like confetti
    pub const FIRST_SAMPLES: u32 = 4;

    /// lod cones (voxels of width per meter) for rays leaving a surface
    pub const SHADOW_CONE: f32 = SUN_SIZE * VOXELS_PER_METER;
    pub const BOUNCE_CONE: f32 = 0.4;

    pub const BORROW_SAMPLES: u32 = 4;

    pub fn resize_accumulation_buffer(&mut self, new_x: usize, new_y: usize) {
        self.frame_accumulation_buffer = vec![[0.0; 4]; new_x * new_y];
        self.frame_depths = vec![f32::INFINITY; new_x * new_y];
        self.frame_faces = vec![SKY_FACE; new_x * new_y];

        self.frame_accumulation_count = 0;
    }

    /// start over and forget the face light too (sun / lighting changes)
    pub fn restart_image(&mut self) {
        self.frame_accumulation_count = 0;
        self.lights.clear();
    }

    /// false once a still image has converged
    pub fn render(&mut self, buffer_address: usize) -> bool {
        /*
            every pixel gets its own camera ray, the light comes out of the face light cache (lightcache.rs)
            still frames jitter and keep a running mean, that's the anti aliasing
        */

        let moving = self.frame_accumulation_count == 0;

        // a finished image just stays in the pixel buffer, nothing to do //

        if self.frame_accumulation_count >= Self::MAX_ACCUMULATED_FRAMES {
            return false;
        }

        let start_time = std::time::Instant::now();

        self.camera.update();

        let budget = Self::FRAME_BUDGETS[self.fps_target];
        let frame = self.frame_index;

        self.frame_accumulation_count += 1;
        self.samples = self.frame_accumulation_count as f32;
        self.frame_index = self.frame_index.wrapping_add(1);

        // sub pixel jitter from the r2 sequence, same for the whole frame //
        // moving frames skip it, jitter there only makes edges shimmer

        let jitter = if moving {
            (0.0, 0.0)
        } else {
            (
                (0.5 + frame as f32 * 0.754_877_7).fract() - 0.5,
                (0.5 + frame as f32 * 0.569_840_3).fract() - 0.5,
            )
        };

        let camera = self.camera;

        let setup = Setup {
            camera,
            builder: PacketBuilder::new(camera.center, camera.pixel_delta_x, camera.pixel_delta_y),
            jitter,
            underwater: self.world.water_at(camera.center),
            moving,
            frame,
            every: if budget.is_finite() { self.every } else { 1 },
            width: self.render_width,
            accumulation: self.frame_accumulation_buffer.as_mut_ptr() as usize,
            depths: self.frame_depths.as_mut_ptr() as usize,
            faces: self.frame_faces.as_mut_ptr() as usize,
            pixels: buffer_address,
        };

        let tile_xcount = self.render_width / TILE;
        let tile_ycount = self.render_height / TILE;
        let max_dim = tile_xcount.max(tile_ycount).next_power_of_two();

        (0..max_dim * max_dim)
            .into_par_iter()
            .filter_map(|i| {
                let (tile_x, tile_y) = morton_decode(i);
                (tile_x < tile_xcount && tile_y < tile_ycount)
                    .then(|| (tile_x * TILE, (tile_ycount - 1 - tile_y) * TILE))
            })
            .for_each(|(x0, y0)| {
                let mut tile = Tile::new(x0, y0);

                tile.primary(self, &setup);
                tile.reflections(self);
                tile.faces(self, &setup);
                tile.shade(self, &setup);
                tile.compose(self, &setup);
            });

        self.faces_lit = self.lights.take_lit();

        // settled faces refresh less often after a slow frame, more often when there's room //

        let ms = start_time.elapsed().as_secs_f32() * 1000.0;

        if budget.is_finite() {
            let limit = budget * Self::BUDGET_SHARE;

            if ms > limit && self.faces_lit > Self::LIT_FLOOR {
                self.every = (self.every * 2).min(Self::MAX_EVERY);
            } else if ms < limit * 0.6 || self.faces_lit < Self::LIT_FLOOR / 2 {
                self.every = (self.every / 2).max(1);
            }
        }

        true
    }

    /// redraw the pixel buffer straight from the accumulation buffer, tone mapping only
    pub fn resolve(&self, buffer_address: usize) {
        // rows don't care about each other, 4 pixels per display call //

        self.frame_accumulation_buffer
            .par_chunks(self.render_width)
            .enumerate()
            .for_each(|(y, row)| {
                let pointer = buffer_address as *mut u32;

                for (i, means) in row.chunks_exact(4).enumerate() {
                    let colors = display(packed([0, 1, 2, 3].map(|p| {
                        let [r, g, b, _] = means[p];
                        [r, g, b]
                    })))
                    .to_array();

                    // every row is written by one thread //
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            colors.as_ptr(),
                            pointer.add(y * self.render_width + i * 4),
                            4,
                        );
                    }
                }
            });
    }

    /// nothing's in the way before start (meters), see beam_start
    #[inline(always)]
    fn primary(&self, packet: Packet, underwater: bool, start: f32) -> PacketHit {
        let tree = &self.world.tree;
        let all = SimdU32::ALL_ACTIVE;

        if underwater {
            let visible = SimdF32::from_float(WATER_VISIBILITY);
            tree.trace_from::<false, true>(packet, all, visible, Self::CAMERA_CONE, start)
        } else {
            tree.trace_from::<false, false>(packet, all, SimdF32::INFY, Self::CAMERA_CONE, start)
        }
    }

    // how far every camera ray of a block can jump before it could hit anything (meters)
    // the center ray marches the skip map, each hop goes until the fan of rays would touch a wall
    fn beam_start(&self, origin: Vec3, dir: Vec3, spread: f32, underwater: bool) -> f32 {
        let tree = &self.world.tree;
        let walls = [Tree64::WIDTH; 3];
        let mut t = 0.0f32;

        for _ in 0..Self::BEAM_HOPS {
            let p = origin + dir * t;
            let reach = spread * t;

            if (0..3).any(|a| p[a] < reach || p[a] > walls[a] - reach) {
                break;
            }

            let meter = [0, 1, 2].map(|a| (p[a] as usize).min(Tree64::METERS_PER_AXIS - 1));
            let empty = tree.skip[Tree64::skip_index(meter)][underwater as usize] as f32;

            // the cube reaches empty - 1 meters around this meter //

            if empty < 2.0 {
                break;
            }

            let mut hop = f32::INFINITY;

            for a in 0..3 {
                let lo = (meter[a] as f32 - (empty - 1.0)).max(0.0);
                let hi = (meter[a] as f32 + empty).min(walls[a]);
                let speed = dir[a].abs();
                let (ahead, behind) = if dir[a] >= 0.0 {
                    (hi - p[a], p[a] - lo)
                } else {
                    (p[a] - lo, hi - p[a])
                };

                // the fan's edge closes on the wall ahead at speed + spread, behind only if it spreads faster //
                hop = hop.min((ahead - reach) / (speed + spread));

                if spread > speed {
                    hop = hop.min((behind - reach) / (spread - speed));
                }
            }

            if hop < Self::BEAM_MIN_HOP {
                break;
            }

            t += hop;
        }

        // a hair short, float rounding never gets to decide //
        (t - 0.01).max(0.0)
    }
}

struct Setup {
    camera: Camera,
    builder: PacketBuilder,
    jitter: (f32, f32),
    underwater: bool,
    moving: bool,
    frame: u32,
    every: u32,
    width: usize,
    /// tiles never overlap, so every pixel is only ever written by one thread
    accumulation: usize,
    depths: usize,
    faces: usize,
    pixels: usize,
}

/// one 16x16 tile, everything on the stack, one thread per tile at a time
struct Tile {
    x0: usize,
    y0: usize,

    views: [View; TILE_PACKETS],
    keys: [[u64; 4]; TILE_PACKETS],

    /// lod level of the water every pixel sees, NO_WATER if it doesn't
    water_levels: [[u8; TILE]; TILE],
    reflections: [[[f32; 3]; TILE]; TILE],
    traced: [[bool; TILE]; TILE],

    jobs: [Job; TILE_JOBS],
    queued: usize,

    /// new faces another tile is busy shading, this tile shades its own copy: (key, light sum, samples)
    privates: [(u64, [f32; 3], f32); TILE_PRIVATES],
    private_count: usize,
    claimed: [u64; TILE_PACKETS * 4],
    claimed_count: usize,
}

impl Tile {
    fn new(x0: usize, y0: usize) -> Self {
        Self {
            x0,
            y0,
            views: [View::EMPTY; TILE_PACKETS],
            keys: [[0; 4]; TILE_PACKETS],
            water_levels: [[NO_WATER; TILE]; TILE],
            reflections: [[[0.0; 3]; TILE]; TILE],
            traced: [[false; TILE]; TILE],
            jobs: [Job::EMPTY; TILE_JOBS],
            queued: 0,
            privates: [(0, [0.0; 3], 0.0); TILE_PRIVATES],
            private_count: 0,
            claimed: [0; TILE_PACKETS * 4],
            claimed_count: 0,
        }
    }

    /* PRIMARY */

    fn primary(&mut self, engine: &Engine, setup: &Setup) {
        let camera = &setup.camera;
        let mut start = 0.0;

        for k in 0..TILE_PACKETS {
            // packets go in morton order, so every 16 of them are one 8x8 block //
            if k % BLOCK_PACKETS == 0 {
                start = self.block_start(engine, setup, k);
            }

            let (px, py) = packet_xy(k);
            let (x, y) = (self.x0 + px, self.y0 + py);

            let packet00 = camera.pixel_00 + camera.pixel_delta_x * (x as f32 + setup.jitter.0)
                - camera.pixel_delta_y * (y as f32 + setup.jitter.1);

            let packet = setup.builder.make(packet00);
            let hit = engine.primary(packet, setup.underwater, start);
            self.views[k] = engine.view(packet, hit, setup.underwater, Engine::CAMERA_CONE);

            let ts = hit.distances.to_array();
            let dirs = lanes(packet.directions);
            let normals = lanes(hit.normals);
            let hits = hit.hit.to_array();

            for lane in 0..4 {
                let i = (y + (lane >> 1)) * setup.width + x + (lane & 1);
                let hit = hits[lane] != 0;

                let depth = if hit {
                    ts[lane] * Vec3 { data: dirs[lane] }.dot(camera.front)
                } else {
                    f32::INFINITY
                };

                // SAFETY: see Setup, this pixel belongs to this tile
                unsafe {
                    *(setup.depths as *mut f32).add(i) = depth;
                    *(setup.faces as *mut u8).add(i) = face_id(normals[lane], hit);
                }
            }
        }
    }

    /// where the camera rays of the 8x8 block at packet k can start walking (see Engine::beam_start)
    fn block_start(&self, engine: &Engine, setup: &Setup, k: usize) -> f32 {
        let camera = &setup.camera;

        let ray = |x: f32, y: f32| {
            (camera.pixel_00 + camera.pixel_delta_x * (x + setup.jitter.0)
                - camera.pixel_delta_y * (y + setup.jitter.1)
                - camera.center)
                .normalize()
        };

        let (bx, by) = packet_xy(k);
        let (bx, by) = ((self.x0 + bx) as f32, (self.y0 + by) as f32);
        let side = Engine::BEAM_PIXELS as f32;
        let half = side * 0.5 - 0.5;
        let center = ray(bx + half, by + half);

        // a pixel past every corner, the block's rays never stray further than those //
        let spread = [(-1.0, -1.0), (side, -1.0), (-1.0, side), (side, side)]
            .map(|(cx, cy)| (ray(bx + cx, by + cy) - center).length())
            .into_iter()
            .fold(0.0, f32::max);

        engine.beam_start(camera.center, center, spread, setup.underwater)
    }

    /* REFLECTIONS */

    fn reflections(&mut self, engine: &Engine) {
        let mut queue = [(0usize, 0usize); TILE_PACKETS * 4];
        let mut queued = 0;

        for k in 0..TILE_PACKETS {
            let view = &self.views[k];

            if view.water.has_none_active() {
                continue;
            }

            let water = view.water.to_array();
            let origins = lanes(view.mirror_origins);

            for lane in 0..4 {
                if water[lane] == 0 {
                    continue;
                }

                let (tx, ty) = lane_xy(k, lane);
                let away = Vec3 {
                    data: origins[lane],
                } - engine.camera.center;
                let level = ladder(away.length(), WATER_REFLECTION_DISTANCE) as usize;
                let step = Engine::REFLECTION_STEP[level];
                self.water_levels[ty][tx] = level as u8;

                if tx % step == 0 && ty % step == 0 {
                    queue[queued] = (k, lane);
                    queued += 1;
                }
            }
        }

        self.trace_mirrors(engine, &queue[..queued]);
        queued = 0;

        for k in 0..TILE_PACKETS {
            for lane in 0..4 {
                let (tx, ty) = lane_xy(k, lane);

                if self.water_levels[ty][tx] != NO_WATER && self.reflection(tx, ty).is_none() {
                    queue[queued] = (k, lane);
                    queued += 1;
                }
            }
        }

        self.trace_mirrors(engine, &queue[..queued]);
    }

    fn trace_mirrors(&mut self, engine: &Engine, queue: &[(usize, usize)]) {
        for group in queue.chunks(4) {
            let pick = [0, 1, 2, 3].map(|j| group[j.min(group.len() - 1)]);
            let origins = pick.map(|(k, lane)| lanes(self.views[k].mirror_origins)[lane]);
            let dirs = pick.map(|(k, lane)| lanes(self.views[k].mirror_dirs)[lane]);
            let colors = lanes(engine.reflect(packed(origins), packed(dirs)));

            for (j, &(k, lane)) in group.iter().enumerate() {
                let (tx, ty) = lane_xy(k, lane);
                self.reflections[ty][tx] = colors[j];
                self.traced[ty][tx] = true;
            }
        }
    }

    /// far out a water pixel blends the traced reflections around it, None = nothing traced yet
    fn reflection(&self, tx: usize, ty: usize) -> Option<[f32; 3]> {
        let level = self.water_levels[ty][tx];

        if level == NO_WATER {
            return None;
        }

        let level = level as usize;

        if level == 0 {
            return self.traced[ty][tx].then_some(self.reflections[ty][tx]);
        }

        let (step, reach) = (
            Engine::REFLECTION_STEP[level],
            Engine::REFLECTION_BLUR[level],
        );
        let grid = |c: usize| {
            (c.saturating_sub(reach).div_ceil(step) * step..=(c + reach).min(TILE - 1))
                .step_by(step)
        };

        let mut total = [0.0f32; 3];
        let mut weight = 0.0;

        for gy in grid(ty) {
            for gx in grid(tx) {
                if !self.traced[gy][gx] {
                    continue;
                }

                let w = ((reach + 1 - gx.abs_diff(tx)) * (reach + 1 - gy.abs_diff(ty))) as f32;
                total = [0, 1, 2].map(|c| self.reflections[gy][gx][c].mul_add(w, total[c]));
                weight += w;
            }
        }

        if weight > 0.0 {
            Some(total.map(|c| c / weight))
        } else {
            self.traced[ty][tx].then_some(self.reflections[ty][tx])
        }
    }

    /* FACES */

    /// which face each lit lane sees, neighbors on the same face don't even ask
    fn faces(&mut self, engine: &Engine, setup: &Setup) {
        let mut last = 0u64;

        for k in 0..TILE_PACKETS {
            let view = self.views[k];
            let lit = view.lit.to_array();
            let points = lanes(view.points);
            let normals = lanes(view.normals);
            let distances = view.distances.to_array();

            for lane in 0..4 {
                if lit[lane] == 0 {
                    continue;
                }

                let (voxel, face) = face_of(points[lane], normals[lane]);
                let key = LightCache::key(voxel, face, ladder(distances[lane], Engine::LOD_START));
                self.keys[k][lane] = key;

                if key == last {
                    continue;
                }
                last = key;

                let (target, samples) = match engine.lights.claim(key, setup.frame, setup.every) {
                    Claim::Lit => continue,
                    Claim::Shade { slot, samples } => {
                        if samples == 0 && self.claimed_count < self.claimed.len() {
                            self.claimed[self.claimed_count] = key;
                            self.claimed_count += 1;
                        }
                        (slot, samples)
                    }
                    Claim::Missing => {
                        // ours after all, already getting its own light, or no room left //
                        if self.claimed[..self.claimed_count].contains(&key)
                            || self.privates[..self.private_count]
                                .iter()
                                .any(|p| p.0 == key)
                            || self.private_count == TILE_PRIVATES
                        {
                            continue;
                        }

                        self.privates[self.private_count].0 = key;
                        self.private_count += 1;
                        (PRIVATE | (self.private_count - 1) as u32, 0)
                    }
                };

                let (tx, ty) = lane_xy(k, lane);
                let pixel = (self.y0 + ty) * setup.width + self.x0 + tx;
                let copies = if samples == 0 {
                    Engine::FIRST_SAMPLES
                } else {
                    1
                };

                for copy in 0..copies {
                    // a full queue gets shaded right away to make room //
                    if self.queued == TILE_JOBS {
                        self.shade(engine, setup);
                    }

                    self.jobs[self.queued] = Job {
                        target,
                        voxel,
                        face,
                        seed: pixel as u32 * Engine::FIRST_SAMPLES + copy,
                    };
                    self.queued += 1;
                }
            }
        }
    }

    /* SHADE FACES */

    fn shade(&mut self, engine: &Engine, setup: &Setup) {
        for group in self.jobs[..self.queued].chunks(4) {
            // short groups just repeat their last job //
            let pick = [0, 1, 2, 3].map(|j| group[j.min(group.len() - 1)]);

            let mut rng = Rng::new(SimdU32::from(pick.map(|job| job.seed)), setup.frame);
            let (u1, u2) = (rng.next().to_array(), rng.next().to_array());

            let mut starts = [[0.0f32; 3]; 4];
            let mut normals = [[0.0f32; 3]; 4];

            for (j, job) in pick.iter().enumerate() {
                let axis = job.face as usize / 2;
                let sign = if job.face & 1 == 1 { 1.0 } else { -1.0 };
                let mut spot = job.voxel.map(|c| c as f32);

                spot[axis] += (sign > 0.0) as u32 as f32;
                spot[(axis + 1) % 3] += u1[j];
                spot[(axis + 2) % 3] += u2[j];

                normals[j][axis] = sign;
                starts[j] =
                    [0, 1, 2].map(|a| spot[a].mul_add(VOXEL_SIZE, normals[j][a] * SURFACE_OFFSET));
            }

            let light = lanes(engine.face_light(packed(starts), packed(normals), &mut rng));

            for (j, job) in group.iter().enumerate() {
                if job.target & PRIVATE == 0 {
                    engine.lights.add(job.target, light[j]);
                } else {
                    let private = &mut self.privates[(job.target & !PRIVATE) as usize];
                    private.1 = [0, 1, 2].map(|c| private.1[c] + light[j][c]);
                    private.2 += 1.0;
                }
            }
        }

        self.queued = 0;
    }

    /* COMPOSE + DISPLAY */

    /// surface times face light, plus reflections, tone mapped into the pixel buffer
    fn compose(&self, engine: &Engine, setup: &Setup) {
        for k in 0..TILE_PACKETS {
            let view = &self.views[k];
            let lit = view.lit.to_array();

            let face_lights = packed([0, 1, 2, 3].map(|lane| {
                if lit[lane] == 0 {
                    return [0.0; 3];
                }

                let key = self.keys[k][lane];

                if let Some((light, _)) = engine.lights.lookup(key) {
                    return light;
                }

                // another tile's new face, or kicked out mid frame (rare) //
                match self.privates[..self.private_count]
                    .iter()
                    .find(|p| p.0 == key)
                {
                    Some(&(_, sum, n)) if n > 0.0 => sum.map(|c| c / n),
                    _ => BOUNCE_AMBIENT.data,
                }
            }));

            let colors = lanes(Packed_Vec3_x4::select(
                view.lit,
                view.radiance + view.base * face_lights,
                view.radiance,
            ));
            let fresnel = view.fresnel.to_array();

            for (lane, color) in colors.iter().enumerate() {
                let (tx, ty) = lane_xy(k, lane);

                let color = match self.reflection(tx, ty) {
                    Some(mirror) => [0, 1, 2].map(|c| mirror[c].mul_add(fresnel[lane], color[c])),
                    None => *color,
                };

                // SAFETY: see Setup, this pixel belongs to this tile
                let mean = unsafe {
                    &mut *(setup.accumulation as *mut [f32; 4])
                        .add((self.y0 + ty) * setup.width + self.x0 + tx)
                };

                *mean = if setup.moving {
                    [color[0], color[1], color[2], 1.0]
                } else {
                    let n = (mean[3] + 1.0).min(Engine::ANTI_ALIAS_FRAMES);
                    let mix = |old: f32, new: f32| old + (new - old) / n;
                    [
                        mix(mean[0], color[0]),
                        mix(mean[1], color[1]),
                        mix(mean[2], color[2]),
                        mean[3] + 1.0,
                    ]
                };
            }
        }

        for ty in 0..TILE {
            let index = (self.y0 + ty) * setup.width + self.x0;

            for x in (0..TILE).step_by(4) {
                // SAFETY: see Setup, these pixels belong to this tile
                unsafe {
                    let first = (setup.accumulation as *const [f32; 4]).add(index + x);
                    let means = *(first as *const [[f32; 4]; 4]);
                    let colors = display(packed(means.map(|[r, g, b, _]| [r, g, b])));

                    std::ptr::copy_nonoverlapping(
                        colors.to_array().as_ptr(),
                        (setup.pixels as *mut u32).add(index + x),
                        4,
                    );
                }
            }
        }
    }
}

/// face id of pixels that see sky
const SKY_FACE: u8 = 255;

/// which of the 6 axis faces a voxel normal is, sky gets its own
#[inline(always)]
pub(super) fn face_id(n: [f32; 3], hit: bool) -> u8 {
    if !hit {
        return SKY_FACE;
    }

    let axis = if n[0] != 0.0 {
        0
    } else if n[1] != 0.0 {
        1
    } else {
        2
    };
    axis * 2 + (n[axis as usize] > 0.0) as u8
}

/// half a voxel back against the normal lands inside the voxel
#[inline(always)]
pub(super) fn face_of(p: [f32; 3], n: [f32; 3]) -> ([u32; 3], u8) {
    let voxel = [0, 1, 2].map(|a| {
        ((p[a] - n[a] * 0.5 * VOXEL_SIZE) * VOXELS_PER_METER)
            .floor()
            .clamp(0.0, 2047.0) as u32
    });

    (voxel, face_id(n, true))
}

/// lod rung for a distance (meters): 0 up to start, then one more every time it doubles
#[inline(always)]
pub(super) fn ladder(distance: f32, start: f32) -> u32 {
    (distance >= start) as u32 + (distance >= start * 2.0) as u32 + (distance >= start * 4.0) as u32
}

/// a packet split into its 4 [x, y, z] lanes
#[inline(always)]
pub(super) fn lanes(v: Packed_Vec3_x4) -> [[f32; 3]; 4] {
    let (x, y, z) = (v.xs.to_array(), v.ys.to_array(), v.zs.to_array());
    [0, 1, 2, 3].map(|i| [x[i], y[i], z[i]])
}

/// 4 [x, y, z] lanes packed back up
#[inline(always)]
pub(super) fn packed(lanes: [[f32; 3]; 4]) -> Packed_Vec3_x4 {
    Packed_Vec3_x4::new(
        lanes.map(|v| v[0]),
        lanes.map(|v| v[1]),
        lanes.map(|v| v[2]),
    )
}

/// top left pixel of packet k in its tile, morton order:
/// bits 0, 2, 4 of k are x, bits 1, 3, 5 are y, in 2 pixel steps
#[inline(always)]
const fn packet_xy(k: usize) -> (usize, usize) {
    let x = (k & 1) | ((k >> 1) & 2) | ((k >> 2) & 4);
    let y = ((k >> 1) & 1) | ((k >> 2) & 2) | ((k >> 3) & 4);
    (x * 2, y * 2)
}

#[inline(always)]
const fn lane_xy(k: usize, lane: usize) -> (usize, usize) {
    let (x, y) = packet_xy(k);
    (x + (lane & 1), y + (lane >> 1))
}

fn morton_decode(i: usize) -> (usize, usize) {
    const EXTRACT_EVEN: fn(usize) -> usize = |i| {
        (((i & 0x55555555).wrapping_mul(0xC00030000C0003)) & 0x0600180060008001)
            .wrapping_mul(0x0101010101010101)
            >> 56
    };
    const EXTRACT_ODD: fn(usize) -> usize = |i| {
        (((i & 0xAAAAAAAA).wrapping_mul(0xC00030000C0003)) & 0x03000C003000C000)
            .wrapping_mul(0x0101010101010101)
            >> 56
    };

    (EXTRACT_EVEN(i), EXTRACT_ODD(i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    #[test]
    fn test_beam_start() {
        let (width, height) = (256, 160);
        let mut engine = Engine::new(width, height, World::new());

        // xorshift, deterministic without pulling in a crate //
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };

        let (mut packets, mut skipped) = (0, 0.0);

        for view in 0..90 {
            let height_above = [2.0, 40.0, 150.0][view % 3];
            let (x, z) = (10.0 + rand() * 236.0, 10.0 + rand() * 236.0);
            let ground = (0..250)
                .rev()
                .find(|&y| {
                    engine
                        .world
                        .voxel([(x * 8.0) as i32, y * 8, (z * 8.0) as i32])
                        .is_some()
                })
                .unwrap_or(0) as f32;

            engine.camera.center = Vec3::new(x, (ground + height_above).min(250.0), z);
            engine.camera.yaw = rand() * std::f32::consts::TAU;
            engine.camera.pitch = (rand() - 0.7) * 1.2;
            engine.camera.update();

            let camera = engine.camera;
            let setup = Setup {
                camera,
                builder: PacketBuilder::new(
                    camera.center,
                    camera.pixel_delta_x,
                    camera.pixel_delta_y,
                ),
                jitter: (rand() - 0.5, rand() - 0.5),
                underwater: engine.world.water_at(camera.center),
                moving: true,
                frame: 0,
                every: 1,
                width,
                accumulation: 0,
                depths: 0,
                faces: 0,
                pixels: 0,
            };

            // every ray from its block's start hits exactly what it hits from the eye //

            for (x0, y0) in [(0, 0), (112, 64), (240, 144), (128, 0), (16, 96)] {
                let tile = Tile::new(x0, y0);

                for k in 0..TILE_PACKETS {
                    let start = tile.block_start(&engine, &setup, k - k % BLOCK_PACKETS);
                    let (px, py) = packet_xy(k);
                    let packet00 = camera.pixel_00
                        + camera.pixel_delta_x * ((x0 + px) as f32 + setup.jitter.0)
                        - camera.pixel_delta_y * ((y0 + py) as f32 + setup.jitter.1);
                    let packet = setup.builder.make(packet00);

                    let full = engine.primary(packet, setup.underwater, 0.0);
                    let jumped = engine.primary(packet, setup.underwater, start);

                    assert_eq!(
                        full.hit.to_array(),
                        jumped.hit.to_array(),
                        "view {view} start {start}"
                    );
                    assert_eq!(full.materials, jumped.materials);
                    assert_eq!(full.distances.to_array(), jumped.distances.to_array());
                    assert_eq!(lanes(full.normals), lanes(jumped.normals));

                    packets += 1;
                    skipped += start;
                }
            }
        }

        println!(
            "{packets} packets, {:.1}m skipped on average",
            skipped / packets as f32
        );
        assert!(skipped / packets as f32 > 1.0);
    }

    #[test]
    fn test_faces_and_patches() {
        let top = [
            (10.0 + 0.3) * VOXEL_SIZE,
            21.0 * VOXEL_SIZE,
            (30.0 + 0.7) * VOXEL_SIZE,
        ];
        assert_eq!(face_of(top, [0.0, 1.0, 0.0]), ([10, 20, 30], 3));

        // same spot seen from below is the voxel above's bottom face //
        assert_eq!(face_of(top, [0.0, -1.0, 0.0]), ([10, 21, 30], 2));

        let lod = |d: f32| ladder(d, Engine::LOD_START);
        assert_eq!(lod(0.5), 0);
        assert_eq!(lod(23.9), 0);
        assert_eq!(lod(24.0), 1);
        assert_eq!(lod(48.0), 2);
        assert_eq!(lod(95.0), 2);
        assert_eq!(lod(96.0), 3);
        assert_eq!(lod(1000.0), 3);

        assert_eq!(ladder(31.9, WATER_REFLECTION_DISTANCE), 0);
        assert_eq!(ladder(32.0, WATER_REFLECTION_DISTANCE), 1);
    }
}
