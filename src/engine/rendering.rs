use super::lightcache::{Claim, LightCache};
use super::shading::*;
use super::*;
use crate::simd::packetray::{Packet, PacketBuilder};
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::sun::SUN_SIZE;
use crate::world::tree64::{PacketHit, Tree64};
use crate::world::voxel::{Material, VOXEL_SIZE, VOXELS_PER_METER};

/// what one packet of camera rays ends up looking at, water already dealt with,
/// the light on the surface gets read from the face light cache later
#[derive(Clone, Copy)]
struct View {
    /// light that made it to the camera without landing on a lit surface (sky, water, fog),
    /// water reflections get added on top later, see REFLECTIONS
    radiance: Packed_Vec3_x4,
    /// how much the surface reflects, already dimmed by any water in front of it
    base: Packed_Vec3_x4,
    /// spot on the surface (meters) and its face normal
    points: Packed_Vec3_x4,
    normals: Packed_Vec3_x4,
    /// how far the light traveled from the surface to the eye (meters), sizes the light patch
    distances: SimdF32,
    /// lanes that actually see a surface worth lighting
    lit: SimdU32,

    /// lanes looking at a water surface, how much of its reflection shows (fresnel),
    /// and the reflection ray: from just above the surface, mirrored
    water: SimdU32,
    fresnel: SimdF32,
    mirror_origins: Packed_Vec3_x4,
    mirror_dirs: Packed_Vec3_x4,
}

impl View {
    const EMPTY: Self = Self {
        radiance: Packed_Vec3_x4::ZEROES,
        base: Packed_Vec3_x4::ZEROES,
        points: Packed_Vec3_x4::ZEROES,
        normals: Packed_Vec3_x4::ZEROES,
        distances: SimdF32::ZEROES,
        lit: SimdU32::ALL_INACTIVE,
        water: SimdU32::ALL_INACTIVE,
        fresnel: SimdF32::ZEROES,
        mirror_origins: Packed_Vec3_x4::ZEROES,
        mirror_dirs: Packed_Vec3_x4::ZEROES,
    };
}

/// one light sample to take for a face
#[derive(Clone, Copy)]
struct Job {
    /// cache slot it goes into, or (PRIVATE | n) for the tile's own n-th private face
    target: u32,
    /// the voxel and which of its faces
    voxel: [u32; 3],
    face: u8,
    /// random seed, different for every sample of the same face
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

/// water level of pixels that don't see water
const NO_WATER: u8 = u8::MAX;

/// marks a job whose light only this tile keeps (see Claim::Missing)
const PRIVATE: u32 = 1 << 31;

/// pixels along one side of a tile
const TILE: usize = Engine::PIXELS_PER_TILE_AXIS;

/// packets in a tile (2x2 pixels each)
const TILE_PACKETS: usize = (TILE / 2) * (TILE / 2);

/// packets in one beam block
const BLOCK_PACKETS: usize = (Engine::BEAM_PIXELS / 2) * (Engine::BEAM_PIXELS / 2);

/// face samples a tile queues up before shading them
const TILE_JOBS: usize = TILE_PACKETS * 4;

/// most private faces a tile keeps, past that they just get ambient light for a frame
const TILE_PRIVATES: usize = 64;

impl Engine {
    /// once a still image has this many frames we just stop tracing
    pub const MAX_ACCUMULATED_FRAMES: u32 = 128;

    /// a still pixel's color is the mean of its last this many frames, that's the anti aliasing
    pub const ANTI_ALIAS_FRAMES: f32 = 16.0;

    /// bounce rays that get further than this (meters) just count as open sky
    pub const BOUNCE_MAX_DISTANCE: f32 = 48.0;

    /// shadow rays from bounce hits give up here (meters), far blockers barely matter for bounce light
    pub const BOUNCE_SHADOW_DISTANCE: f32 = 64.0;

    /// frame budgets for the fps target setting (ms), infinity = no target at all
    pub const FRAME_BUDGETS: [f32; 3] = [1000.0 / 60.0, 1000.0 / 30.0, f32::INFINITY];

    /// share of the budget a frame aims for, the rest goes to ui and present
    pub const BUDGET_SHARE: f32 = 0.85;

    /// slowest the budget lets settled faces refresh (every nth frame)
    pub const MAX_EVERY: u32 = 64;

    /// face samples a frame always gets to shade, even over budget. shading is cheap,
    /// a slow frame is mostly camera rays, so starving the faces wouldn't buy much
    pub const LIT_FLOOR: u32 = 16384;

    /*
        the lod ladder, same for the geometry and the face light: full res up to LOD_START,
        then 2x2x2 voxel blocks, 4x4x4 from twice that, 8x8x8 (a whole meter) from 4x as far.
        camera rays get one voxel wider every LOD_START / 2 meters, so the traversal's
        floor(log2(width)) lands on exactly those steps
    */
    pub const LOD_START: f32 = 24.0;
    pub const CAMERA_CONE: f32 = 2.0 / Self::LOD_START;

    /// far water only traces every nth pixel's reflection (per lod level), every water pixel
    /// out there then blurs together the ones within REFLECTION_BLUR pixels of it
    pub const REFLECTION_STEP: [usize; 4] = [1, 2, 4, 4];
    pub const REFLECTION_BLUR: [usize; 4] = [0, 2, 4, 6];

    /// skip map hops a beam gets at most, and the shortest one still worth a lookup (meters)
    pub const BEAM_HOPS: usize = 16;
    pub const BEAM_MIN_HOP: f32 = 0.25;

    /// pixels along one side of a beam block, every packet in it starts at the block's beam_start
    pub const BEAM_PIXELS: usize = 8;

    /// samples a brand new face gets right away, one sample per face looks like confetti
    pub const FIRST_SAMPLES: u32 = 4;

    /// lod cones (voxels of width per meter traveled) for rays that start on a surface.
    /// sun shadows widen with the sun square anyway, bounce light is blurry to begin with
    pub const SHADOW_CONE: f32 = SUN_SIZE * VOXELS_PER_METER;
    pub const BOUNCE_CONE: f32 = 0.4;

    /// a bounce hit only borrows a face's cached light once it has this many samples
    pub const BORROW_SAMPLES: u32 = 4;

    pub fn resize_accumulation_buffer(&mut self, new_x: usize, new_y: usize) {
        self.frame_accumulation_buffer = vec![[0.0; 4]; new_x * new_y];
        self.frame_depths = vec![f32::INFINITY; new_x * new_y];
        self.frame_faces = vec![SKY_FACE; new_x * new_y];

        // the face light lives in the world, a new size doesn't change any of it //
        self.frame_accumulation_count = 0;
    }

    /// start the image over and forget all the face light too,
    /// for stuff that changes how the world looks (sun, lighting), not just where you are
    pub fn restart_image(&mut self) {
        self.frame_accumulation_count = 0;
        self.lights.clear();
    }

    /// traces one more frame into the pixel buffer
    /// true when every pixel got rewritten, false for a converged frame
    pub fn render(&mut self, buffer_address: usize) -> bool {
        /*
            ok so every pixel gets its own camera ray, so whatever you look at stays sharp,
            but the light comes out of the face light cache (lightcache.rs). the light on a
            voxel face is the same from wherever you see it, so it's kept per face in the world
            and only faces that need it get shaded, by whichever lane got to them first.
            moving the camera never throws any of it away

            moving: rays go through pixel centers, the image is just this frame
            still: rays get jittered and every pixel keeps a running mean of its color,
            that's the anti aliasing, the faces keep converging underneath it

            every 16x16 tile runs the stages in Tile on its own thread
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
        // packets stay coherent and the image anti aliases itself as frames pile up.
        // moving frames go straight through pixel centers, jitter there only makes edges shimmer

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
            // an eye inside water sees everything through it
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

        // size up the next frame from what this one cost: settled faces get refreshed //
        // less often when it's slow, more often when there's room (or barely anything got lit)

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

    /// what one packet of camera rays hits first, an eye under water looks right through it.
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

    /*
        how far every camera ray of a block can go before it could possibly hit anything
        (meters). origin + dir is the block's center ray, the block's other rays never get
        further than spread * t from it

        the center ray marches through the skip map, every meter knows the empty cube around it.
        as long as the whole fan of rays fits inside the cube it's safe to jump, so each hop
        goes until the fan's edge would touch a wall. the volume's sides count as walls too,
        every ray has to start somewhere inside it (above the terrain's top is fine, a ray
        up there just walks out through empty space)
    */
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

                // the fan's edge closes in on the wall ahead at speed + spread, //
                // and on the wall behind only if it spreads faster than the center leaves it
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

    /*
        what 4 camera rays end up looking at, everything that depends on where you look from:
        sky, water (reflection ray + refraction ray, mixed by fresnel) and fog under water.
        whatever surface is left gets its light from the face cache, see render
    */
    #[inline(always)]
    fn view(&self, ray: Packet, surface: PacketHit, underwater: bool, cone: f32) -> View {
        let tree = &self.world.tree;
        let sun = &self.world.sun;
        let offset = SimdF32::from_float(SURFACE_OFFSET);

        let mut surface = surface;
        let mut ray = ray;
        let mut throughput = Packed_Vec3_x4::ONES;
        let mut traveled = surface.distances;

        // rays that get out see the sky and the sun //

        let mut radiance = Packed_Vec3_x4::select(
            surface.hit,
            Packed_Vec3_x4::ZEROES,
            sky_with_sun(ray.directions, sun),
        );

        // under water everything's seen through the water //

        if underwater {
            throughput = water_transmission(surface.distances);
            radiance = (Packed_Vec3_x4::ONES - throughput) * water_scatter(sun);
        }

        /* WATER */

        let water = water_lanes(surface.materials) & surface.hit;
        let mut mirror_origins = Packed_Vec3_x4::ZEROES;
        let mut mirror_dirs = Packed_Vec3_x4::ZEROES;
        let mut mirror_share = SimdF32::ZEROES;

        if water.has_active() {
            // water seen from above is always its flat top, past LOD_START the surface gets
            // lumped into blocks and a ray could catch the side of one, that'd mirror like a wall
            let n = Packed_Vec3_x4::select(water, splat(Vec3::new(0.0, 1.0, 0.0)), surface.normals);
            let d = ray.directions;
            let p = ray.at(surface.distances);

            // schlick fresnel decides how the light splits between the two rays //

            let cos_i = d.dot(n).neg().clamp_to(SimdF32::ZEROES, SimdF32::ONES);
            let k = SimdF32::ONES - cos_i;
            let k2 = k * k;
            let fresnel =
                (k2 * k2 * k).mul_then_add(SimdF32::from_float(0.98), SimdF32::from_float(0.02));

            // the reflection ray gets traced later, far water doesn't need one for every pixel //

            let mirrored = d.add_scaled(n, d.dot(n) * SimdF32::from_float(-2.0));
            mirror_origins = p.add_scaled(n, offset);
            mirror_dirs = mirrored;
            mirror_share = fresnel;

            // refraction, snell from air into water, eta = 1 / 1.33 //
            // you see the bottom up to WATER_VISIBILITY, past that it's just water color

            let eta = SimdF32::from_float(1.0 / 1.33);
            let sin2 = (SimdF32::ONES - cos_i * cos_i) * eta * eta;
            let cos_t = (SimdF32::ONES - sin2).max(SimdF32::ZEROES).sqrt();
            let bent = d
                .scale(eta)
                .add_scaled(n, eta.mul_then_add(cos_i, cos_t.neg()))
                .normalize();

            let under_ray = Packet::new(
                Packed_Vec3_x4::select(water, p.add_scaled(n, offset.neg()), ray.origins),
                Packed_Vec3_x4::select(water, bent, ray.directions),
            );
            let under = tree.trace::<false, true>(
                under_ray,
                water,
                SimdF32::from_float(WATER_VISIBILITY),
                cone,
            );

            // whatever light the water eats gets swapped for its own scattered color //

            let transmission = water_transmission(under.distances);
            let scatter = water_scatter(sun);

            let refracted = SimdF32::ONES - fresnel;

            radiance = Packed_Vec3_x4::select(
                water,
                ((Packed_Vec3_x4::ONES - transmission) * scatter).scale(refracted),
                radiance,
            );
            throughput = Packed_Vec3_x4::select(water, transmission.scale(refracted), throughput);

            // from here on the surface to light is whatever the refracted ray found //

            ray = under_ray;
            traveled = SimdF32::select(water, traveled + under.distances, traveled);
            surface.hit = (surface.hit & water.not()) | under.hit;
            surface.distances = SimdF32::select(water, under.distances, surface.distances);
            surface.normals = Packed_Vec3_x4::select(water, under.normals, surface.normals);

            let water_array = water.to_array();
            let under_hits = under.hit.to_array();

            for (lane, material) in surface.materials.iter_mut().enumerate() {
                if water_array[lane] != 0 {
                    *material = if under_hits[lane] != 0 {
                        under.materials[lane]
                    } else {
                        Material::Air
                    };
                }
            }
        }

        let p = ray.at(surface.distances);

        View {
            radiance,
            base: albedo(surface.materials, p, surface.normals) * throughput,
            points: p,
            normals: surface.normals,
            distances: traveled,
            lit: surface.hit,
            water,
            fresnel: mirror_share,
            mirror_origins,
            mirror_dirs,
        }
    }

    /// what 4 water lanes see in their reflection, up to WATER_REFLECTION_DISTANCE, sky past that.
    /// whatever it hits shows the light its face has in the cache, a face that's
    /// never been on screen just gets the sun without a shadow ray
    #[inline(always)]
    fn reflect(&self, origins: Packed_Vec3_x4, dirs: Packed_Vec3_x4) -> Packed_Vec3_x4 {
        let sun = &self.world.sun;
        let ray = Packet::new(origins, dirs);
        let mirror = self.world.tree.trace::<false, true>(
            ray,
            SimdU32::ALL_ACTIVE,
            SimdF32::from_float(WATER_REFLECTION_DISTANCE),
            Self::CAMERA_CONE,
        );

        let n = mirror.normals;
        let p = ray.at(mirror.distances);
        let (cached, found) = self.cached_light(p, n, mirror.hit, 1);
        let guessed = splat(sun.rgb).scale(n.dot(splat(sun.dir)).max(SimdF32::ZEROES))
            + splat(BOUNCE_AMBIENT);

        Packed_Vec3_x4::select(
            mirror.hit,
            albedo(mirror.materials, p, n) * Packed_Vec3_x4::select(found, cached, guessed),
            sky_with_sun(dirs, sun),
        )
    }

    /// one light sample for 4 face spots (just off the surface): the sun with a shadow ray,
    /// plus one cosine bounce. it's the light landing there, the surface color isn't in it
    #[inline(always)]
    fn face_light(
        &self,
        start: Packed_Vec3_x4,
        n: Packed_Vec3_x4,
        rng: &mut Rng,
    ) -> Packed_Vec3_x4 {
        let sun = &self.world.sun;

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n_dot_l = n.dot(to_sun).max(SimdF32::ZEROES);
        let facing = n_dot_l.is_greater_than(SimdF32::ZEROES);

        let shadow = self.world.tree.trace::<true, true>(
            Packet::new(start, to_sun),
            facing,
            SimdF32::INFY,
            Self::SHADOW_CONE,
        );
        let sunlit = facing & shadow.hit.not();

        let direct = Packed_Vec3_x4::select(
            sunlit,
            splat(sun.rgb).scale(n_dot_l),
            Packed_Vec3_x4::ZEROES,
        );

        direct + self.bounce(start, n, rng)
    }

    /// light that one cosine bounce brings back, sky if it escapes.
    /// a hit on a face that's already in the cache just borrows its light (that's every
    /// bounce after the first, for free), anything else gets the sun with its own shadow ray
    #[inline(always)]
    fn bounce(&self, start: Packed_Vec3_x4, n: Packed_Vec3_x4, rng: &mut Rng) -> Packed_Vec3_x4 {
        let tree = &self.world.tree;
        let sun = &self.world.sun;

        let bounce_dirs = cosine_sample(n, rng.next(), rng.next());

        // bounce lighting off means nobody traces, everyone just sees open sky //

        if !self.bounce_lighting {
            return sky(bounce_dirs);
        }

        let bounce_ray = Packet::new(start, bounce_dirs);
        let bounce = tree.trace::<false, true>(
            bounce_ray,
            SimdU32::ALL_ACTIVE,
            SimdF32::from_float(Self::BOUNCE_MAX_DISTANCE),
            Self::BOUNCE_CONE,
        );

        // escaped bounces pick up the sky dome, the sun was already counted as direct light //

        let indirect = sky(bounce_dirs);

        if bounce.hit.has_none_active() {
            return indirect;
        }

        let n2 = bounce.normals;
        let p2 = bounce_ray.at(bounce.distances);
        let (cached, found) = self.cached_light(p2, n2, bounce.hit, Self::BORROW_SAMPLES);

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n2_dot_l = n2.dot(to_sun).max(SimdF32::ZEROES);
        let facing2 = bounce.hit & found.not() & n2_dot_l.is_greater_than(SimdF32::ZEROES);

        let shadow2 = tree.trace::<true, true>(
            Packet::new(
                p2.add_scaled(n2, SimdF32::from_float(SURFACE_OFFSET)),
                to_sun,
            ),
            facing2,
            SimdF32::from_float(Self::BOUNCE_SHADOW_DISTANCE),
            Self::BOUNCE_CONE,
        );
        let sunlit2 = facing2 & shadow2.hit.not();

        let guessed = Packed_Vec3_x4::select(
            sunlit2,
            splat(sun.rgb).scale(n2_dot_l),
            Packed_Vec3_x4::ZEROES,
        ) + splat(BOUNCE_AMBIENT);

        Packed_Vec3_x4::select(
            bounce.hit,
            albedo(bounce.materials, p2, n2) * Packed_Vec3_x4::select(found, cached, guessed),
            indirect,
        )
    }

    /// light the cache has for every hit lane's face (finest patch first),
    /// and the lanes that had at least min_samples of it
    #[inline(always)]
    fn cached_light(
        &self,
        points: Packed_Vec3_x4,
        normals: Packed_Vec3_x4,
        hit: SimdU32,
        min_samples: u32,
    ) -> (Packed_Vec3_x4, SimdU32) {
        let hits = hit.to_array();
        let points = lanes(points);
        let normals = lanes(normals);
        let mut found = [0u32; 4];

        let light = [0, 1, 2, 3].map(|lane| {
            if hits[lane] == 0 {
                return [0.0; 3];
            }

            let (voxel, face) = face_of(points[lane], normals[lane]);

            for level in 0..4 {
                if let Some((light, samples)) =
                    self.lights.lookup(LightCache::key(voxel, face, level))
                    && samples >= min_samples
                {
                    found[lane] = u32::MAX;
                    return light;
                }
            }

            [0.0; 3]
        });

        (packed(light), SimdU32::from(found))
    }
}

/// what every tile needs to know about the frame
struct Setup {
    camera: Camera,
    builder: PacketBuilder,
    /// sub pixel offset of every camera ray this frame (pixels)
    jitter: (f32, f32),
    underwater: bool,
    moving: bool,
    frame: u32,
    /// settled faces get shaded every nth frame, see LightCache::claim
    every: u32,
    /// render width (pixels)
    width: usize,
    /// the per pixel buffers and the output, tiles never overlap so every pixel
    /// is only ever written by one thread
    accumulation: usize,
    depths: usize,
    faces: usize,
    pixels: usize,
}

/// one 16x16 tile, everything on the stack, one thread per tile at a time
struct Tile {
    /// top left pixel of the tile
    x0: usize,
    y0: usize,

    /// what every packet of camera rays sees
    views: [View; TILE_PACKETS],
    /// face (or patch) key every lit lane sees, see LightCache::key
    keys: [[u64; 4]; TILE_PACKETS],

    /// lod level of the water every pixel sees, NO_WATER if it doesn't
    water_levels: [[u8; TILE]; TILE],
    /// reflections of the pixels that traced one
    reflections: [[[f32; 3]; TILE]; TILE],
    traced: [[bool; TILE]; TILE],

    /// face samples waiting to be shaded
    jobs: [Job; TILE_JOBS],
    queued: usize,

    /// faces nobody had light for yet that another tile is busy shading,
    /// this tile shades them for itself: (key, light sum, samples)
    privates: [(u64, [f32; 3], f32); TILE_PRIVATES],
    private_count: usize,
    /// new faces this tile claimed, its own pixels can just wait on those
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

    /// every packet's camera rays, what they hit and the water in front of it,
    /// plus every pixel's view depth and face for the brush preview
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

    /// where every camera ray of the 8x8 block starting at packet k can start walking,
    /// see Engine::beam_start
    fn block_start(&self, engine: &Engine, setup: &Setup, k: usize) -> f32 {
        let camera = &setup.camera;

        // the ray through a pixel (x, y), plus jitter, normalized //
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

    /// near water traces a reflection for every pixel, further out only every
    /// REFLECTION_STEP-th pixel does and the rest blur together the ones around them
    /// (see reflection). far pixels with none around them (shores) trace their own
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

    /// reflection rays for the queued (packet, lane)s, 4 at a time
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

    /// what a water pixel shows in its reflection: up close its own, further out the traced
    /// ones on its grid within REFLECTION_BLUR, tent weighted so it fades out smoothly
    /// (blurrier the further out). None = not water, or nothing traced around it yet
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
            // a shore pixel that had to trace its own //
            self.traced[ty][tx].then_some(self.reflections[ty][tx])
        }
    }

    /* FACES */

    /// which face (or patch) each lit lane sees, and who has to shade what.
    /// neighbors mostly see the same face, those don't even ask
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

    /// every queued face sample, 4 at a time so no simd lane sits around idle
    fn shade(&mut self, engine: &Engine, setup: &Setup) {
        for group in self.jobs[..self.queued].chunks(4) {
            // short groups just repeat their last job //
            let pick = [0, 1, 2, 3].map(|j| group[j.min(group.len() - 1)]);

            let mut rng = Rng::new(SimdU32::from(pick.map(|job| job.seed)), setup.frame);
            let (u1, u2) = (rng.next().to_array(), rng.next().to_array());

            // a random spot on the face, just off it //

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

    /// every pixel's surface times its face light, plus whatever got there without
    /// touching a lit surface and any reflection. still frames keep a running mean,
    /// then the whole tile gets tone mapped into the pixel buffer
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

                // water gets its reflection on top //
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

        // tone map the tile's means straight out of the accumulation buffer, 4 at a time //

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
fn face_id(n: [f32; 3], hit: bool) -> u8 {
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

/// the voxel a hit point (meters, on the face) belongs to and which face of it,
/// half a voxel back against the normal lands inside the voxel
#[inline(always)]
fn face_of(p: [f32; 3], n: [f32; 3]) -> ([u32; 3], u8) {
    let voxel = [0, 1, 2].map(|a| {
        ((p[a] - n[a] * 0.5 * VOXEL_SIZE) * VOXELS_PER_METER)
            .floor()
            .clamp(0.0, 2047.0) as u32
    });

    (voxel, face_id(n, true))
}

/// rung of a ladder something this far away (meters) is on, 0 = full res up to start,
/// then 1, 2, 3 each twice as far as the last. light patches climb it from LOD_START
/// (2^level voxels wide), the reflection grid from WATER_REFLECTION_DISTANCE
#[inline(always)]
fn ladder(distance: f32, start: f32) -> u32 {
    (distance >= start) as u32 + (distance >= start * 2.0) as u32 + (distance >= start * 4.0) as u32
}

/// a packet split into its 4 [x, y, z] lanes
#[inline(always)]
fn lanes(v: Packed_Vec3_x4) -> [[f32; 3]; 4] {
    let (x, y, z) = (v.xs.to_array(), v.ys.to_array(), v.zs.to_array());
    [0, 1, 2, 3].map(|i| [x[i], y[i], z[i]])
}

/// 4 [x, y, z] lanes packed back up
#[inline(always)]
fn packed(lanes: [[f32; 3]; 4]) -> Packed_Vec3_x4 {
    Packed_Vec3_x4::new(
        lanes.map(|v| v[0]),
        lanes.map(|v| v[1]),
        lanes.map(|v| v[2]),
    )
}

/// top left pixel of the k-th packet in its tile (relative), packets go in morton order:
/// bits 0, 2, 4 of k are x, bits 1, 3, 5 are y, in 2 pixel steps
#[inline(always)]
const fn packet_xy(k: usize) -> (usize, usize) {
    let x = (k & 1) | ((k >> 1) & 2) | ((k >> 2) & 4);
    let y = ((k >> 1) & 1) | ((k >> 2) & 2) | ((k >> 3) & 4);
    (x * 2, y * 2)
}

/// pixel (relative to its tile) of a packet's lane, lanes go left right, top bottom
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

        // cameras all over, high up and skimming the ground, looking every which way //

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

        // and it actually skips something //
        println!(
            "{packets} packets, {:.1}m skipped on average",
            skipped / packets as f32
        );
        assert!(skipped / packets as f32 > 1.0);
    }

    #[test]
    fn test_faces_and_patches() {
        // a point on the top face of voxel (10, 20, 30) //

        let top = [
            (10.0 + 0.3) * VOXEL_SIZE,
            21.0 * VOXEL_SIZE,
            (30.0 + 0.7) * VOXEL_SIZE,
        ];
        assert_eq!(face_of(top, [0.0, 1.0, 0.0]), ([10, 20, 30], 3));

        // same spot seen from below is the voxel above's bottom face //
        assert_eq!(face_of(top, [0.0, -1.0, 0.0]), ([10, 21, 30], 2));

        // the lod ladder: full res up close, then twice as coarse every time the distance doubles //

        let lod = |d: f32| ladder(d, Engine::LOD_START);
        assert_eq!(lod(0.5), 0);
        assert_eq!(lod(23.9), 0);
        assert_eq!(lod(24.0), 1);
        assert_eq!(lod(48.0), 2);
        assert_eq!(lod(95.0), 2);
        assert_eq!(lod(96.0), 3);
        assert_eq!(lod(1000.0), 3);

        // reflections keep their own ladder, full res out to where they stop //

        assert_eq!(ladder(31.9, WATER_REFLECTION_DISTANCE), 0);
        assert_eq!(ladder(32.0, WATER_REFLECTION_DISTANCE), 1);
    }
}
