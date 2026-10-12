use super::shading::*;
use super::*;
use crate::simd::packetray::{Packet, PacketBuilder};
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::sun::SUN_SIZE;
use crate::world::tree64::PacketHit;
use crate::world::voxel::{Material, VOXELS_PER_METER};

/// one packet of surfaces after any water and the direct sun,
/// the bounce light gets added on top later by compose
#[derive(Clone, Copy)]
struct Direct {
    /// light that made it to the camera without a diffuse bounce (sky, water, reflections)
    radiance: Packed_Vec3_x4,
    /// how much the lit surface reflects, already dimmed by any water in front of it
    base: Packed_Vec3_x4,
    /// sun light landing on the lit lanes
    incoming: Packed_Vec3_x4,
    /// bounce rays start here, nudged just off the surface
    start: Packed_Vec3_x4,
    normals: Packed_Vec3_x4,
    /// lanes that actually hit a surface worth lighting
    lit: SimdU32,
    /// lit lanes seen through water on a moving frame, they skip the bounce
    through_water: SimdU32,
}

/// pixels along one side of a tile
const TILE: usize = Engine::PIXELS_PER_TILE_AXIS;

/// packets in a tile at full res (2x2 pixels each)
const TILE_PACKETS: usize = (TILE / 2) * (TILE / 2);

impl Engine {
    /// once a still image has this many samples per pixel we just stop tracing
    pub const MAX_ACCUMULATED_FRAMES: u32 = 512;

    /// bounce rays that get further than this (meters) just count as open sky
    pub const BOUNCE_MAX_DISTANCE: f32 = 48.0;

    /// shadow rays from bounce hits give up here (meters), far blockers barely matter for bounce light
    pub const BOUNCE_SHADOW_DISTANCE: f32 = 64.0;

    /// frame budgets for the fps target setting (ms), infinity = no target at all
    pub const FRAME_BUDGETS: [f32; 3] = [1000.0 / 60.0, 1000.0 / 30.0, f32::INFINITY];

    /// how many pixels wide a cell can get on screen before camera rays stop looking inside it,
    /// 1 means a 2x2x2 block only gets lumped together once it's no bigger than a pixel
    pub const LOD_PIXELS: f32 = 1.0;

    /// lod cones (voxels of width per meter traveled) for rays that start on a surface.
    /// sun shadows widen with the sun square anyway, bounce light is blurry to begin with
    pub const SHADOW_CONE: f32 = SUN_SIZE * VOXELS_PER_METER;
    pub const BOUNCE_CONE: f32 = 0.4;

    /// share of the budget a moving frame aims for
    pub const MOVING_SHARE: f32 = 0.85;

    /// moving frame quality, best first: (shade every nth pixel that already has history,
    /// rays per block side in far tiles). the frame budget walks up and down this list
    pub const MOVING_STEPS: [(usize, usize); 7] =
        [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (32, 2)];

    /// tiles where everything was at least this far last frame (meters) are allowed
    /// coarser rays, up close always gets one ray per pixel
    pub const FAR_DISTANCE: f32 = 32.0;

    /// most samples a pixel's history keeps while moving, more means less noise
    /// but slower to let go of stuff (like sliding past a voxel edge)
    pub const MOVING_HISTORY: f32 = 32.0;

    /// pixels with fewer light samples than this get their light smoothed with same face
    /// neighbors on screen (display only, history never sees it)
    pub const YOUNG_SAMPLES: f32 = 4.0;

    /// water reflects whatever the view is, so its history barely gets trusted
    pub const WATER_HISTORY: f32 = 2.0;

    /// history only counts when last frame saw the same spot at about the same view depth:
    /// within this share of the depth, or this many meters up close. plenty to tell a tree
    /// from the ground behind it, loose enough for a block's guessed spots to still match
    pub const DEPTH_TOLERANCE: f32 = 0.05;
    pub const DEPTH_SLACK: f32 = 0.25;

    /// share of the budget a still frame aims for, the rest goes to ui and present
    pub const STILL_SHARE: f32 = 0.85;

    /// most slices a still frame ever gets split into
    pub const MAX_SLICES: usize = 16;

    pub fn resize_accumulation_buffer(&mut self, new_x: usize, new_y: usize) {
        self.frame_accumulation_buffer = vec![[0.0; 4]; new_x * new_y];
        self.history = vec![[0.0; 4]; new_x * new_y];
        self.frame_depths = vec![f32::INFINITY; new_x * new_y];
        self.history_depths = vec![f32::INFINITY; new_x * new_y];
        self.frame_faces = vec![SKY_FACE; new_x * new_y];
        self.history_faces = vec![SKY_FACE; new_x * new_y];
        self.frame_albedo = vec![[0.0; 4]; new_x * new_y];
        self.restart_image();
    }

    /// start the image over and forget the history too,
    /// for stuff that changes how the world looks (sun, lighting, size), not just where you are
    pub fn restart_image(&mut self) {
        self.frame_accumulation_count = 0;
        self.history_valid = false;
    }

    /// traces one more sample into the pixel buffer
    /// true when every pixel got rewritten, false for a converged or sliced frame
    pub fn render(&mut self, buffer_address: usize) -> bool {
        /*
            ok so there's two kinds of frames:

            moving (the camera just moved): the old image gets swapped into history.
            every pixel still gets its own camera ray, so whatever you look at stays sharp,
            but lighting it is the expensive part. the world is static and the light is
            diffuse, so a spot that was on screen last frame already has its light, the
            pixel just reprojects it from history. only freshly revealed spots and a
            rotating 1 in n of the rest get shaded again, those keep adding samples so
            the image gets cleaner the longer you move. far away tiles (32m+) may also
            trace one ray per 2x2 block if the budget is really tight.
            moving counts get stored negative, they're a smudged estimate

            still: keep adding samples in place, but each frame only traces 1 of
            `slices` sets of tiles (rotating), so a frame stays in budget no matter
            how slow a full sample is. every pixel keeps its own sample count in the 4th float,
            and a moving estimate only counts as one sample once you stop
        */

        let moving = self.frame_accumulation_count == 0;

        // a finished image just stays in the pixel buffer, nothing to do //

        if !moving && self.samples >= Self::MAX_ACCUMULATED_FRAMES as f32 {
            return false;
        }

        let start_time = std::time::Instant::now();

        self.camera.update();

        // an eye inside water sees everything through it //
        let underwater = self.world.water_at(self.camera.center);

        let budget = Self::FRAME_BUDGETS[self.fps_target];
        let slices = if moving || !budget.is_finite() {
            1
        } else {
            self.still_slices
        };

        let slice = self.frame_index as usize % slices;

        if moving {
            self.samples = 0.0;

            // last frame becomes the history, every pixel gets rewritten below //
            std::mem::swap(&mut self.frame_accumulation_buffer, &mut self.history);
            std::mem::swap(&mut self.frame_depths, &mut self.history_depths);
            std::mem::swap(&mut self.frame_faces, &mut self.history_faces);
        } else {
            self.samples += 1.0 / slices as f32;
        }

        let (refresh, far_scale) = if moving && budget.is_finite() {
            Self::MOVING_STEPS[self.moving_step]
        } else {
            (1, 1)
        };

        let camera = self.camera;
        let width = self.render_width;
        let past = Reprojection::new(&self.last_camera);
        let history_valid = self.history_valid;

        let tile_xcount = width / TILE;
        let tile_ycount = self.render_height / TILE;

        let max_dim = tile_xcount.max(tile_ycount).next_power_of_two();
        let total_morton_tiles = max_dim * max_dim;

        let frame = self.frame_index;

        self.frame_accumulation_count += 1;
        self.frame_index = self.frame_index.wrapping_add(1);

        // sub pixel jitter from the r2 sequence, same for the whole frame //
        // packets stay coherent and the image anti aliases itself as samples pile up

        // moving frames go straight through pixel centers, jitter there only makes edges shimmer //

        let jitter = if moving {
            (0.5, 0.5)
        } else {
            (
                (0.5 + frame as f32 * 0.754_877_7).fract(),
                (0.5 + frame as f32 * 0.569_840_3).fract(),
            )
        };

        let accumulation = self.frame_accumulation_buffer.as_mut_ptr() as usize;
        let depths = self.frame_depths.as_mut_ptr() as usize;
        let faces = self.frame_faces.as_mut_ptr() as usize;
        let albedos = self.frame_albedo.as_mut_ptr() as usize;
        let (history, history_depths, history_faces) =
            (&self.history, &self.history_depths, &self.history_faces);

        (0..total_morton_tiles)
            .into_par_iter()
            .filter_map(|i| {
                // neighboring morton tiles land in different slices, so every slice covers the whole screen //
                if i % slices != slice {
                    return None;
                }

                let (tile_x, mut tile_y) = morton_decode(i);
                if tile_x >= tile_xcount || tile_y >= tile_ycount {
                    return None;
                }
                tile_y = tile_ycount - 1 - tile_y;
                Some((tile_x, tile_y))
            })
            .for_each(|(tile_x, tile_y)| {
                // convert tiles to pixel space
                // corresponds to the top left pixel of the tile
                let tile_pixel00_x = tile_x * TILE;
                let tile_pixel00_y = tile_y * TILE;

                // far tiles may go coarse, only if everything in them was far last frame too //

                let far = history_valid
                    && (0..TILE).all(|y| {
                        let row = (tile_pixel00_y + y) * width + tile_pixel00_x;
                        history_depths[row..row + TILE]
                            .iter()
                            .all(|&d| d >= Self::FAR_DISTANCE)
                    });

                let scale = if moving && far { far_scale } else { 1 };
                let packet_count = (TILE / (2 * scale)).pow(2);

                // camera rays get about one (traced) pixel wider per meter, that's their lod cone //
                let cone = camera.pixel_delta_y.length()
                    * VOXELS_PER_METER
                    * scale as f32
                    * Self::LOD_PIXELS;

                let jitter_x = jitter.0 * scale as f32 - 0.5;
                let jitter_y = jitter.1 * scale as f32 - 0.5;

                let packet_builder = PacketBuilder::new(
                    camera.center,
                    camera.pixel_delta_x * scale as f32,
                    camera.pixel_delta_y * scale as f32,
                );

                /* PRIMARY */

                // every packet's camera rays, what they hit, and the top left pixel of each lane's block //

                let mut packets =
                    [Packet::new(Packed_Vec3_x4::ZEROES, Packed_Vec3_x4::ONES); TILE_PACKETS];
                let mut hits = [PacketHit::MISS; TILE_PACKETS];
                let mut corners = [[(0usize, 0usize); 4]; TILE_PACKETS];

                for k in 0..packet_count {
                    // the top left pixel of the packet, relative to the tile
                    let (rel_x, rel_y) = morton_to_xy(k);

                    // top left x and y globally
                    let packet_pixel_x = tile_pixel00_x + rel_x * scale;
                    let packet_pixel_y = tile_pixel00_y + rel_y * scale;

                    // packet top left coord as vec3
                    let packet00_coord = camera.pixel_00
                        + (camera.pixel_delta_x * (packet_pixel_x as f32 + jitter_x))
                        - (camera.pixel_delta_y * (packet_pixel_y as f32 + jitter_y));

                    packets[k] = packet_builder.make(packet00_coord);
                    hits[k] = self.primary(packets[k], underwater, cone);
                    corners[k] = [0, 1, 2, 3].map(|lane| {
                        (
                            packet_pixel_x + (lane & 1) * scale,
                            packet_pixel_y + (lane >> 1) * scale,
                        )
                    });
                }

                /* HISTORY */

                // every pixel's own spot on the world and what history it can keep, //
                // a lane needs light if any of its pixels has no history, or it's its turn

                let mut kept = [[None::<[f32; 4]>; TILE]; TILE];
                let mut needs = [[!moving; 4]; TILE_PACKETS];

                // each lane's surface color, light gets stored without it (see ACCUMULATE), //
                // and every pixel's own surface color, they only differ in coarse far tiles
                let mut colors = [[[1.0f32; 3]; 4]; TILE_PACKETS];
                let mut tints = [[[1.0f32; 3]; TILE]; TILE];

                for k in 0..packet_count {
                    let dirs = lanes(packets[k].directions);
                    let normals = lanes(hits[k].normals);
                    let ts = hits[k].distances.to_array();
                    let hit = hits[k].hit.to_array();
                    let surface = lanes(albedo(
                        hits[k].materials,
                        packets[k].at(hits[k].distances),
                        hits[k].normals,
                    ));

                    for lane in 0..4 {
                        let (corner_x, corner_y) = corners[k][lane];
                        let hit = hit[lane] != 0;
                        let water = underwater || hits[k].materials[lane] == Material::Water;
                        let face = Vec3 {
                            data: normals[lane],
                        };
                        let face_id = face_id(normals[lane], hit);

                        // sky and water aren't a plain surface color times light, they keep 1 //
                        if hit && !water {
                            colors[k][lane] = surface[lane].map(|c| c.max(1e-3));
                        }
                        let on_face = camera.center + Vec3 { data: dirs[lane] } * ts[lane];

                        // water looks different from every angle, so it gets fresh light every other frame, //
                        // and up close gets it twice as often as far away, that's where noise shows

                        let every = if water {
                            refresh.min(2)
                        } else if ts[lane] < Self::FAR_DISTANCE {
                            (refresh / 2).max(1)
                        } else {
                            refresh
                        };
                        needs[k][lane] |=
                            (corner_x * 7 + corner_y * 13 + frame as usize).is_multiple_of(every);

                        let mut block_points = [on_face.data; 4];

                        for block in 0..scale * scale {
                            let x = corner_x + block % scale;
                            let y = corner_y + block / scale;

                            // this pixel's own spot on the voxel face the lane hit,
                            // faces are flat so that's spot on as long as it's the same face
                            let ray = camera.pixel_00 + camera.pixel_delta_x * x as f32
                                - camera.pixel_delta_y * y as f32
                                - camera.center;
                            let along = ray.dot(face);

                            let point = if hit && along < -1e-6 {
                                camera.center + ray * ((on_face - camera.center).dot(face) / along)
                            } else {
                                on_face
                            };

                            block_points[block.min(3)] = point.data;
                            tints[y - tile_pixel00_y][x - tile_pixel00_x] = colors[k][lane];

                            // view depth (along front) is all the occlusion test needs, no sqrt //
                            let depth = if hit {
                                (point - camera.center).dot(camera.front)
                            } else {
                                f32::INFINITY
                            };

                            // tiles never overlap, so every pixel is written by one thread
                            unsafe {
                                *(depths as *mut f32).add(y * width + x) = depth;
                                *(faces as *mut u8).add(y * width + x) = face_id;
                            }

                            if moving {
                                let past_sum = (hit && history_valid)
                                    .then(|| {
                                        past.fetch(
                                            point,
                                            face_id,
                                            history,
                                            history_depths,
                                            history_faces,
                                        )
                                    })
                                    .flatten();

                                needs[k][lane] |= past_sum.is_none();
                                kept[y - tile_pixel00_y][x - tile_pixel00_x] = past_sum;
                            }
                        }

                        // a coarse 2x2 block still gets every pixel's own texture, one simd call //

                        if scale == 2 && hit && !water {
                            let block_colors = lanes(albedo(
                                [hits[k].materials[lane]; 4],
                                packed(block_points),
                                Packed_Vec3_x4::from_single(face),
                            ));

                            for (block, color) in block_colors.iter().enumerate() {
                                tints[corner_y + block / 2 - tile_pixel00_y]
                                    [corner_x + block % 2 - tile_pixel00_x] =
                                    color.map(|c| c.max(1e-3));
                            }
                        }
                    }
                }

                /* SHADE */

                // the lanes that need light go out 4 at a time, so no simd lane sits around idle //

                let mut queue = [(0usize, 0usize); TILE_PACKETS * 4];
                let mut queued = 0;
                let mut samples = [[[0.0f32; 3]; 4]; TILE_PACKETS];

                for k in 0..packet_count {
                    let hit = hits[k].hit.to_array();

                    // sky is just a color, it never needs a trip through the shading //

                    if !underwater && hit.contains(&0) {
                        let sky = lanes(sky_with_sun(packets[k].directions, &self.world.sun));

                        for lane in 0..4 {
                            if hit[lane] == 0 {
                                samples[k][lane] = sky[lane];
                            }
                        }
                    }

                    for lane in 0..4 {
                        if needs[k][lane] && (underwater || hit[lane] != 0) {
                            queue[queued] = (k, lane);
                            queued += 1;
                        }
                    }
                }

                for group in queue[..queued].chunks(4) {
                    // short groups just repeat their last lane //
                    let pick = [0, 1, 2, 3].map(|j| group[j.min(group.len() - 1)]);

                    let ray = Packet::new(
                        Packed_Vec3_x4::from_single(camera.center),
                        packed(pick.map(|(k, lane)| lanes(packets[k].directions)[lane])),
                    );
                    let surface = PacketHit {
                        hit: SimdU32::from(pick.map(|(k, lane)| hits[k].hit.to_array()[lane])),
                        distances: SimdF32::from(
                            pick.map(|(k, lane)| hits[k].distances.to_array()[lane]),
                        ),
                        normals: packed(pick.map(|(k, lane)| lanes(hits[k].normals)[lane])),
                        materials: pick.map(|(k, lane)| hits[k].materials[lane]),
                    };

                    let mut rng = Rng::new(
                        SimdU32::from(pick.map(|(k, lane)| {
                            let (x, y) = corners[k][lane];
                            (y * width + x) as u32
                        })),
                        frame,
                    );

                    let direct = self.shade(ray, surface, underwater, moving, cone, &mut rng);
                    let indirect = self.bounce(
                        direct.start,
                        direct.normals,
                        direct.lit & direct.through_water.not(),
                        &mut rng,
                    );
                    let radiance = lanes(compose(&direct, indirect));

                    for (j, &(k, lane)) in group.iter().enumerate() {
                        samples[k][lane] = radiance[j];
                    }
                }

                /* ACCUMULATE */

                // every pixel's mean light, how many samples that is, and its mean surface color //
                let mut lights = [[[0.0f32; 3]; TILE]; TILE];
                let mut counts = [[0.0f32; TILE]; TILE];
                let mut surfaces = [[[0.0f32; 3]; TILE]; TILE];

                for k in 0..packet_count {
                    for lane in 0..4 {
                        let (corner_x, corner_y) = corners[k][lane];
                        let water = underwater || hits[k].materials[lane] == Material::Water;
                        let cap = if water {
                            Self::WATER_HISTORY
                        } else {
                            Self::MOVING_HISTORY
                        };

                        // history holds just the light, the surface color gets put back on fresh
                        // every frame, so textures and voxel edges never smear while moving

                        let color = colors[k][lane];
                        let sample = samples[k][lane];
                        let light = [0, 1, 2].map(|c| sample[c] / color[c]);

                        for block in 0..scale * scale {
                            let x = corner_x + block % scale;
                            let y = corner_y + block / scale;
                            let (tx, ty) = (x - tile_pixel00_x, y - tile_pixel00_y);

                            // tiles never overlap, so every pixel is written by one thread
                            let sum =
                                unsafe { &mut *(accumulation as *mut [f32; 4]).add(y * width + x) };
                            let tint =
                                unsafe { &mut *(albedos as *mut [f32; 4]).add(y * width + x) };

                            *sum = match (moving, needs[k][lane]) {
                                (true, true) => blend(kept[ty][tx], light, cap),
                                // reused, just its reprojected history //
                                (true, false) => kept[ty][tx].unwrap_or([0.0; 4]),
                                (false, _) => still(*sum, light),
                            };

                            // moving frames take the color as is, still ones average it (anti aliasing) //
                            let own = tints[ty][tx];
                            *tint = if moving || tint[3] == 0.0 {
                                [own[0], own[1], own[2], 1.0]
                            } else {
                                [
                                    tint[0] + own[0],
                                    tint[1] + own[1],
                                    tint[2] + own[2],
                                    tint[3] + 1.0,
                                ]
                            };

                            let count = sum[3].abs().max(1.0);
                            lights[ty][tx] = [sum[0] / count, sum[1] / count, sum[2] / count];
                            counts[ty][tx] = count;
                            surfaces[ty][tx] = [0, 1, 2].map(|c| tint[c] / tint[3].max(1.0));
                        }
                    }
                }

                /* DISPLAY */

                // young pixels (just a few light samples) borrow light from neighbors on the same //
                // face at the same depth. it's only light, so texture and edges stay crisp,
                // and it never goes back into history so nothing smears over time

                let at = |tx: usize, ty: usize| (tile_pixel00_y + ty) * width + tile_pixel00_x + tx;
                let face_at =
                    |tx: usize, ty: usize| unsafe { *(faces as *const u8).add(at(tx, ty)) };
                let depth_at =
                    |tx: usize, ty: usize| unsafe { *(depths as *const f32).add(at(tx, ty)) };

                let mut means = [[[0.0f32; 3]; TILE]; TILE];

                for ty in 0..TILE {
                    for tx in 0..TILE {
                        let mut light = lights[ty][tx];
                        let face = face_at(tx, ty);

                        if counts[ty][tx] < Self::YOUNG_SAMPLES && face != SKY_FACE {
                            let depth = depth_at(tx, ty);
                            let close = (depth * Self::DEPTH_TOLERANCE).max(Self::DEPTH_SLACK);
                            let mut total = [0.0f32; 3];
                            let mut weight = 0.0;

                            for ny in ty.saturating_sub(1)..(ty + 2).min(TILE) {
                                for nx in tx.saturating_sub(1)..(tx + 2).min(TILE) {
                                    if face_at(nx, ny) != face
                                        || (depth_at(nx, ny) - depth).abs() > close
                                    {
                                        continue;
                                    }

                                    let w = counts[ny][nx];
                                    total =
                                        [0, 1, 2].map(|c| lights[ny][nx][c].mul_add(w, total[c]));
                                    weight += w;
                                }
                            }

                            light = total.map(|c| c / weight);
                        }

                        means[ty][tx] = [0, 1, 2].map(|c| light[c] * surfaces[ty][tx][c]);
                    }
                }

                let pointer = buffer_address as *mut u32;

                for (y, row) in means.iter().enumerate() {
                    let index = (tile_pixel00_y + y) * width + tile_pixel00_x;

                    for (x, quad) in row.chunks_exact(4).enumerate() {
                        let colors = display(packed([quad[0], quad[1], quad[2], quad[3]]));

                        // tiles never overlap, so every pixel is written by one thread //
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                colors.to_array().as_ptr(),
                                pointer.add(index + x * 4),
                                4,
                            );
                        }
                    }
                }
            });

        self.last_camera = camera;
        self.history_valid = true;

        // size up the next frame from what this one cost //

        let ms = start_time.elapsed().as_secs_f32() * 1000.0;

        if moving && budget.is_finite() {
            // walk the moving quality list one step at a time, down only with room to spare //

            let limit = budget * Self::MOVING_SHARE;

            if ms > limit {
                self.moving_step = (self.moving_step + 1).min(Self::MOVING_STEPS.len() - 1);
            } else if ms < limit * 0.6 {
                self.moving_step = self.moving_step.saturating_sub(1);
            }
        } else if budget.is_finite() {
            let full_ms = ms * slices as f32;

            self.still_slices = ((full_ms / (budget * Self::STILL_SHARE)).ceil() as usize)
                .next_power_of_two()
                .min(Self::MAX_SLICES);
        }

        slices == 1
    }

    /// redraw the pixel buffer straight from the accumulation buffer, tone mapping only
    pub fn resolve(&self, buffer_address: usize) {
        // rows don't care about each other, 4 pixels per display call //

        self.frame_accumulation_buffer
            .par_chunks(self.render_width)
            .enumerate()
            .for_each(|(y, row)| {
                let pointer = buffer_address as *mut u32;

                for (i, sums) in row.chunks_exact(4).enumerate() {
                    let tints = &self.frame_albedo[y * self.render_width + i * 4..][..4];
                    let means = [0, 1, 2, 3].map(|p| mean_color(sums[p], tints[p]));
                    let colors = display(packed(means)).to_array();

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

    /// what one packet of camera rays hits first, an eye under water looks right through it
    #[inline(always)]
    fn primary(&self, packet: Packet, underwater: bool, cone: f32) -> PacketHit {
        let tree = &self.world.tree;

        if underwater {
            tree.trace::<false, true>(
                packet,
                SimdU32::ALL_ACTIVE,
                SimdF32::from_float(WATER_VISIBILITY),
                cone,
            )
        } else {
            tree.trace::<false, false>(packet, SimdU32::ALL_ACTIVE, SimdF32::INFY, cone)
        }
    }

    /*
        one sample of light for 4 camera rays and whatever they hit first

        (water: reflection ray + refraction ray, mixed by fresnel)
        -> sun shadow ray, then bounce adds one cosine bounce with its own sun shadow ray
    */
    #[inline(always)]
    fn shade(
        &self,
        ray: Packet,
        surface: PacketHit,
        underwater: bool,
        moving: bool,
        cone: f32,
        rng: &mut Rng,
    ) -> Direct {
        let tree = &self.world.tree;
        let sun = &self.world.sun;
        let unbounded = SimdF32::INFY;
        let offset = SimdF32::from_float(SURFACE_OFFSET);

        let mut surface = surface;
        let mut ray = ray;
        let mut throughput = Packed_Vec3_x4::ONES;

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
        let seen_through = water;

        if water.has_active() {
            let n = surface.normals;
            let d = ray.directions;
            let p = ray.at(surface.distances);

            // schlick fresnel decides how the light splits between the two rays //

            let cos_i = d.dot(n).neg().clamp_to(SimdF32::ZEROES, SimdF32::ONES);
            let k = SimdF32::ONES - cos_i;
            let k2 = k * k;
            let fresnel =
                (k2 * k2 * k).mul_then_add(SimdF32::from_float(0.98), SimdF32::from_float(0.02));

            // reflection, whatever it hits gets the sun without a shadow ray, keeps it to one ray //

            let mirrored = d.add_scaled(n, d.dot(n) * SimdF32::from_float(-2.0));
            let mirror_ray = Packet::new(p.add_scaled(n, offset), mirrored);
            let mirror = tree.trace::<false, true>(
                mirror_ray,
                water,
                SimdF32::from_float(WATER_REFLECTION_DISTANCE),
                cone,
            );

            let mirror_n = mirror.normals;
            let mirror_light = splat(sun.rgb)
                .scale(mirror_n.dot(splat(sun.dir)).max(SimdF32::ZEROES))
                + splat(BOUNCE_AMBIENT);
            let reflected = Packed_Vec3_x4::select(
                mirror.hit,
                albedo(mirror.materials, mirror_ray.at(mirror.distances), mirror_n) * mirror_light,
                sky_with_sun(mirrored, sun),
            );

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
                reflected.scale(fresnel)
                    + ((Packed_Vec3_x4::ONES - transmission) * scatter).scale(refracted),
                radiance,
            );
            throughput = Packed_Vec3_x4::select(water, transmission.scale(refracted), throughput);

            // from here on the surface to light is whatever the refracted ray found //

            ray = under_ray;
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

        let lit = surface.hit;
        let n = surface.normals;
        let p = ray.at(surface.distances);
        let start = p.add_scaled(n, offset);

        if lit.has_none_active() {
            return Direct {
                radiance,
                base: Packed_Vec3_x4::ZEROES,
                incoming: Packed_Vec3_x4::ZEROES,
                start,
                normals: n,
                lit,
                through_water: SimdU32::ALL_INACTIVE,
            };
        }

        /* DIRECT */

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n_dot_l = n.dot(to_sun).max(SimdF32::ZEROES);
        let facing = lit & n_dot_l.is_greater_than(SimdF32::ZEROES);

        // while moving, whatever's seen through water skips its shadow and bounce rays, //
        // the water dims and tints it anyway, a flat ambient stands in and still frames fix it up

        let through_water = if !moving {
            SimdU32::ALL_INACTIVE
        } else if underwater {
            lit
        } else {
            seen_through & lit
        };

        let shadow = tree.trace::<true, true>(
            Packet::new(start, to_sun),
            facing & through_water.not(),
            unbounded,
            Self::SHADOW_CONE,
        );
        let sunlit = facing & shadow.hit.not();

        let incoming = Packed_Vec3_x4::select(
            sunlit,
            splat(sun.rgb).scale(n_dot_l),
            Packed_Vec3_x4::ZEROES,
        );

        Direct {
            radiance,
            base: albedo(surface.materials, p, n) * throughput,
            incoming: Packed_Vec3_x4::select(
                through_water,
                incoming + splat(BOUNCE_AMBIENT),
                incoming,
            ),
            start,
            normals: n,
            lit,
            through_water,
        }
    }

    /// light that one cosine bounce brings back to the active lanes, sky if it escapes
    #[inline(always)]
    fn bounce(
        &self,
        start: Packed_Vec3_x4,
        n: Packed_Vec3_x4,
        active: SimdU32,
        rng: &mut Rng,
    ) -> Packed_Vec3_x4 {
        let tree = &self.world.tree;
        let sun = &self.world.sun;

        // bounce lighting off means nobody traces, everyone just sees open sky //

        let bouncing = if self.bounce_lighting {
            active
        } else {
            SimdU32::ALL_INACTIVE
        };

        let bounce_dirs = cosine_sample(n, rng.next(), rng.next());
        let bounce_ray = Packet::new(start, bounce_dirs);
        let bounce = tree.trace::<false, true>(
            bounce_ray,
            bouncing,
            SimdF32::from_float(Self::BOUNCE_MAX_DISTANCE),
            Self::BOUNCE_CONE,
        );

        // escaped bounces pick up the sky dome, the sun was already counted as direct light //

        let mut indirect = Packed_Vec3_x4::select(
            active & bounce.hit.not(),
            sky(bounce_dirs),
            Packed_Vec3_x4::ZEROES,
        );

        if bounce.hit.has_none_active() {
            return indirect;
        }

        let n2 = bounce.normals;
        let p2 = bounce_ray.at(bounce.distances);
        let start2 = p2.add_scaled(n2, SimdF32::from_float(SURFACE_OFFSET));

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n2_dot_l = n2.dot(to_sun).max(SimdF32::ZEROES);
        let facing2 = bounce.hit & n2_dot_l.is_greater_than(SimdF32::ZEROES);

        let shadow2 = tree.trace::<true, true>(
            Packet::new(start2, to_sun),
            facing2,
            SimdF32::from_float(Self::BOUNCE_SHADOW_DISTANCE),
            Self::BOUNCE_CONE,
        );
        let sunlit2 = facing2 & shadow2.hit.not();

        let light2 = Packed_Vec3_x4::select(
            sunlit2,
            splat(sun.rgb).scale(n2_dot_l),
            Packed_Vec3_x4::ZEROES,
        ) + splat(BOUNCE_AMBIENT);

        indirect = Packed_Vec3_x4::select(
            bounce.hit,
            albedo(bounce.materials, p2, n2) * light2,
            indirect,
        );

        indirect
    }
}

/// a pixel's running sum after a moving frame:
/// whatever history made it over (capped at `cap` samples) plus the new one.
/// the count gets stored negative to mark it as a moving estimate, it's clean but soft
#[inline(always)]
fn blend(kept: Option<[f32; 4]>, rgb: [f32; 3], cap: f32) -> [f32; 4] {
    let Some(old) = kept.filter(|old| old[3] != 0.0) else {
        return [rgb[0], rgb[1], rgb[2], -1.0];
    };

    let count = old[3].abs().min(cap);
    let keep = count / old[3].abs();

    [
        old[0].mul_add(keep, rgb[0]),
        old[1].mul_add(keep, rgb[1]),
        old[2].mul_add(keep, rgb[2]),
        -(count + 1.0),
    ]
}

/// a pixel's running sum after one more still sample,
/// a soft moving estimate only counts as a single sample once the camera stops
#[inline(always)]
fn still(sum: [f32; 4], rgb: [f32; 3]) -> [f32; 4] {
    let keep = if sum[3] < 0.0 { 1.0 / -sum[3] } else { 1.0 };
    let count = if sum[3] < 0.0 { 1.0 } else { sum[3] };

    [
        sum[0].mul_add(keep, rgb[0]),
        sum[1].mul_add(keep, rgb[1]),
        sum[2].mul_add(keep, rgb[2]),
        count + 1.0,
    ]
}

/// last frame's camera, just enough to find where a world point was on screen
struct Reprojection {
    center: Vec3,
    front: Vec3,
    pixel_00: Vec3,
    /// pixel_delta / |pixel_delta|^2, a dot product with these lands in pixel units
    to_x: Vec3,
    to_y: Vec3,
    width: usize,
    height: usize,
}

impl Reprojection {
    fn new(camera: &Camera) -> Self {
        Self {
            center: camera.center,
            front: camera.front,
            pixel_00: camera.pixel_00,
            to_x: camera.pixel_delta_x / camera.pixel_delta_x.length_squared(),
            to_y: camera.pixel_delta_y / camera.pixel_delta_y.length_squared(),
            width: camera.render_width,
            height: camera.render_height,
        }
    }

    /// where a world point (meters) landed on screen last frame, in pixels (centers on
    /// whole numbers), and its view depth from that camera. None if it was behind it
    #[inline(always)]
    fn find(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let to_point = p - self.center;
        let ahead = to_point.dot(self.front);

        if ahead <= 1e-4 {
            return None;
        }

        // onto the viewport plane (focal length 1), then into pixel units //

        let on_plane = to_point * (1.0 / ahead) + self.center - self.pixel_00;

        Some((on_plane.dot(self.to_x), -on_plane.dot(self.to_y), ahead))
    }

    /// last frame's sum for a world point (meters), blended from the 4 pixels around
    /// where it landed. only pixels that saw the same face at about the same depth count,
    /// so history never leaks across an edge, and the blend keeps it from wobbling
    #[inline(always)]
    fn fetch(
        &self,
        p: Vec3,
        face: u8,
        history: &[[f32; 4]],
        depths: &[f32],
        faces: &[u8],
    ) -> Option<[f32; 4]> {
        let (x, y, then) = self.find(p)?;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let close = (then * Engine::DEPTH_TOLERANCE).max(Engine::DEPTH_SLACK);

        let mut total = [0.0f32; 4];
        let mut weights = 0.0;

        for (dx, dy, weight) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let (px, py) = (x0 as i32 + dx, y0 as i32 + dy);

            if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
                continue;
            }

            let i = py as usize * self.width + px as usize;
            let old = history[i];

            // same depth isn't enough, a voxel's side and top sit right next to each other //

            // far off faces flicker faster than a pixel, so there it's just depth //
            let other_face = faces[i] != face && then < Engine::FAR_DISTANCE;

            if weight <= 0.0 || old[3] == 0.0 || other_face || (depths[i] - then).abs() > close {
                continue;
            }

            // mix means, not sums, the counts can be wildly different //

            let count = old[3].abs();
            total[0] += old[0] / count * weight;
            total[1] += old[1] / count * weight;
            total[2] += old[2] / count * weight;
            total[3] += count * weight;
            weights += weight;
        }

        if weights < 1e-3 {
            return None;
        }

        // back to a sum, marked negative since it's a smudged moving estimate now //

        let count = total[3] / weights;
        let scale = count / weights;

        Some([total[0] * scale, total[1] * scale, total[2] * scale, -count])
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

/// a pixel's color from its running light sum and its running surface color sum
#[inline(always)]
fn mean_color(light: [f32; 4], tint: [f32; 4]) -> [f32; 3] {
    let inv = 1.0 / (light[3].abs().max(1.0) * tint[3].max(1.0));
    [0, 1, 2].map(|c| light[c] * tint[c] * inv)
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

/// final radiance of a packet once we know its bounce light
#[inline(always)]
fn compose(direct: &Direct, indirect: Packed_Vec3_x4) -> Packed_Vec3_x4 {
    Packed_Vec3_x4::select(
        direct.lit,
        direct.radiance + direct.base * (direct.incoming + indirect),
        direct.radiance,
    )
}

/// turn the tile 1D coordinate into the relative
/// XY coordinates for the top left pixel in each packet
/// yields every packet's top left pixel in the tile
fn morton_to_xy(i: usize) -> (usize, usize) {
    // this is not mine i found this shit on stack overflow bc i am not this smart
    // https://stackoverflow.com/questions/4909263/how-to-efficiently-de-interleave-bits-inverse-morton

    const EXTRACT_EVEN: fn(usize) -> usize = |i| {
        (((i & 0x5555).wrapping_mul(0xC00030000C0003)) & 0x0600180060008001)
            .wrapping_mul(0x0101010101010101)
            >> 56
    };
    const EXTRACT_ODD: fn(usize) -> usize = |i| {
        (((i & 0xAAAA).wrapping_mul(0xC00030000C0003)) & 0x03000C003000C000)
            .wrapping_mul(0x0101010101010101)
            >> 56
    };

    (EXTRACT_EVEN(i) * 2, EXTRACT_ODD(i) * 2)
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

    #[test]
    fn test_reprojection() {
        let camera = Camera::new(Vec3::new(10.0, 20.0, 30.0), 64, 32, 75.0, 2.0, -10.0, 40.0);
        let past = Reprojection::new(&camera);
        let ray_at = |x: f32, y: f32| {
            camera.pixel_00 + camera.pixel_delta_x * x - camera.pixel_delta_y * y - camera.center
        };

        // a spot along any pixel's ray lands right back on that pixel, at its view depth //

        for (x, y) in [(0.0, 0.0), (63.0, 31.0), (17.0, 9.0)] {
            let (px, py, depth) = past.find(camera.center + ray_at(x, y) * 7.0).unwrap();
            assert!((px - x).abs() < 1e-3 && (py - y).abs() < 1e-3, "{px} {py}");
            assert!((depth - 7.0).abs() < 1e-3, "depth {depth}");
        }

        // behind the camera is nowhere //

        assert!(past.find(camera.center - camera.front).is_none());

        // fetching a pixel's own spot gives back its history, as long as the depth matches //

        let mut history = vec![[0.0f32; 4]; 64 * 32];
        let mut depths = vec![f32::INFINITY; 64 * 32];
        let mut faces = vec![SKY_FACE; 64 * 32];
        history[9 * 64 + 17] = [2.0, 4.0, 6.0, 2.0];
        depths[9 * 64 + 17] = 7.0;
        faces[9 * 64 + 17] = 2;

        let spot = camera.center + ray_at(17.0, 9.0) * 7.0;
        let kept = past.fetch(spot, 2, &history, &depths, &faces).unwrap();
        assert!((kept[0] / -kept[3] - 1.0).abs() < 1e-3 && (kept[3] + 2.0).abs() < 1e-3);

        // another face or another depth means it was looking at something else //

        assert!(past.fetch(spot, 3, &history, &depths, &faces).is_none());
        depths[9 * 64 + 17] = 3.0;
        assert!(past.fetch(spot, 2, &history, &depths, &faces).is_none());
    }
}
