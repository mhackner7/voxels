use super::*;
use crate::{
    math::vec3::v3,
    simd::{self, packetray::PacketBuilder, simdf32::SimdF32},
    world::voxel::Material,
};

impl Engine {
    pub fn resize_accumulation_buffer(&mut self, new_x: usize, new_y: usize) {
        self.frame_accumulation_buffer = vec![0; new_x * new_y];
        self.frame_accumulation_count = 0;
    }

    pub fn render(&mut self, buffer_address: usize) {
        self.camera.update();

        let packet_builder = PacketBuilder::new(
            self.camera.center,
            self.camera.pixel_delta_x,
            self.camera.pixel_delta_y,
        );

        /*
            The pixel buffer gets broken up into tiles of 16x16 pixels
            the tiles are then looped through in parallel to render

            each tile, in the loop, is then subdivided again into
            16 subtiles, or 4x4 subtiles
            each subtile contains 4x4 pixels, yielding 4 packets per subtile

            each packet is a 2x2 square of pixels which will be used for ray tracing

            packets are ray traced alltogether using simd
        */

        let tile_xcount = self.render_width / Engine::PIXELS_PER_TILE_AXIS;
        let tile_ycount = self.render_height / Engine::PIXELS_PER_TILE_AXIS;

        let max_dim = tile_xcount.max(tile_ycount).next_power_of_two();
        let total_morton_tiles = max_dim * max_dim;

        let accum_ptr = self.frame_accumulation_buffer.as_mut_ptr() as usize;
        let accum_frames = self.frame_accumulation_count as f32;
        self.frame_accumulation_count += 1;

        (0..total_morton_tiles)
            .into_par_iter()
            .filter_map(|i| {
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
                let tile_pixel00_x = tile_x * Engine::PIXELS_PER_TILE_AXIS as usize;
                let tile_pixel00_y = tile_y * Engine::PIXELS_PER_TILE_AXIS as usize;

                let mut buffer = [[0u32; Engine::PIXELS_PER_TILE_AXIS as usize];
                    Engine::PIXELS_PER_TILE_AXIS as usize];

                const MAX_ITERATIONS: usize = (Engine::PIXELS_PER_TILE_AXIS / 2).pow(2) as usize;

                for i in 0..MAX_ITERATIONS {
                    /* BUILD PIXEL PACKET */

                    // the top left pixel of the packet, relative to the tile
                    let (rel_x, rel_y) = morton_to_xy(i);

                    // top left x and y globally
                    let packet_pixel_x = tile_pixel00_x + rel_x;
                    let packet_pixel_y = tile_pixel00_y + rel_y;

                    let jitter_x = fastrand::f32() - 0.5;
                    let jitter_y = fastrand::f32() - 0.5;

                    // packet top left coord as vec3
                    let packet00_coord = self.camera.pixel_00
                        + (self.camera.pixel_delta_x * (packet_pixel_x as f32 + jitter_x))
                        - (self.camera.pixel_delta_y * (packet_pixel_y as f32 + jitter_y));

                    let packet = packet_builder.make(packet00_coord);

                    /* RENDER PIXEL PACKET */

                    let (materials, distances, valid) = self.world.trace(packet);

                    let write_pixel = |idx: usize| -> u32 {
                        if valid[idx] != 0 {
                            const SCALE: f32 = 1.0 / 64.0;

                            let intensity = 1.0 - (distances[idx] * SCALE).clamp(0.0, 1.0);

                            let alpha = (intensity * 255.0) as u8;

                            let color = Material::COLORS[materials[idx] as usize];

                            color | ((alpha as u32) << 24)
                        } else {
                            // Dark grey background for misses
                            Material::COLORS[0] | 0xFF000000 as u32
                        }
                    };

                    /* WRITE TO GLOBAL ACCUMULATOR AND LOCAL TILE BUFFER */

                    buffer[rel_y][rel_x] = write_pixel(0);
                    buffer[rel_y][rel_x + 1] = write_pixel(1);
                    buffer[rel_y + 1][rel_x] = write_pixel(2);
                    buffer[rel_y + 1][rel_x + 1] = write_pixel(3);
                }

                /* WRITE THE LOCAL BUFFER INTO THE ACTUAL PIXEL BUFFER */

                let mut index = (tile_pixel00_y * self.render_width) + tile_pixel00_x;

                let pointer = buffer_address as *mut u32;

                for y in 0..Engine::PIXELS_PER_TILE_AXIS {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buffer[y].as_ptr(),
                            pointer.add(index),
                            Engine::PIXELS_PER_TILE_AXIS,
                        );
                    }

                    index += self.render_width;
                }
            });
    }
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
