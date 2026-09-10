pub mod camera;
pub mod input;
pub mod player;
pub mod rendering;

use crate::math::vec3::{Vec3, v3};
use crate::world::World;
use camera::Camera;
use input::handler::Handler;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Engine {
    pub render_distance: f32, // in terms of voxels NOT chunks
    pub render_width: usize,
    pub render_height: usize,

    /// TIME
    pub last_time: std::time::Instant,
    pub time_accumulator: f32,
    pub fps_time_accumulator: f32,
    pub frame_count: f32,
    pub smooth_fps: f32,

    pub set_fullscreen: bool,

    pub frame_accumulation_buffer: Vec<u32>,
    pub frame_accumulation_count: u32,

    pub camera: Camera,
    pub input: Handler,
    pub world: World,
}

impl Engine {
    /// number of pixels along tile axis
    pub const PIXELS_PER_TILE_AXIS: usize = 16;

    /// 64 ticks per second
    pub const TICK_RATE: f32 = 64.0;
    /// in microseconds -> 1 / 64 seconds = 15625 micro seconds
    pub const TICK_LENGTH: f32 = 1.0 / Self::TICK_RATE;

    /// 250ms
    pub const MAX_FRAME_TIME: f32 = 0.25;

    pub fn new(render_width: usize, render_height: usize) -> Self {
        let camera_pos = Vec3::new(0.5, 0.5, -2.0);
        let aspect_ratio = render_width as f32 / render_height as f32;

        let mut camera = Camera::new(
            camera_pos,
            render_width,
            render_height,
            75.0,
            aspect_ratio,
            0.0,
            0.0,
        );
        camera.look_at(v3(0.5, 0.5, 0.5));

        Self {
            render_distance: 64.0,
            render_width,
            render_height,
            last_time: std::time::Instant::now(),
            time_accumulator: 0.0,
            fps_time_accumulator: 0.0,
            frame_count: 0.0,
            smooth_fps: 0.0,
            camera,
            world: World::new(),
            input: Default::default(),
            set_fullscreen: false,
            frame_accumulation_buffer: vec![0; render_width * render_height],
            frame_accumulation_count: 0,
        }
    }

    pub fn frame(&mut self, buffer_address: usize) {
        /* CALCULATE TIMES */

        let delta_time = self
            .last_time
            .elapsed()
            .as_secs_f32()
            .min(Self::MAX_FRAME_TIME);
        self.last_time = std::time::Instant::now();

        self.fps_time_accumulator += delta_time;
        self.frame_count += 1.0;
        if self.fps_time_accumulator >= 1.0 {
            // 0.5s
            self.smooth_fps = self.frame_count / self.fps_time_accumulator;
            self.frame_count = 0.0;
            self.fps_time_accumulator = 0.0;
        }
        eprintln!(
            "\t\t Delta Time: {:.2}ms  \t Frame Rate: {:.1} \t",
            delta_time * 1000.0,
            self.smooth_fps
        );

        /* PROCESS FRAME DATA */

        self.handle_input(delta_time);
        let _ = self.render_distance;

        self.world.update(delta_time);

        /* PROCESS TICKED DATA */

        self.time_accumulator += delta_time;

        while self.time_accumulator >= Self::TICK_LENGTH {
            self.time_accumulator -= Self::TICK_LENGTH;
        }

        // let interpolation_alpha = self.time_accumulator as f64 / Self::TICK_LENGTH as f64;

        /* RENDER */

        self.render(buffer_address);

        // self.render(buffer_address);

        /* CLEANUP */

        self.cleanup();
    }

    // pub fn tick(&mut self) {}

    // pub fn shutdown(&mut self) {}

    // pub fn save(&mut self) {}

    // pub fn load(&mut self) {}

    pub fn cleanup(&mut self) {
        self.input.activity.cleanup();
    }
}
