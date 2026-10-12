pub mod camera;
pub mod input;
pub mod player;
pub mod rendering;
pub mod shading;
pub mod ui;
pub mod usage;

use crate::math::vec3::Vec3;
use crate::world::World;
use camera::Camera;
use input::handler::Handler;
use player::Player;
use ui::Ui;
use usage::Usage;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

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

    /// rows of the render buffer, the width just follows the window aspect,
    /// so the window size (fullscreen too) never changes how much we trace
    pub render_rows: usize,
    /// render_rows changed, the app resizes the pixel buffer after this frame
    pub resize_render: bool,

    /// one diffuse bounce per sample, off = just sky ambient (about half the rays)
    pub bounce_lighting: bool,
    /// index into FRAME_BUDGETS: 60 fps, 30 fps, no target
    pub fps_target: usize,
    /// index into MOVING_STEPS, how much moving frames cut corners to fit the budget
    pub moving_step: usize,
    /// how many tile slices the next still frame gets split into to fit the budget
    pub still_slices: usize,
    /// samples per pixel since the camera last moved (on average)
    pub samples: f32,

    /// milliseconds the last frame spent tracing / presenting
    pub render_ms: f32,
    pub present_ms: f32,

    pub ui: Ui,
    pub usage: Usage,

    /// running linear rgb sum of every sample since the camera last moved
    pub frame_accumulation_buffer: Vec<[f32; 4]>,
    /// last frame's sums, a moving frame reprojects these instead of starting over
    pub history: Vec<[f32; 4]>,
    /// view depth (meters along the camera front) of each pixel's first hit, infinity = sky
    pub frame_depths: Vec<f32>,
    pub history_depths: Vec<f32>,
    /// which voxel face each pixel sees (0..6, 255 = sky), history only sticks to the same face
    pub frame_faces: Vec<u8>,
    pub history_faces: Vec<u8>,
    /// running sum of each pixel's surface color + count, the accumulation only holds light
    pub frame_albedo: Vec<[f32; 4]>,
    /// the camera that drew history
    pub last_camera: Camera,
    /// false once the world looks different (sun, lighting), so old samples are no good
    pub history_valid: bool,
    /// samples in the accumulation buffer, 0 = the next frame overwrites it
    pub frame_accumulation_count: u32,
    /// frames rendered since start, seeds the per pixel random numbers
    pub frame_index: u32,

    pub camera: Camera,
    pub input: Handler,
    pub player: Player,
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

    /// render_rows choices of the dev menu, multiples of the tile size
    pub const RENDER_ROWS: [usize; 4] = [352, 528, 720, 1088];

    pub fn new(render_width: usize, render_height: usize) -> Self {
        println!("[Engine] Initializing render pipeline...");
        println!(
            "[Engine] Framebuffer: {}x{} ({} pixels, Morton tile size: {}x{})",
            render_width,
            render_height,
            render_width * render_height,
            Self::PIXELS_PER_TILE_AXIS,
            Self::PIXELS_PER_TILE_AXIS
        );

        let camera_pos = Vec3::new(128.0, 160.0, -20.0);
        let look_target = Vec3::new(128.0, 112.0, 128.0);
        let aspect_ratio = render_width as f32 / render_height as f32;

        println!(
            "[Engine] Initializing Camera at pos [{:.1}, {:.1}, {:.1}], target [{:.1}, {:.1}, {:.1}]",
            camera_pos.data[0],
            camera_pos.data[1],
            camera_pos.data[2],
            look_target.data[0],
            look_target.data[1],
            look_target.data[2],
        );
        println!(
            "[Engine] Camera FOV: 75.0 deg, aspect ratio: {:.3}",
            aspect_ratio
        );

        let mut camera = Camera::new(
            camera_pos,
            render_width,
            render_height,
            75.0,
            aspect_ratio,
            0.0,
            0.0,
        );
        camera.look_at(look_target);

        let world_start = std::time::Instant::now();
        println!("[Engine] Initializing Voxel World...");
        let world = World::new();
        println!(
            "[Engine] World initialization finished in {:.2}ms",
            world_start.elapsed().as_secs_f64() * 1000.0
        );

        Self {
            render_distance: 256.0,
            render_width,
            render_height,
            last_time: std::time::Instant::now(),
            time_accumulator: 0.0,
            fps_time_accumulator: 0.0,
            frame_count: 0.0,
            smooth_fps: 0.0,
            camera,
            player: Player::new("player1", camera_pos),
            world,
            input: Default::default(),
            set_fullscreen: false,
            render_rows: render_height,
            resize_render: false,
            bounce_lighting: true,
            fps_target: 1,
            moving_step: 0,
            still_slices: 1,
            samples: 0.0,
            render_ms: 0.0,
            present_ms: 0.0,
            ui: Ui::new(),
            usage: Usage::new(),
            frame_accumulation_buffer: vec![[0.0; 4]; render_width * render_height],
            history: vec![[0.0; 4]; render_width * render_height],
            frame_depths: vec![f32::INFINITY; render_width * render_height],
            history_depths: vec![f32::INFINITY; render_width * render_height],
            frame_faces: vec![255; render_width * render_height],
            history_faces: vec![255; render_width * render_height],
            frame_albedo: vec![[0.0; 4]; render_width * render_height],
            last_camera: camera,
            history_valid: false,
            frame_accumulation_count: 0,
            frame_index: 0,
        }
    }

    pub fn frame(&mut self, buffer_address: usize) {
        // calculate frame delta and fps //

        let delta_time = self
            .last_time
            .elapsed()
            .as_secs_f32()
            .min(Self::MAX_FRAME_TIME);
        self.last_time = std::time::Instant::now();
        self.ui.record(delta_time * 1000.0);

        self.fps_time_accumulator += delta_time;
        self.frame_count += 1.0;
        if self.fps_time_accumulator >= 1.0 {
            // averaged over the last second, printed only when it updates
            self.smooth_fps = self.frame_count / self.fps_time_accumulator;
            self.frame_count = 0.0;
            self.fps_time_accumulator = 0.0;

            eprintln!(
                "\t\t Delta Time: {:.2}ms  \t Frame Rate: {:.1} \t",
                delta_time * 1000.0,
                self.smooth_fps
            );
        }

        // keys, mouse and the player move //

        self.handle_input(delta_time);

        let _ = self.render_distance;

        self.world.update(delta_time);

        // fixed tick loop //

        self.time_accumulator += delta_time;

        while self.time_accumulator >= Self::TICK_LENGTH {
            self.time_accumulator -= Self::TICK_LENGTH;
        }

        // let interpolation_alpha = self.time_accumulator as f64 / Self::TICK_LENGTH as f64;

        // render to pixel buffer //

        let render_start = std::time::Instant::now();

        // a converged frame keeps its image, the old ui on top of it is resolved away //

        if !self.render(buffer_address) && (self.ui.drawn || self.ui.visible()) {
            self.resolve(buffer_address);
        }

        self.render_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        self.draw_ui(buffer_address);

        // frame cleanup //

        self.cleanup();
    }

    /// new render size in pixels (multiples of the tile size), restarts accumulation
    pub fn resize(&mut self, width: usize, height: usize) {
        self.render_width = width;
        self.render_height = height;
        self.resize_accumulation_buffer(width, height);

        self.camera.render_width = width;
        self.camera.render_height = height;
        self.camera.aspect_ratio = width as f32 / height as f32;
        self.camera.update();
    }

    // pub fn tick(&mut self) {}

    // pub fn shutdown(&mut self) {}

    // pub fn save(&mut self) {}

    // pub fn load(&mut self) {}

    pub fn cleanup(&mut self) {
        self.input.activity.cleanup();
    }
}
