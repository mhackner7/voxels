pub mod camera;
pub mod input;
pub mod lightcache;
pub mod lighting;
pub mod player;
pub mod rendering;
pub mod shading;
pub mod ui;
pub mod usage;

use crate::math::vec3::Vec3;
use crate::world::World;
use crate::world::edit::Target;
use camera::Camera;
use input::Input;
use lightcache::LightCache;
use player::Player;
use ui::Ui;
use usage::Usage;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

pub struct Engine {
    pub render_width: usize,
    pub render_height: usize,

    pub last_time: std::time::Instant,
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
    /// settled faces get re-lit every nth frame, the frame budget moves it
    pub every: u32,
    pub samples: f32,
    pub faces_lit: u32,

    /// milliseconds the last frame spent tracing / presenting
    pub render_ms: f32,
    pub present_ms: f32,

    pub ui: Ui,
    pub usage: Usage,

    /// running mean color per pixel (linear rgb) + frames in it
    pub frame_accumulation_buffer: Vec<[f32; 4]>,
    /// view depth (meters along the camera front) of each pixel's first hit, infinity = sky
    pub frame_depths: Vec<f32>,
    /// which voxel face each pixel sees (0..6, 255 = sky)
    pub frame_faces: Vec<u8>,
    pub lights: LightCache,
    /// samples in the accumulation buffer, 0 = the next frame overwrites it
    pub frame_accumulation_count: u32,
    /// frames rendered since start, seeds the per pixel random numbers
    pub frame_index: u32,

    pub camera: Camera,
    pub input: Input,
    pub player: Player,
    pub world: World,
    pub target: Option<Target>,
}

impl Engine {
    /// number of pixels along tile axis
    pub const PIXELS_PER_TILE_AXIS: usize = 16;

    /// longest frame the simulation steps over at once (seconds)
    pub const MAX_FRAME_TIME: f32 = 0.25;

    /// render_rows choices of the dev menu, multiples of the tile size
    pub const RENDER_ROWS: [usize; 4] = [352, 528, 720, 1088];

    pub fn new(render_width: usize, render_height: usize, world: World) -> Self {
        let camera_pos = Vec3::new(128.0, 160.0, -20.0);
        let look_target = Vec3::new(128.0, 112.0, 128.0);
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
        camera.look_at(look_target);

        Self {
            render_width,
            render_height,
            last_time: std::time::Instant::now(),
            fps_time_accumulator: 0.0,
            frame_count: 0.0,
            smooth_fps: 0.0,
            camera,
            player: Player::new(camera_pos),
            world,
            input: Input::new(),
            set_fullscreen: false,
            render_rows: render_height,
            resize_render: false,
            bounce_lighting: true,
            fps_target: 1,
            every: 1,
            samples: 0.0,
            faces_lit: 0,
            render_ms: 0.0,
            present_ms: 0.0,
            ui: Ui::new(),
            usage: Usage::new(),
            frame_accumulation_buffer: vec![[0.0; 4]; render_width * render_height],
            frame_depths: vec![f32::INFINITY; render_width * render_height],
            frame_faces: vec![255; render_width * render_height],
            lights: LightCache::new(LightCache::SLOTS),
            frame_accumulation_count: 0,
            frame_index: 0,
            target: None,
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

        // render to pixel buffer //

        let render_start = std::time::Instant::now();

        // a converged frame keeps its image, the old ui on top of it is resolved away //

        if !self.render(buffer_address) && (self.ui.drawn || self.ui.visible()) {
            self.resolve(buffer_address);
        }

        self.render_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        self.draw_ui(buffer_address);

        // presses only count for the frame they happened in //

        self.input.end_frame();
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
}
