use pixels::{Pixels, ScalingMode, SurfaceTexture};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::error::EventLoopError;
use winit::event::{MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

use crate::engine::Engine;
use crate::engine::input::Action;
use crate::world::World;

pub struct App<'a> {
    window: Option<Arc<Window>>,
    pixel_buffer: Option<Pixels<'a>>,
    /// window size, logical before the window exists, physical after
    width: usize,
    height: usize,
    engine: Engine,
}

impl App<'_> {
    /// render_rows is the fixed render height, the width follows the window aspect
    pub fn new(width: usize, height: usize, render_rows: usize, world: World) -> Self {
        let (render_width, render_height) = render_size(width, height, render_rows);

        Self {
            window: None,
            pixel_buffer: None,
            width,
            height,
            engine: Engine::new(render_width, render_height, world),
        }
    }

    pub fn new_event_loop() -> Result<EventLoop<()>, EventLoopError> {
        let el = EventLoop::new()?;
        el.set_control_flow(ControlFlow::Poll);
        Ok(el)
    }

    /// the surface always follows the window, the render buffer only
    /// changes with the window aspect or the render rows, never with its size
    fn resize_window_to(&mut self, size: PhysicalSize<u32>) {
        let width = size.width.max(1);
        let height = size.height.max(1);
        let px_ref = self.pixel_buffer.as_mut().unwrap();

        if let Err(e) = px_ref.resize_surface(width, height) {
            eprintln!("pixel surface resizing error: {e}");
        }

        let (render_width, render_height) =
            render_size(width as usize, height as usize, self.engine.render_rows);

        if (render_width, render_height) != (self.engine.render_width, self.engine.render_height) {
            if let Err(e) = px_ref.resize_buffer(render_width as u32, render_height as u32) {
                eprintln!("pixel buffer resizing error: {e}");
            }

            self.engine.resize(render_width, render_height);
        }

        self.width = width as usize;
        self.height = height as usize;
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = LogicalSize::new(self.width as u32, self.height as u32);
        let resize_inc = PhysicalSize::new(16, 16);

        let attributes = WindowAttributes::default()
            .with_title("CPU voxel ray-tracer test")
            .with_active(true)
            .with_inner_size(window_size)
            .with_resize_increments(resize_inc)
            .with_resizable(true)
            .with_decorations(true)
            .with_visible(true);

        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        let _ = window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked));
        window.set_cursor_visible(false);

        let surface_size = window.inner_size();
        let (render_width, render_height) = (self.engine.render_width, self.engine.render_height);

        let surface_texture =
            SurfaceTexture::new(surface_size.width, surface_size.height, window.clone());

        // the low resolution buffer is stretched over the whole window, filtered //

        let mut pixels =
            Pixels::new(render_width as u32, render_height as u32, surface_texture).unwrap();
        pixels.set_scaling_mode(ScalingMode::Fill);
        self.pixel_buffer = Some(pixels);

        self.window = Some(window);

        self.resize_window_to(surface_size);

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(new_size) => {
                self.resize_window_to(new_size);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let buffer = self.pixel_buffer.as_mut().unwrap().frame_mut();

                self.engine.frame(buffer.as_mut_ptr() as usize);

                if self.engine.set_fullscreen {
                    self.engine.set_fullscreen = false;
                    let win = self.window.as_ref().unwrap();
                    if win.fullscreen().is_some() {
                        win.set_fullscreen(None);
                    } else {
                        win.set_fullscreen(Some(Fullscreen::Borderless(None)));
                    }
                }

                // the dev menu changed the render rows //

                if self.engine.resize_render {
                    self.engine.resize_render = false;
                    let size = self.window.as_ref().unwrap().inner_size();
                    self.resize_window_to(size);
                }

                let present_start = std::time::Instant::now();
                self.pixel_buffer.as_mut().unwrap().render().unwrap();
                self.engine.present_ms = present_start.elapsed().as_secs_f32() * 1000.0;

                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key
                    && let Some(action) = Action::from_key(key)
                {
                    self.engine.input.set(action, event.state.is_pressed());
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(action) = Action::from_button(button) {
                    self.engine.input.set(action, state.is_pressed());
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // trackpads send zero deltas when a scroll ends, those would count as down //

                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };

                if y != 0.0 {
                    let action = if y > 0.0 {
                        Action::BrushGrow
                    } else {
                        Action::BrushShrink
                    };
                    self.engine.input.tap(action);
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.engine.input.mouse_motion = delta;
        }
    }
}

/// render buffer size for a window: fixed rows, the width keeps the window aspect,
/// both rounded to whole 16 pixel tiles
fn render_size(window_width: usize, window_height: usize, rows: usize) -> (usize, usize) {
    let tile = Engine::PIXELS_PER_TILE_AXIS;
    let aspect = window_width as f64 / window_height.max(1) as f64;
    let width = ((rows as f64 * aspect / tile as f64).round() as usize).max(1) * tile;

    (width, (rows / tile).max(1) * tile)
}
