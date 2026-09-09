use pixels::{Pixels, SurfaceTexture};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

use crate::engine::Engine;

pub struct App<'a> {
    window: Option<Arc<Window>>,
    pixel_buffer: Option<Pixels<'a>>,
    width: usize,
    height: usize,
    render_scale: f64,
    engine: Engine, // scale_factor: u32,
                    // camera: Camera,
                    // raytracer : RayTracer,
                    // world : World,
                    // user_info : Info,
                    // cursor : Cursor,
}

impl App<'_> {
    pub fn new(width: usize, height: usize, render_scale: f64) -> Self {
        Self {
            window: None,
            pixel_buffer: None,
            render_scale,
            width,
            height,
            engine: Engine::new(
                (width as f64 * render_scale) as usize,
                (height as f64 * render_scale) as usize,
            ),
        }
    }

    pub fn with_resolution(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_render_scale(mut self, scale: f64) -> Self {
        self.render_scale = scale;
        self
    }

    pub fn new_event_loop() -> Result<EventLoop<()>, EventLoopError> {
        let el = EventLoop::new()?;
        el.set_control_flow(ControlFlow::Poll);
        Ok(el)
    }

    fn resize_window_to(&mut self, size: PhysicalSize<u32>) {
        let width = size.width;
        let height = size.height;
        let px_ref = self.pixel_buffer.as_mut().unwrap();

        let render_width = (width as f64 * self.render_scale) as usize;
        let render_height = (height as f64 * self.render_scale) as usize;

        let tile_size = Engine::PIXELS_PER_TILE_AXIS;
        self.engine.render_width = ((render_width / tile_size) * tile_size).max(1);
        self.engine.render_height = ((render_height / tile_size) * tile_size).max(1);

        self.engine.camera.render_width = self.engine.render_width;
        self.engine.camera.render_height = self.engine.render_height;
        self.engine.camera.aspect_ratio =
            self.engine.render_width as f32 / self.engine.render_height as f32;
        self.engine.camera.update();

        if let Err(e) = px_ref.resize_surface(width, height) {
            eprintln!("pixel surface resizing error: {e}");
        }

        if let Err(e) = px_ref.resize_buffer(
            self.engine.render_width as u32,
            self.engine.render_height as u32,
        ) {
            eprintln!("pixel buffer resizing error: {e}");
        }

        self.width = width as usize;
        self.height = height as usize;
    }

    fn shutdown(&mut self) {
        eprintln!("Window Closing");
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = PhysicalSize::new(self.width as u32, self.height as u32);
        let resize_inc = PhysicalSize::new(16, 16);

        let attributes = WindowAttributes::default()
            .with_title("CPU voxel ray-tracer test")
            .with_active(true)
            .with_inner_size(window_size)
            .with_resize_increments(resize_inc)
            .with_resizable(true)
            .with_decorations(true)
            .with_visible(true);

        eprintln!("Opening Window");
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
            .unwrap();
        window.set_cursor_visible(false);

        eprintln!("Opening Pixel Buffer");
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, window.clone());

        let render_width = (self.width as f64 * self.render_scale) as u32;
        let render_height = (self.height as f64 * self.render_scale) as u32;
        self.pixel_buffer =
            Some(Pixels::new(render_width, render_height, surface_texture).unwrap());

        self.window = Some(window);

        self.resize_window_to(window_size);

        self.window.as_ref().unwrap().request_redraw();

        self.width = window_size.width as usize;
        self.height = window_size.height as usize;
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
                self.shutdown();
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

                self.pixel_buffer.as_mut().unwrap().render().unwrap();

                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => match event.physical_key {
                PhysicalKey::Code(keycode) => {
                    self.engine.input.push(keycode.into(), event.state.into())
                }
                _ => (),
            },
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => self.engine.input.push(button.into(), state.into()),
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
            } => {
                self.engine
                    .input
                    .push(delta.into(), crate::engine::input::util::State::Pressed);
            }
            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        eprintln!("Goodbye!");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            eprintln!("Mouse Moved! delta: {:?}", delta);
            self.engine.input.last_mouse_motion = delta;
        }
    }
}
