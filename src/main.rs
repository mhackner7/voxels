mod app;
mod engine;
mod math;
mod simd;
mod world;

use app::App;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing voxel engine...");

    let mut app = App::new(1280, 720, 1.0);

    let event_loop = App::new_event_loop()?;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
