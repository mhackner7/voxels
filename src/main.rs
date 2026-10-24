mod app;
mod engine;
mod math;
mod simd;
mod world;

use app::App;
use world::World;
use world::cache::Reuse;

/// the test world
const SEED: u64 = 42;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  VOXELS ENGINE - High-Performance CPU Software Raytracer  ");
    println!("  Architecture: 64-Tree (Contree), 4-Wide SIMD (NEON)       ");
    println!("============================================================");

    let total_init_start = std::time::Instant::now();

    // --reuse / --new-world answer the "reuse cached world?" question up front //

    let args: Vec<String> = std::env::args().collect();
    let reuse = if args.iter().any(|a| a == "--reuse") {
        Reuse::Yes
    } else if args.iter().any(|a| a == "--new-world") {
        Reuse::No
    } else {
        Reuse::Ask
    };

    let world = World::load_or_build(SEED, reuse);

    // fixed 720 render rows, the gpu stretches the pixel buffer over any window size
    println!("[Main] Target window resolution: 1280x720 (render rows: 720)");
    let mut app = App::new(1280, 720, 720, world);

    println!("[Main] Creating event loop (ControlFlow::Poll)...");
    let event_loop = App::new_event_loop()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    println!(
        "[Main] All systems initialized in {:.2}ms. Entering run loop.",
        total_init_start.elapsed().as_secs_f64() * 1000.0
    );

    event_loop.run_app(&mut app)?;

    Ok(())
}
