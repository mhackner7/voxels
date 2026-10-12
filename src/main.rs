mod app;
mod engine;
mod math;
mod simd;
mod world;

use app::App;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  VOXELS ENGINE - High-Performance CPU Software Raytracer  ");
    println!("  Architecture: 64-Tree (Contree), 4-Wide SIMD (NEON/AVX)   ");
    println!("============================================================");

    let total_init_start = std::time::Instant::now();

    // fixed 720 render rows, the gpu stretches the pixel buffer over any window size
    println!("[Main] Target window resolution: 1280x720 (render rows: 720)");
    let mut app = App::new(1280, 720, 720);

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
