mod app;
mod engine;
mod math;
mod simd;
mod world;

use app::App;
use world::World;
use world::cache::Reuse;

const SEED: u64 = 42;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut app = App::new(1280, 720, 720, world);

    let event_loop = App::new_event_loop()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
