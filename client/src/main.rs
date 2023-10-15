use crate::renderer::VertexRenderer;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, SystemTime};
use tracing::{debug_span, error, info};
use winit::event::{Event, WindowEvent};

mod renderer;

fn main() -> ExitCode {
    let span = debug_span!("Client");
    let _guard = span.enter();

    // TODO - Env override for logging level
    #[cfg(debug_assertions)]
    let error_level = tracing::Level::DEBUG;
    #[cfg(not(debug_assertions))]
    let error_level = tracing::Level::ERROR;

    tracing_subscriber::fmt()
        .with_max_level(error_level)
        .with_target(true)
        .init();

    info!("Client using Common {}", common::version());

    let event_loop = winit::event_loop::EventLoopBuilder::new().build().unwrap();

    let window = winit::window::WindowBuilder::new()
        .with_transparent(false)
        .with_active(true)
        .with_title("Application")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut renderer = VertexRenderer::new("survival-game", (0, 1, 0), &window);
    if let Err(error_message) = renderer.load_shader(
        Path::new("res/shaders/basic.vert.spv"),
        Path::new("res/shaders/basic.frag.spv"),
        String::from("basic"),
    ) {
        error!("Failed to create basic shader pipeline: {}", error_message);
        return ExitCode::FAILURE;
    }

    const TARGET_FRAME_TIME: Duration = Duration::new(0, 1000000000 / 60);
    let _ = event_loop.run(|event, _window_target, control_flow| {
        let start_time = SystemTime::now();
        control_flow.set_poll();
        match event {
            Event::WindowEvent { event, .. } => {
                if event == WindowEvent::CloseRequested {
                    control_flow.set_exit();
                }
            }
            Event::RedrawRequested(_id) => {
                renderer.render(&window);
            }
            _ => {}
        }

        window.request_redraw();

        let current_time = SystemTime::now();
        while let Ok(time_to_sleep) = current_time.duration_since(start_time) {
            std::thread::sleep(time_to_sleep - TARGET_FRAME_TIME);
        }
    });

    ExitCode::SUCCESS
}
