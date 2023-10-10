use std::ops::Deref;
use std::path::Path;
use tracing::debug;
use winit::event::{Event, WindowEvent};
use winit::platform::wayland::EventLoopWindowTargetExtWayland;
use crate::renderer::Renderer;

mod renderer;

fn main() {
    println!("Client using Common {}", common::version());

    #[cfg(debug_assertions)]
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    let event_loop = winit::event_loop::EventLoopBuilder::new()
        .build()
        .unwrap();

    let window = winit::window::WindowBuilder::new()
        .with_transparent(false)
        .with_active(true)
        .with_title("Application")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut renderer = Renderer::new("survival-game", (0, 1, 0), &window);
    renderer.load_shader(Path::new("res/shaders/basic.vert.spv"), Path::new("res/shaders/basic.frag.spv"), String::from("basic"))
        .expect("Failed to load basic shader");

    let _ = event_loop.run(|event, _window_target, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent { event, .. } => {
                if event == WindowEvent::CloseRequested {
                    control_flow.set_exit();
                }
            }
            Event::RedrawRequested(_id) => {
                window.pre_present_notify();
                renderer.render();
                debug!("Redraw");
            }
            _ => {}
        }
    });
}