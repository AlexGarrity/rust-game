use std::path::Path;
use winit::event::{Event, WindowEvent};

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
        .with_transparent(true)
        .with_title("Application")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let vulkan_context = renderer::vulkan::Context::new("application", (0, 1, 0));
    let mut vulkan_device = renderer::vulkan::Device::new(&vulkan_context);
    let vulkan_surface = renderer::vulkan::Surface::new(&vulkan_context, &vulkan_device, &window);
    vulkan_device.create_pipeline(&vulkan_surface, Path::new("res/shaders/basic.vert.spv"), Path::new("res/shaders/basic.frag.spv"), String::from("basic"));

    let _ = event_loop.run(|event, _window_target, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => {
                if event == WindowEvent::CloseRequested {
                    control_flow.set_exit();
                }
            }
            Event::RedrawRequested(_id) => {
                
            }
            _ => {}
        }
    });
}