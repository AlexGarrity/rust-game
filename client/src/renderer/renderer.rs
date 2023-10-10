use std::ops::{DerefMut};
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::renderer::vulkan::{Context, Device, Surface};

pub struct Renderer {
    // These must stay in order as objects are dropped in the order they're declared
    // Surface depends on device, which depends on context
    surface: Surface,
    device: Arc<RwLock<Device>>,
    context: Context,
}

impl Renderer {
    pub fn new(application_name: &str, application_version: (u32, u32, u32), window: &winit::window::Window) -> Self {
        let context = Context::new(application_name, application_version);
        let mut device = Arc::new(RwLock::new(Device::new(&context)));
        let surface = Surface::new(&context, &device, &window);

        Renderer {
            surface,
            device,
            context,
        }
    }

    pub fn load_shader(&mut self, vertex_shader_path: &Path, fragment_shader_path: &Path, shader_name: String) -> Result<(), &'static str> {
        let device_guard = self.device.write();
        let mut device_lock = device_guard.unwrap();
        let device = device_lock.deref_mut();

        if device.create_pipeline(&self.surface, vertex_shader_path, fragment_shader_path, shader_name.clone()) {
            Err("Failed to create pipeline on device")
        } else {
            let pipeline = device.get_pipeline(shader_name.as_str())
                .expect("Failed to get pipeline after creation");
            self.surface.create_framebuffers_for_pipeline(&device, pipeline);
            Ok(())
        }
    }

    pub fn render(&mut self) {
        let next_image = {
            let device_guard = self.device.write();
            let mut device_lock = device_guard.unwrap();
            let device = device_lock.deref_mut();

            let current_frame_index = self.surface.get_current_frame_index();
            let next_frame_index = device.begin_graphics_render_pass(current_frame_index, &mut self.surface, "basic");
            device.draw_vertices(current_frame_index, 3);
            device.end_graphics_render_pass(current_frame_index);
            next_frame_index
        };

        self.surface.flip_buffers(next_image);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.device.read().unwrap().logical_device.device_wait_idle() }
            .expect("Device was removed during cleanup");
    }
}