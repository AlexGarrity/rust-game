use std::ffi::CString;

use ash::extensions;
use ash::vk;
use tracing::{debug, debug_span};

pub struct Context {
    pub application_name: CString,
    pub engine_name: CString,
    pub entry_point: ash::Entry,
    pub instance: ash::Instance,
}

impl Context {

    /// Constructs a new Context
    ///
    /// # Arguments
    ///
    /// * `application_name`: The name of the application that the context will be used by, as a `str`
    /// * `application_version`: The version of the application that the context will be used by, as a 3-tuple of `u32`s
    ///
    /// # Examples
    ///
    /// ```
    /// use client::renderer::vulkan::Context;
    ///
    /// let context = Context::new("my-application", (1.4.2));
    /// ```
    pub fn new(application_name: &str, application_version: (u32, u32, u32)) -> Self {
        let span = debug_span!("Vulkan/Context");
        let _guard = span.enter();

        // TODO - See if we can link statically instead
        debug!("Loading Vulkan dynamically");
        let entry_point = unsafe { ash::Entry::load() }
            .expect("Failed to load Vulkan libraries");
        debug!("Loaded successfully");

        let engine_name = CString::new("engine").unwrap();
        let application_name = CString::new(application_name).unwrap();

        let application_info = vk::ApplicationInfo::builder()
            .engine_name(engine_name.as_ref())
            .application_name(application_name.as_ref())
            .api_version(vk::API_VERSION_1_3)
            .application_version(vk::make_api_version(0, application_version.0, application_version.1, application_version.2))
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .build();

        let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();

        // TODO - Figure out if it's worth just targeting Wayland on Unix
        // TODO - Test for extensions before using them (albeit if we don't have surface then we're a bit scuppered anyway)
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&[
                extensions::khr::Surface::name().as_ptr(),
                #[cfg(target_os = "windows")]
                    extensions::khr::Win32Surface::name().as_ptr(),
                #[cfg(target_os = "linux")]
                    extensions::khr::XcbSurface::name().as_ptr(),
                #[cfg(target_os = "linux")]
                    extensions::khr::WaylandSurface::name().as_ptr(),
                #[cfg(target_os = "macos")]
                    extensions::ext::MetalSurface::name().as_ptr()
            ])
            .enabled_layer_names(&[
                #[cfg(debug_assertions)]
                    validation_layer_name.as_ptr()
            ])
            .build();

        debug!("Creating Vulkan Instance");
        let instance = unsafe { entry_point.create_instance(&instance_create_info, None) }
            .expect("Failed to create a Vulkan instance");
        debug!("Created successfully");

        Context {
            application_name,
            engine_name,
            entry_point,
            instance,
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Context");
        let _guard = span.enter();

        debug!("Destroying instance");
        unsafe { self.instance.destroy_instance(None); }
        debug!("Successfully destroyed instance");
    }
}