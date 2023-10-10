use std::ops::Deref;
use std::sync::{Arc, RwLock};

use ash::{extensions, vk};
use num;
use tracing::{debug, debug_span};
use winit::window::raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::renderer::vulkan::{Context, Device, Pipeline};

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct SwapChainInfo {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct SwapChainParameters {
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

pub struct Surface {
    device: Arc<RwLock<Device>>,
    surface_extension: extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    swapchain_extension: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    pub(in super::super::vulkan) swapchain_parameters: SwapChainParameters,
    swapchain_images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Option<Vec<vk::Framebuffer>>,
    current_framebuffer_index: usize,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    pub(in super::super::vulkan) frame_in_flight: Vec<vk::Fence>,
}

impl Surface {
    /// Constructs a new `Surface`
    ///
    /// # Arguments
    ///
    /// * `context`: The `Context` which will render to the `Surface`
    /// * `device`: The `Device` which will render to the `Surface`
    /// * `window`: The `Window` that the surface should be created on
    ///
    /// # Examples
    ///
    /// ```
    /// use winit::{window::WindowBuilder, event_loop::EventLoopBuilder};
    /// use client::renderer::vulkan::{Context, Device, Surface};
    ///
    /// let event_loop = EventLoopBuilder::new().build().unwrap();
    /// let window = WindowBuilder::new().build(&event_loop).unwrap();
    ///
    /// let context = Context::new("my-application", (1.4.2));
    /// let device = Device::new(&context);
    /// let surface = Surface::new(&context, &device, &window);
    /// ```
    pub fn new(context: &Context, device_ref: &Arc<RwLock<Device>>, window: &winit::window::Window) -> Self {
        let span = debug_span!("Vulkan/Surface");
        let _guard = span.enter();

        let device_guard = device_ref.read();
        let device_lock = device_guard.unwrap();
        let device = device_lock.deref();

        let extension = extensions::khr::Surface::new(&context.entry_point, &context.instance);
        debug!("Creating SurfaceKHR");
        let surface = unsafe { ash_window::create_surface(&context.entry_point, &context.instance, window.raw_display_handle(), window.raw_window_handle(), None) }
            .expect("Failed to create Vulkan surface");
        debug!("Successfully created surface");

        let device_swapchain_info = get_swapchain_info(device, &surface, &extension);
        let swapchain_parameters = get_swapchain_parameters(&device_swapchain_info, window, None, None);

        let swapchain_extension = extensions::khr::Swapchain::new(&context.instance, device.logical_device.as_ref());
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .image_format(swapchain_parameters.surface_format.format)
            .image_color_space(swapchain_parameters.surface_format.color_space)
            .present_mode(swapchain_parameters.present_mode)
            .image_extent(swapchain_parameters.extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_array_layers(1)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .min_image_count(if device_swapchain_info.capabilities.max_image_count >= 2 { 2 } else { 1 })
            .build();

        debug!("Creating SwapchainKHR");
        let swapchain = unsafe { swapchain_extension.create_swapchain(&swapchain_create_info, None) }
            .expect("Failed to create Vulkan swapchain");
        debug!("Successfully created swapchain");

        let swapchain_images = unsafe { swapchain_extension.get_swapchain_images(swapchain) }
            .expect("Failed to create swapchain images");

        let image_views = swapchain_images.iter().map(|image| {
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY)
                        .build()
                )
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .base_array_layer(0)
                        .level_count(1)
                        .layer_count(1)
                        .build()
                )
                .format(swapchain_parameters.surface_format.format)
                .build();

            let image_view = unsafe { device.logical_device.create_image_view(&image_view_create_info, None) }
                .expect("Failed to create swapchain image view");
            image_view
        })
            .collect::<Vec<vk::ImageView>>();

        let semaphore_create_info = vk::SemaphoreCreateInfo::builder()
            .build();

        let image_available: Vec<vk::Semaphore> = (0..MAX_FRAMES_IN_FLIGHT).map(|i| {
            unsafe { device.logical_device.create_semaphore(&semaphore_create_info, None) }
                .expect("Failed to create semaphore for checking if framebuffer is available")
        })
            .collect();
        let render_finished: Vec<vk::Semaphore> = (0..MAX_FRAMES_IN_FLIGHT).map(|i| {
            unsafe { device.logical_device.create_semaphore(&semaphore_create_info, None) }
                .expect("Failed to create semaphore for checking if render is finished")
        })
            .collect();

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();

        let frame_in_flight: Vec<vk::Fence> = (0..MAX_FRAMES_IN_FLIGHT).map(|i| {
            unsafe { device.logical_device.create_fence(&fence_create_info, None) }
                .expect("Failed to create fence for checking if frame is in flight")
        })
            .collect();


        Surface {
            device: device_ref.clone(),
            surface_extension: extension,
            surface,
            swapchain_extension,
            swapchain,
            swapchain_parameters,
            swapchain_images,
            image_views,
            framebuffers: None,
            current_framebuffer_index: 0,
            image_available,
            render_finished,
            frame_in_flight,
        }
    }

    pub fn create_framebuffers_for_pipeline(&mut self, device: &Device, pipeline: &Pipeline) {
        let framebuffers = (0..self.image_views.len()).map(|index| {
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(pipeline.render_pass)
                .width(self.swapchain_parameters.extent.width)
                .height(self.swapchain_parameters.extent.height)
                .attachments(&[self.image_views[index]])
                .layers(1)
                .build();

            unsafe { device.logical_device.create_framebuffer(&framebuffer_create_info, None) }
                .expect("Failed to create framebuffer")
        })
            .collect::<Vec<vk::Framebuffer>>();

        self.framebuffers = Some(framebuffers);
    }

    pub fn get_framebuffer(&mut self, index: usize) -> &vk::Framebuffer {
        let framebuffers = self.framebuffers
            .as_ref()
            .expect("No framebuffers have been created, but one has been requested");

        framebuffers.get(index)
            .unwrap()
    }

    pub fn acquire_next_image(&self) -> u32 {
        unsafe {
            self.swapchain_extension.acquire_next_image(
                self.swapchain, u64::MAX,
                *self.image_available.get(self.current_framebuffer_index).unwrap(),
                *self.frame_in_flight.get(self.current_framebuffer_index).unwrap(),
            )
        }
            .expect("Failed to acquire next image")
            .0
    }

    pub fn flip_buffers(&mut self, next_image: u32) {
        let device_guard = self.device.read();
        let device_lock = device_guard.unwrap();
        let device = device_lock.deref();

        device.submit_graphics_queue(
            self.current_framebuffer_index,
            &[*self.render_finished.get(self.current_framebuffer_index).unwrap()],
            &[*self.image_available.get(self.current_framebuffer_index).unwrap()],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            self.frame_in_flight.get(self.current_framebuffer_index).unwrap(),
        );

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&[*self.render_finished.get(self.current_framebuffer_index).unwrap()])
            .swapchains(&[self.swapchain])
            .image_indices(&[next_image])
            .build();

        device.present_queue(next_image as usize, &self.swapchain_extension, &present_info);

        self.current_framebuffer_index = (self.current_framebuffer_index + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn get_current_frame_index(&self) -> usize {
        return self.current_framebuffer_index;
    }

    pub fn get_frame_in_flight(&self) -> vk::Fence {
        *self.frame_in_flight.get(self.current_framebuffer_index).unwrap()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Surface");
        let _guard = span.enter();

        let device_guard = self.device.read();
        let device_lock = device_guard.unwrap();
        let device = device_lock.deref();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe { device.logical_device.destroy_fence(*self.frame_in_flight.get(i).unwrap(), None) };
            unsafe { device.logical_device.destroy_semaphore(*self.render_finished.get(i).unwrap(), None) };
            unsafe { device.logical_device.destroy_semaphore(*self.image_available.get(i).unwrap(), None) };
        }

        match &self.framebuffers {
            Some(framebuffers) => {
                for framebuffer in framebuffers {
                    debug!("Destroying framebuffer {:?}", framebuffer);
                    unsafe { device.logical_device.destroy_framebuffer(*framebuffer, None) };
                    debug!("Successfully destroyed framebuffer");
                }
            }
            None => {}
        }

        for image_view in &self.image_views {
            debug!("Destroying image view {:?}", image_view);
            unsafe { device.logical_device.destroy_image_view(*image_view, None) };
            debug!("Successfully destroyed image view");
        }

        debug!("Destroying swapchain");
        unsafe { self.swapchain_extension.destroy_swapchain(self.swapchain, None) };
        debug!("Successfully destroyed swapchain");

        debug!("Destroying surface");
        unsafe { self.surface_extension.destroy_surface(self.surface, None) };
        debug!("Successfully destroyed surface");
    }
}

/// Gets the optimal parameters for the given swapchain, according to the information provided by `swapchain_info`.
///
/// By default:
/// - The preferred surface format is `SRGB_NONLINEAR` `B8G8R8A8_UNORM` or `B8G8R8A8_SRGB`, falling back to the first in the list if neither is available
/// - The preferred present mode is `FIFO_RELAXED`, falling back to `FIFO` as it's always available
///
/// The defaults can be overridden with the `preferred_*` variables
///
/// # Arguments
///
/// * `swapchain_info`: A `SwapChainInfo` struct containing information returned by [`get_swapchain_info()`]
/// * `window`: The window that the swapchain is being created for
/// * `preferred_surface_format`: If a different surface format to the ones described above is preferred, this can be set to try prioritise using something else
/// * `preferred_present_mode`: If a different present mode to the ones described above is preferred, this can be set to try prioritise using something else
///
/// # Examples
///
/// ```
/// use client::renderer::vulkan::{Context, Device};
/// use winit::{window::WindowBuilder, event_loop::EventLoopBuilder};
/// use ash::{vk, extensions};
///
/// let event_loop = EventLoopBuilder::new().build().unwrap();
/// let window = WindowBuilder::new().build(&event_loop).unwrap();
///
/// let context = Context::new("my-application", (1.4.2));
/// let device = Device::new(&context);
///
/// let extension = extensions::khr::Surface::new(&context.entry_point, &context.instance);
/// let surface = unsafe { ash_window::create_surface(&context.entry_point, &context.instance, window.raw_display_handle(), window.raw_window_handle(), None) }
///     .expect("Failed to create Vulkan surface");
///
/// let device_swapchain_info = get_swapchain_info(device, &surface, &extension);
/// let swapchain_parameters = get_swapchain_parameters(&device_swapchain_info, window);
/// ```
fn get_swapchain_parameters(swapchain_info: &SwapChainInfo, window: &winit::window::Window, preferred_surface_format: Option<(vk::Format, vk::ColorSpaceKHR)>, preferred_present_mode: Option<vk::PresentModeKHR>) -> SwapChainParameters {
    debug!("Selecting most appropriate swapchain parameters");

    // Only MacOS and iOS implement `UNDEFINED`, and only with `SRGB_NONLINEAR_KHR`, so `UNDEFINED`/<anything else> will never be found
    // https://www.vulkan.gpuinfo.org/listsurfaceformats.php
    let preferred = preferred_surface_format.unwrap_or((vk::Format::UNDEFINED, vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT));
    let format = swapchain_info.formats.iter().reduce(|accum, format| {
        if format.format == preferred.0 && format.color_space == preferred.1 {
            format
        } else if format.format == vk::Format::B8G8R8A8_UNORM && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            format
        } else if format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            format
        } else {
            accum
        }
    })
        .or(swapchain_info.formats.first())
        .expect("The device should support at least one surface format");
    debug!("Selected image format is {:?} with colour space {:?}", format.format, format.color_space);

    // Only Android implements `SHARED_*`, so it shouldn't show up on our targets
    let preferred = preferred_present_mode.unwrap_or(vk::PresentModeKHR::SHARED_DEMAND_REFRESH);
    let present_mode = swapchain_info.present_modes.iter().reduce(|accum, mode| {
        if *mode == preferred {
            mode
        } else if *mode == vk::PresentModeKHR::FIFO_RELAXED {
            mode
        } else if *mode == vk::PresentModeKHR::FIFO {
            mode
        } else {
            accum
        }
    })
        .expect("Could not find a valid present mode - FIFO should be supported");
    debug!("Selected present mode is {:?}", present_mode);

    let extent = {
        if swapchain_info.capabilities.current_extent.width != u32::MAX {
            swapchain_info.capabilities.current_extent
        } else {
            let window_size = window.inner_size();
            vk::Extent2D::builder()
                .width(num::clamp(window_size.width, swapchain_info.capabilities.min_image_extent.width, swapchain_info.capabilities.max_image_extent.width))
                .height(num::clamp(window_size.height, swapchain_info.capabilities.min_image_extent.height, swapchain_info.capabilities.max_image_extent.height))
                .build()
        }
    };
    debug!("Swapchain extent is {}x{}", extent.width, extent.height);

    SwapChainParameters {
        surface_format: *format,
        present_mode: *present_mode,
        extent,
    }
}

/// Gets information about the swapchain, based on the surface and device, which can be used by [get_swapchain_parameters()]
///
/// # Arguments
///
/// * `device`: The `Device` which will be rendering to the `surface`
/// * `surface`: The surface that the swapchain is being created for
/// * `surface_extension`: The instance of the `VK_KHR_surface` extension being used, as an `ash::extensions::khr::Surface`
///
/// # Examples
///
/// ```
/// use client::renderer::vulkan::{Context, Device};
/// use winit::{window::WindowBuilder, event_loop::EventLoopBuilder};
/// use ash::{vk, extensions};
///
/// let event_loop = EventLoopBuilder::new().build().unwrap();
/// let window = WindowBuilder::new().build(&event_loop).unwrap();
///
/// let context = Context::new("my-application", (1.4.2));
/// let device = Device::new(&context);
///
/// let extension = extensions::khr::Surface::new(&context.entry_point, &context.instance);
/// let surface = unsafe { ash_window::create_surface(&context.entry_point, &context.instance, window.raw_display_handle(), window.raw_window_handle(), None) }
///     .expect("Failed to create Vulkan surface");
///
/// let device_swapchain_info = get_swapchain_info(device, &surface, &extension);
/// ```
fn get_swapchain_info(device: &Device, surface: &vk::SurfaceKHR, surface_extension: &extensions::khr::Surface) -> SwapChainInfo {
    debug!("Getting device swapchain support");

    let capabilities = unsafe { surface_extension.get_physical_device_surface_capabilities(device.physical_device, *surface) }
        .expect("Failed to get physical device surface capabilities");

    let formats = unsafe { surface_extension.get_physical_device_surface_formats(device.physical_device, *surface) }
        .expect("Failed to get physical device surface formats");
    debug!("Device supports {} surface formats", formats.len());

    let present_modes = unsafe { surface_extension.get_physical_device_surface_present_modes(device.physical_device, *surface) }
        .expect("Failed to get physical device present modes");
    debug!("Device supports {} present modes", formats.len());

    SwapChainInfo {
        capabilities,
        formats,
        present_modes,
    }
}