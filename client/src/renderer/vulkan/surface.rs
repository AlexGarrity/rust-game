use std::rc::{Rc, Weak};
use ash::{extensions, vk};
use winit::window::raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use num;
use crate::renderer::vulkan::{Context, Device};
use tracing::{debug, debug_span};

struct SwapChainInfo {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct SwapChainParameters {
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D
}

pub struct Surface {
    device: Weak<ash::Device>,
    surface_extension: extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    swapchain_extension: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    pub swapchain_parameters: SwapChainParameters,
    swapchain_images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
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
    pub fn new(context: &Context, device: &Device, window: &winit::window::Window) -> Self {
        let span = debug_span!("Vulkan/Surface");
        let _guard = span.enter();

        let extension = extensions::khr::Surface::new(&context.entry_point, &context.instance);
        debug!("Creating SurfaceKHR");
        let surface = unsafe { ash_window::create_surface(&context.entry_point, &context.instance, window.raw_display_handle(), window.raw_window_handle(), None) }
            .expect("Failed to create Vulkan surface");
        debug!("Successfully created surface");

        let device_swapchain_info = get_swapchain_info(device, &surface, &extension);
        let swapchain_parameters = get_swapchain_parameters(&device_swapchain_info, window, None, None);

        let swapchain_extension = extensions::khr::Swapchain::new(&context.instance, &device.logical_device);
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

        let mut image_views = vec!();
        for image in &swapchain_images {
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
            image_views.push(image_view);
        }

        Surface {
            device: Rc::downgrade(&device.logical_device),
            surface_extension: extension,
            surface,
            swapchain_extension,
            swapchain,
            swapchain_parameters,
            swapchain_images,
            image_views,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Surface");
        let _guard = span.enter();

        for image_view in &self.image_views {
            debug!("Destroying image view {:?}", image_view);
            let device = self.device.upgrade().expect("Device should still exist");
            unsafe { device.destroy_image_view(*image_view, None) };
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
        }
        else if format.format == vk::Format::B8G8R8A8_UNORM && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            format
        }
        else if format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            format
        }
        else {
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
        }
        else if *mode == vk::PresentModeKHR::FIFO_RELAXED {
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
        extent
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