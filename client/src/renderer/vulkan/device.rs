use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::rc::Rc;

use ash::vk;
use tracing::{debug, debug_span};

use crate::renderer::vulkan::surface::MAX_FRAMES_IN_FLIGHT;
use crate::renderer::vulkan::{Context, Pipeline, Surface};

struct DeviceQueueTriplet<T> {
    graphics: T,
    present: T,
    transfer: T,
    compute: T,
}

struct QueueFamilyInfo {
    index: u32,
    count: u32,
}

type DeviceQueueFamilyIndices = DeviceQueueTriplet<QueueFamilyInfo>;

// TODO - The vk::Queues probably need wrapping in RwLocks because otherwise
// a queue could be written to in multiple places at the same time if queues are shared
type DeviceQueues = DeviceQueueTriplet<Vec<vk::Queue>>;
type DeviceCommandPools = DeviceQueueTriplet<vk::CommandPool>;
type DeviceCommandBuffers = DeviceQueueTriplet<Vec<vk::CommandBuffer>>;

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: Rc<ash::Device>,
    _queue_family_indices: DeviceQueueFamilyIndices,
    queue_families: DeviceQueues,
    pipelines: HashMap<String, Pipeline>,
    command_pools: DeviceCommandPools,
    command_buffers: DeviceCommandBuffers,
}

impl Device {
    /// Constructs a new Device, based on some rough heuristics to guess which is best.
    /// The device will be constructed with separate queues for graphics, transfer, and compute if possible, but otherwise they will be shared
    ///
    /// # Arguments
    ///
    /// * `context`: The `Context` to create the device using
    ///
    /// # Examples
    ///
    /// ```
    /// use client::renderer::vulkan::{Context, Device};
    ///
    /// let context = new Context("my-application", (1.4.2));
    /// let device = Device::new(&context);
    /// ```
    pub fn new(context: &Context, surface: &Surface) -> Device {
        let span = debug_span!("Vulkan/Device");
        let _guard = span.enter();

        let physical_devices = unsafe { context.instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices");

        // TODO - Expand this. Some people still have multi-GPU setups and it would be nice to be able to support that

        let physical_device = physical_devices
            .iter()
            .reduce(|accum, current| {
                let device_type =
                    unsafe { context.instance.get_physical_device_properties(*current) }
                        .device_type;
                let current_memory = get_device_local_memory_size(context, current);
                let accum_memory = get_device_local_memory_size(context, accum);

                if device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
                    accum
                } else {
                    if current_memory > accum_memory {
                        current
                    } else {
                        accum
                    }
                }
            })
            .expect("Failed to select a physical device");

        debug!("Selected physical device {:?}", unsafe {
            CStr::from_ptr(
                context
                    .instance
                    .get_physical_device_properties(*physical_device)
                    .device_name
                    .as_ptr(),
            )
        });

        let current_memory = get_device_local_memory_size(context, physical_device);
        debug!(
            "Device has {} GB of dedicated memory",
            current_memory / (1024 * 1024 * 1024)
        );

        let queue_family_indices = find_device_queues_indices(context, physical_device, surface);
        debug!(
            "Selected queue index {} for graphics, {} for present, {} for transfer, and {} for compute",
            queue_family_indices.graphics.index,
            queue_family_indices.present.index,
            queue_family_indices.transfer.index,
            queue_family_indices.compute.index
        );

        // TODO - Handle what happens when the same queue family is requested multiple times.
        // The same queue family index can only be found in one QueueCreateInfo, unless the queue family has the protected bit set
        // Most devices don't have it set, so it's probably not worth factoring in
        // Consider converting the Queues inside DeviceQueues to Rcs and Weaks

        let mut queue_create_infos = vec![];

        let mut indices_used = HashSet::new();

        let graphics_queue_priorities: Vec<f32> = (0..queue_family_indices.graphics.count)
            .map(|_| 1.0)
            .collect();
        let graphics_queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_indices.graphics.index)
            .queue_priorities(graphics_queue_priorities.as_slice())
            .build();

        queue_create_infos.push(graphics_queue_create_info);
        indices_used.insert(queue_family_indices.graphics.index);

        if !indices_used.contains(&queue_family_indices.transfer.index) {
            let transfer_queue_priorities: Vec<f32> = (0..queue_family_indices.transfer.count)
                .map(|_| 1.0)
                .collect();
            let transfer_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_indices.transfer.index)
                .queue_priorities(transfer_queue_priorities.as_slice())
                .build();
            queue_create_infos.push(transfer_queue_create_info);
            indices_used.insert(queue_family_indices.transfer.index);
        }

        if !indices_used.contains(&queue_family_indices.compute.index) {
            let compute_queue_priorities: Vec<f32> = (0..queue_family_indices.compute.count)
                .map(|_| 1.0)
                .collect();
            let compute_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_indices.compute.index)
                .queue_priorities(compute_queue_priorities.as_slice())
                .build();
            queue_create_infos.push(compute_queue_create_info);
            indices_used.insert(queue_family_indices.compute.index);
        }

        // We do present last, since we only make one queue
        if !indices_used.contains(&queue_family_indices.present.index) {
            let present_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_indices.present.index)
                .queue_priorities(&[1.0])
                .build();
            queue_create_infos.push(present_queue_create_info);
            indices_used.insert(queue_family_indices.present.index);
        }

        let device_feature_info = vk::PhysicalDeviceFeatures::builder().build();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&[ash::extensions::khr::Swapchain::name().as_ptr()])
            .enabled_features(&device_feature_info)
            .queue_create_infos(queue_create_infos.as_slice())
            .build();

        debug!("Creating logical device");
        let logical_device = unsafe {
            context
                .instance
                .create_device(*physical_device, &device_create_info, None)
        }
        .expect("Failed to create a logical device");
        debug!("Successfully created logical device");

        let queue_families = create_device_queues(&logical_device, &queue_family_indices);
        debug!(
            "Created {} queues for graphics, {} queues for present, {} queues for transfer, and {} queues for compute",
            queue_families.graphics.len(),
            queue_families.present.len(),
            queue_families.transfer.len(),
            queue_families.compute.len()
        );

        let command_pools = create_command_pools(&logical_device, &queue_family_indices);
        let command_buffers = create_command_buffers(&logical_device, &command_pools);

        Device {
            physical_device: *physical_device,
            logical_device: Rc::new(logical_device),
            _queue_family_indices: queue_family_indices,
            queue_families,
            pipelines: HashMap::new(),
            command_pools,
            command_buffers,
        }
    }

    /// Constructs a new graphics pipeline on the device, referencable by the name provided
    ///
    /// If the device already has a pipeline with the given name or insertion fails, returns `false`
    ///
    /// If the device does not have a pipeline with the given name and insertion succeeds, returns `true`
    ///
    /// # Arguments
    ///
    /// * `surface`: The `Surface` that the `Pipeline` should render to
    /// * `vertex_shader_path`: A `Path` which references a compiled SPIR-V vertex shader, relative to the application executable
    /// * `fragment_shader_path`: A `Path` which references a compiled SPIR-V vertex shader, relative to the application executable
    /// * `name`: The name that the `Pipeline` should be referencable as later
    ///
    /// # Examples
    ///
    /// ```
    /// use winit::{window::WindowBuilder, event_loop::EventLoopBuilder};
    /// use client::renderer::vulkan::{Context, Device, Surface};
    /// use std::path::Path;
    ///
    /// let event_loop = EventLoopBuilder::new().build().unwrap();
    /// let window = WindowBuilder::new().build(&event_loop).unwrap();
    ///
    /// let context = Context::new("my-application", (1.4.2));
    /// let device = Device::new(&context);
    /// let surface = Surface::new(&context, &device, &window);
    ///
    /// let result = device.create_pipeline(&surface, Path::new("vertex_shader.spv"), Path::new("fragment_shader.spv"), String::from("my_shader"));
    /// match result {
    ///     true => println!("Successfully created and attached pipeline"),
    ///     false => println("Failed to create shader")
    /// }
    ///
    /// let result = device.create_pipeline(&surface, Path::new("vertex_shader_2.spv"), Path::new("fragment_shader_2.spv"), String::from("my_shader"));
    /// assert_eq!(result, false);
    /// ```
    pub fn create_pipeline(
        &mut self,
        surface: &Surface,
        vertex_shader_path: &std::path::Path,
        fragment_shader_path: &std::path::Path,
        name: String,
    ) -> Result<(), &'static str> {
        if self.pipelines.contains_key(name.as_str()) {
            Err("A pipeline already exists with the specified name")
        } else if !vertex_shader_path.exists() {
            Err("A shader file could not be found at the specified path")
        } else if !fragment_shader_path.exists() {
            Err("A shader file could not be found at the specified path")
        } else {
            if self
                .pipelines
                .insert(
                    name,
                    Pipeline::new(self, surface, vertex_shader_path, fragment_shader_path),
                )
                .is_some()
            {
                Ok(())
            } else {
                Err("Failed to insert pipeline into device map")
            }
        }
    }

    /// Get a pipeline by name
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the `Pipeline` to get
    ///
    pub fn get_pipeline(&self, name: &str) -> Option<&Pipeline> {
        self.pipelines.get(name)
    }

    pub fn begin_graphics_render_pass(
        &self,
        current_frame: usize,
        surface: &mut Surface,
        pipeline_name: &str,
    ) -> u32 {
        let command_buffer = *self.command_buffers.graphics.get(current_frame).unwrap();
        let frame_in_flight = *surface.frame_in_flight.get(current_frame).unwrap();

        unsafe {
            self.logical_device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
        }
        .expect("Device was removed or timed out whilst waiting for a fence");
        unsafe { self.logical_device.reset_fences(&[frame_in_flight]) }
            .expect("Could not reset fence");

        let image_index = surface.acquire_next_image();

        unsafe {
            self.logical_device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .expect("Failed to reset graphics command buffer");

        let command_buffer_info = vk::CommandBufferBeginInfo::builder().build();
        unsafe {
            self.logical_device
                .begin_command_buffer(command_buffer, &command_buffer_info)
        }
        .expect("Failed to begin graphics command buffer)");

        let pipeline = self
            .get_pipeline(pipeline_name)
            .expect("Failed to get graphics pipeline");

        let framebuffer = surface.get_framebuffer(image_index as usize).clone();
        let clear_values = vk::ClearValue::default();

        let scissor = vk::Rect2D::builder()
            .extent(surface.swapchain_parameters.as_ref().unwrap().extent)
            .offset(vk::Offset2D::builder().x(0).y(0).build())
            .build();

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(pipeline.render_pass)
            .framebuffer(framebuffer)
            .clear_values(&[clear_values])
            .render_area(scissor)
            .build();

        unsafe {
            self.logical_device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            )
        };

        unsafe {
            self.logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            )
        }

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(surface.swapchain_parameters.as_ref().unwrap().extent.width as f32)
            .height(surface.swapchain_parameters.as_ref().unwrap().extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        unsafe {
            self.logical_device
                .cmd_set_viewport(command_buffer, 0, &[viewport])
        };
        unsafe {
            self.logical_device
                .cmd_set_scissor(command_buffer, 0, &[scissor])
        };

        image_index
    }

    pub fn submit_graphics_queue(
        &self,
        frame_index: usize,
        signal_semaphores: &[vk::Semaphore],
        wait_semaphores: &[vk::Semaphore],
        stage_flags: &[vk::PipelineStageFlags],
        wait_fence: &vk::Fence,
    ) {
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[*self.command_buffers.graphics.get(frame_index).unwrap()])
            .signal_semaphores(signal_semaphores)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(stage_flags)
            .build();

        // FIXME - Validation error `VUID-vkQueueSubmit-fence-00064` (fence is already in use by another submission)
        // Probably because I forgot that present queues are a thing
        unsafe {
            self.logical_device.queue_submit(
                *self.queue_families.graphics.get(frame_index).unwrap(),
                &[submit_info],
                *wait_fence,
            )
        }
        .expect("Failed to submit graphics queue");
    }

    pub fn present_queue(
        &self,
        frame_index: usize,
        swapchain_ext: &ash::extensions::khr::Swapchain,
        present_info: &vk::PresentInfoKHR,
    ) {
        // FIXME - Use the present queue for this
        unsafe {
            swapchain_ext.queue_present(
                *self.queue_families.graphics.get(frame_index).unwrap(),
                present_info,
            )
        }
        .expect("Failed to present graphics queue");
    }

    pub fn draw_vertices(&mut self, frame_index: usize, vertex_count: u32) {
        let command_buffer = *self.command_buffers.graphics.get(frame_index).unwrap();
        unsafe {
            self.logical_device
                .cmd_draw(command_buffer, vertex_count, 1, 0, 0)
        };
    }

    pub fn end_graphics_render_pass(&mut self, frame_index: usize) {
        let command_buffer = *self.command_buffers.graphics.get(frame_index).unwrap();
        unsafe { self.logical_device.cmd_end_render_pass(command_buffer) };
        unsafe { self.logical_device.end_command_buffer(command_buffer) }
            .expect("Failed to end graphics command buffer")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Device");
        let _guard = span.enter();

        unsafe {
            self.logical_device.free_command_buffers(
                self.command_pools.graphics,
                self.command_buffers.graphics.as_slice(),
            )
        };
        unsafe {
            self.logical_device.free_command_buffers(
                self.command_pools.present,
                &self.command_buffers.present.as_slice(),
            )
        };
        unsafe {
            self.logical_device.free_command_buffers(
                self.command_pools.transfer,
                self.command_buffers.transfer.as_slice(),
            )
        };
        unsafe {
            self.logical_device.free_command_buffers(
                self.command_pools.compute,
                self.command_buffers.compute.as_slice(),
            )
        };

        unsafe {
            self.logical_device
                .destroy_command_pool(self.command_pools.graphics, None)
        };
        unsafe {
            self.logical_device
                .destroy_command_pool(self.command_pools.present, None)
        };
        unsafe {
            self.logical_device
                .destroy_command_pool(self.command_pools.transfer, None)
        };
        unsafe {
            self.logical_device
                .destroy_command_pool(self.command_pools.compute, None)
        };

        self.pipelines.clear();

        debug!("Destroying logical device");
        unsafe {
            self.logical_device.destroy_device(None);
        }
        debug!("Successfully destroyed device");
    }
}

fn create_command_buffers(
    device: &ash::Device,
    command_pools: &DeviceCommandPools,
) -> DeviceCommandBuffers {
    let graphics_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
        .command_pool(command_pools.graphics)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let graphics = unsafe { device.allocate_command_buffers(&graphics_buffer_allocate_info) }
        .expect("Failed to allocate primary graphics command buffer");

    let present_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
        .command_pool(command_pools.present)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let present = unsafe { device.allocate_command_buffers(&present_buffer_allocate_info) }
        .expect("Failed to allocate primary present command buffer");

    let transfer_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
        .command_pool(command_pools.transfer)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let transfer = unsafe { device.allocate_command_buffers(&transfer_buffer_allocate_info) }
        .expect("Failed to allocate primary transfer command buffer");

    let compute_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
        .command_pool(command_pools.compute)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let compute = unsafe { device.allocate_command_buffers(&compute_buffer_allocate_info) }
        .expect("Failed to allocate primary compute command buffer");

    DeviceCommandBuffers {
        graphics,
        present,
        transfer,
        compute,
    }
}

fn create_command_pools(
    device: &ash::Device,
    queue_family_indices: &DeviceQueueFamilyIndices,
) -> DeviceCommandPools {
    let graphics_queue_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.graphics.index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    let graphics = unsafe { device.create_command_pool(&graphics_queue_pool_create_info, None) }
        .expect("Failed to create graphics command pool");

    let present_queue_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.present.index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    let present = unsafe { device.create_command_pool(&present_queue_pool_create_info, None) }
        .expect("Failed to create present command pool");

    let transfer_queue_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.transfer.index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    let transfer = unsafe { device.create_command_pool(&transfer_queue_pool_create_info, None) }
        .expect("Failed to create transfer command pool");

    let compute_queue_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.compute.index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    let compute = unsafe { device.create_command_pool(&compute_queue_pool_create_info, None) }
        .expect("Failed to create graphics command pool");

    DeviceCommandPools {
        graphics,
        present,
        transfer,
        compute,
    }
}

/// Gets the queues from a logical device, given a list of queue indices
///
/// # Arguments
///
/// * `device`: The logical device to get the queues from
/// * `indices`: The indices of the queues to get
///
/// # Examples
///
/// ```
/// use client::renderer::vulkan::Context;
/// use ash::vk;
///
/// let context = Context::new("my-application", (1.4.2));
/// let physical_device = unsafe { context.instance.enumerate_physical_devices() }
///     .unwrap()
///     .first()
///     .unwrap();
///
/// let queue_family_indices = find_device_queues_indices(context, physical_device);
///
/// let graphics_queue_priorities: Vec<f32> = (0..queue_family_indices.graphics.1).map(|_| { 1.0 }).collect();
/// let graphics_queue_create_info = vk::DeviceQueueCreateInfo::builder()
///     .queue_family_index(queue_family_indices.graphics.0)
///     .queue_priorities(graphics_queue_priorities.as_slice())
///     .build();
///
/// let transfer_queue_priorities: Vec<f32> = (0..queue_family_indices.transfer.1).map(|_| { 1.0 }).collect();
/// let transfer_queue_create_info = vk::DeviceQueueCreateInfo::builder()
///     .queue_family_index(queue_family_indices.transfer.0)
///     .queue_priorities(transfer_queue_priorities.as_slice())
///     .build();
///
/// let compute_queue_priorities: Vec<f32> = (0..queue_family_indices.compute.1).map(|_| { 1.0 }).collect();
/// let compute_queue_create_info = vk::DeviceQueueCreateInfo::builder()
///     .queue_family_index(queue_family_indices.compute.0)
///     .queue_priorities(compute_queue_priorities.as_slice())
///     .build();
///
/// let device_feature_info = vk::PhysicalDeviceFeatures::builder()
///     .build();
///
/// let device_create_info = vk::DeviceCreateInfo::builder()
///     .enabled_extension_names(&[
///         ash::extensions::khr::Swapchain::name().as_ptr()
///     ])
///     .enabled_features(&device_feature_info)
///     .queue_create_infos(&[graphics_queue_create_info, transfer_queue_create_info, compute_queue_create_info])
///     .build();
///
/// let logical_device = unsafe { context.instance.create_device(*physical_device, &device_create_info, None) }
///     .expect("Failed to create a logical device");
/// ```
fn create_device_queues(device: &ash::Device, indices: &DeviceQueueFamilyIndices) -> DeviceQueues {
    // TODO - It's possible that this will retrieve the same queues multiple times.
    // Whilst I don't think it's necessarily harmful, I can't imagine that calling
    // free on the same n queues multiple times is something that the API likes

    DeviceQueues {
        graphics: (0..indices.graphics.count)
            .map(|i| unsafe { device.get_device_queue(indices.graphics.index, i) })
            .collect(),
        present: (0..indices.present.count)
            .map(|i| unsafe { device.get_device_queue(indices.present.index, i) })
            .collect(),
        transfer: (0..indices.transfer.count)
            .map(|i| unsafe { device.get_device_queue(indices.transfer.index, i) })
            .collect(),
        compute: (0..indices.compute.count)
            .map(|i| unsafe { device.get_device_queue(indices.compute.index, i) })
            .collect(),
    }
}

/// Gets the indices required to create graphics, transfer, and compute queues.
/// The function will attempt to get unique queue family indices if possible
/// (ie. 3 <type>-only queues) but will otherwise fallback to whichever queue family
/// has the most queues available
///
/// # Arguments
///
/// * `context`: The `Context` to use when querying the physical device
/// * `device`: The physical device to create the queues on
///
/// # Examples
/// ```
/// use client::renderer::vulkan::Context;
/// use ash::vk;
///
/// let context = Context::new("my-application", (1.4.2));
/// let physical_device = unsafe { context.instance.enumerate_physical_devices() }
///     .unwrap()
///     .first()
///     .unwrap();
///
/// let queue_family_indices = find_device_queues_indices(context, physical_device);
/// ```
fn find_device_queues_indices(
    context: &Context,
    device: &vk::PhysicalDevice,
    surface: &Surface,
) -> DeviceQueueFamilyIndices {
    let surface_extension = &surface.surface_extension;

    let queue_properties = unsafe {
        context
            .instance
            .get_physical_device_queue_family_properties(*device)
    };

    // Find the best graphics queue possible - high queue count and graphics supported
    let graphics_queue = queue_properties
        .iter()
        .enumerate()
        .reduce(|accum, current| {
            if !current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                accum
            } else if current.1.queue_count < accum.1.queue_count {
                accum
            } else {
                current
            }
        })
        .expect("Failed to find a valid graphics queue");

    let present_queue = {
        let graphics_queue_surface_support = unsafe {
            surface_extension.get_physical_device_surface_support(
                *device,
                graphics_queue.0 as u32,
                surface.surface,
            )
        }
        .unwrap();

        if graphics_queue_surface_support {
            (graphics_queue.0, graphics_queue.1)
        } else {
            queue_properties
                .iter()
                .enumerate()
                .reduce(|accum, current| {
                    let queue_surface_support = unsafe {
                        surface_extension.get_physical_device_surface_support(
                            *device,
                            current.0 as u32,
                            surface.surface,
                        )
                    }
                    .unwrap();
                    let accum_surface_support = unsafe {
                        surface_extension.get_physical_device_surface_support(
                            *device,
                            accum.0 as u32,
                            surface.surface,
                        )
                    }
                    .unwrap();

                    if queue_surface_support && !accum_surface_support {
                        current
                    } else if accum_surface_support && !queue_surface_support {
                        accum
                    } else if current.1.queue_count > accum.1.queue_count {
                        current
                    } else {
                        accum
                    }
                })
                .unwrap()
        }
    };

    let transfer_queue = queue_properties
        .iter()
        .enumerate()
        .reduce(|accum, current| {
            if !current.1.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                // If current doesn't support transfer, fallback to accum
                accum
            } else {
                let accum_is_mixed = accum.1.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    || accum.1.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let current_is_mixed = current.1.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    || current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let current_has_more_queues = current.1.queue_count >= accum.1.queue_count;

                if !accum_is_mixed && current_is_mixed {
                    // Accum is a dedicated queue where current is not
                    accum
                } else if accum_is_mixed && !current_is_mixed {
                    // Current is a dedicated queue where accum is not
                    current
                } else if current_has_more_queues {
                    // Either both queues are dedicated, or both are mixed, so pick current if it has more queues
                    current
                } else {
                    accum
                }
            }
        })
        .expect("Failed to find a valid transfer queue");

    let compute_queue = queue_properties
        .iter()
        .enumerate()
        .reduce(|accum, current| {
            if !current.1.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                // Same logic as above
                accum
            } else if accum.1.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                current
            } else if current.1.queue_count > accum.1.queue_count {
                current
            } else {
                accum
            }
        })
        .expect("Failed to find a valid compute queue");

    let graphics = QueueFamilyInfo {
        index: graphics_queue.0 as u32,
        count: graphics_queue.1.queue_count,
    };
    let present = QueueFamilyInfo {
        index: present_queue.0 as u32,
        count: present_queue.1.queue_count,
    };
    let transfer = QueueFamilyInfo {
        index: transfer_queue.0 as u32,
        count: transfer_queue.1.queue_count,
    };
    let compute = QueueFamilyInfo {
        index: compute_queue.0 as u32,
        count: compute_queue.1.queue_count,
    };

    DeviceQueueFamilyIndices {
        graphics,
        present,
        transfer,
        compute,
    }
}

/// Gets the size of the device-local memory on a physical device (ie. the dedicated GDDRX / HBM memory)
///
/// # Arguments
///
/// * `context`: The `Context` the physical device was queried from
/// * `device`: The physical device to get the size of the memory of
///
/// # Examples
///
/// ```
/// use client::renderer::vulkan::Context;
/// use ash::vk;
///
/// let context = Context::new("my-application", (1.4.2));
/// let physical_device = unsafe { context.instance.enumerate_physical_devices() }
///     .unwrap()
///     .first()
///     .unwrap();
///
/// let current_memory = get_device_local_memory_size(context, physical_device);
/// ```
fn get_device_local_memory_size(context: &Context, device: &vk::PhysicalDevice) -> u64 {
    let device_memory_properties = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(*device)
    };
    let heap_info = &device_memory_properties.memory_heaps;

    // FIXME - This isn't foolproof, and will return multiple GB on an iGPU despite the fact that it's shared
    heap_info
        .iter()
        .enumerate()
        .map(|heap| {
            if heap.0 >= device_memory_properties.memory_heap_count as usize {
                0u64
            } else {
                if heap.1.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    heap.1.size as u64
                } else {
                    0u64
                }
            }
        })
        .collect::<Vec<u64>>()
        .iter()
        .sum()
}
