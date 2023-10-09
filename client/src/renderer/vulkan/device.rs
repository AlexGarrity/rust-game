use std::collections::HashMap;
use std::ffi::CStr;
use std::rc::Rc;

use ash::vk;
use tracing::{debug, debug_span};

use crate::renderer::vulkan::{Context, Pipeline, Surface};

struct DeviceQueueIndices {
    graphics: (u32, u32),
    transfer: (u32, u32),
    compute: (u32, u32),
}

struct DeviceQueues {
    graphics: Vec<vk::Queue>,
    transfer: Vec<vk::Queue>,
    compute: Vec<vk::Queue>,
}

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: Rc<ash::Device>,
    queue_family_indices: DeviceQueueIndices,
    queue_families: DeviceQueues,
    pipelines: HashMap<String, Pipeline>,
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
    pub fn new(context: &Context) -> Self {
        let span = debug_span!("Vulkan/Device");
        let _guard = span.enter();

        let physical_devices = unsafe { context.instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices");

        // TODO - Expand this. Some people still have multi-GPU setups

        let physical_device = physical_devices.iter().reduce(|accum, current| {
            let device_type = unsafe { context.instance.get_physical_device_properties(*current) }.device_type;
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
        }).expect("Failed to select a physical device");

        debug!("Selected physical device {:?}", unsafe { CStr::from_ptr( context.instance.get_physical_device_properties(*physical_device).device_name.as_ptr()) });

        let current_memory = get_device_local_memory_size(context, physical_device);
        debug!("Device has {} GB of dedicated memory", current_memory / (1024 * 1024 * 1024));

        let queue_family_indices = find_device_queues_indices(context, physical_device);
        debug!("Selected queue index {} for graphics, {} for transfer, and {} for compute", queue_family_indices.graphics.0, queue_family_indices.transfer.0, queue_family_indices.compute.0);

        // TODO - Handle what happens when the same queue family is requested multiple times.
        // The same queue family index can only be found in one QueueCreateInfo, unless the queue family has the protected bit set
        // Most devices don't have it set, so it's probably not worth factoring in
        // Consider converting the Queues inside DeviceQueues to Rcs and Weaks

        let graphics_queue_priorities: Vec<f32> = (0..queue_family_indices.graphics.1).map(|_| { 1.0 }).collect();
        let graphics_queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_indices.graphics.0)
            .queue_priorities(graphics_queue_priorities.as_slice())
            .build();

        let transfer_queue_priorities: Vec<f32> = (0..queue_family_indices.transfer.1).map(|_| { 1.0 }).collect();
        let transfer_queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_indices.transfer.0)
            .queue_priorities(transfer_queue_priorities.as_slice())
            .build();

        let compute_queue_priorities: Vec<f32> = (0..queue_family_indices.compute.1).map(|_| { 1.0 }).collect();
        let compute_queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_indices.compute.0)
            .queue_priorities(compute_queue_priorities.as_slice())
            .build();

        let device_feature_info = vk::PhysicalDeviceFeatures::builder()
            .build();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&[
                ash::extensions::khr::Swapchain::name().as_ptr()
            ])
            .enabled_features(&device_feature_info)
            .queue_create_infos(&[graphics_queue_create_info, transfer_queue_create_info, compute_queue_create_info])
            .build();

        debug!("Creating logical device");
        let logical_device = unsafe { context.instance.create_device(*physical_device, &device_create_info, None) }
            .expect("Failed to create a logical device");
        debug!("Successfully created logical device");

        let queue_families = create_device_queues(&logical_device, &queue_family_indices);
        debug!("Created {} queues for graphics, {} queues for transfer, and {} queues for compute",
            queue_families.graphics.len(),
            queue_families.transfer.len(),
            queue_families.compute.len()
        );

        Device {
            physical_device: *physical_device,
            logical_device: Rc::new(logical_device),
            queue_family_indices,
            queue_families,
            pipelines: HashMap::new(),
        }
    }

    /// Constructs a new graphics pipeline on the device, referencable by the name provided
    ///
    /// If the device already has a pipeline with the given name or insertion fails, returns `false`
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
    pub fn create_pipeline(&mut self, surface: &Surface, vertex_shader_path: &std::path::Path, fragment_shader_path: &std::path::Path, name: String) -> bool {
        if self.pipelines.contains_key(name.as_str()) {
            false
        } else {
            self.pipelines.insert(name, Pipeline::new(self, surface, vertex_shader_path, fragment_shader_path))
                .is_some()
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Device");
        let _guard = span.enter();

        self.pipelines.clear();

        debug!("Destroying logical device");
        unsafe { self.logical_device.destroy_device(None); }
        debug!("Successfully destroyed device");
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
fn create_device_queues(device: &ash::Device, indices: &DeviceQueueIndices) -> DeviceQueues {
    DeviceQueues {
        graphics: (0..indices.graphics.1).map(|i| {
            unsafe { device.get_device_queue(indices.graphics.0, i) }
        })
            .collect(),
        transfer: (0..indices.transfer.1).map(|i| {
            unsafe { device.get_device_queue(indices.transfer.0, i) }
        })
            .collect(),
        compute: (0..indices.compute.1).map(|i| {
            unsafe { device.get_device_queue(indices.compute.0, i) }
        })
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
fn find_device_queues_indices(context: &Context, device: &vk::PhysicalDevice) -> DeviceQueueIndices {
    let queue_properties = unsafe { context.instance.get_physical_device_queue_family_properties(*device) };

    // Find the best graphics queue possible - high queue count and graphics supported
    let graphics_queue = queue_properties.iter().enumerate().reduce(|accum, current| {
        if !current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            accum
        } else if current.1.queue_count < accum.1.queue_count {
            accum
        } else {
            current
        }
    })
        .expect("Failed to find a valid graphics queue");

    let transfer_queue = queue_properties.iter().enumerate().reduce(|accum, current| {
        if !current.1.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            // If current doesn't support transfer, fallback to accum
            accum
        } else {
            let accum_is_mixed = accum.1.queue_flags.contains(vk::QueueFlags::COMPUTE) || accum.1.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let current_is_mixed = current.1.queue_flags.contains(vk::QueueFlags::COMPUTE) || current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS);
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

    let compute_queue = queue_properties.iter().enumerate().reduce(|accum, current| {
        if !current.1.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            // Same logic as above
            accum
        } else if accum.1.queue_flags.contains(vk::QueueFlags::GRAPHICS) && !current.1.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            current
        } else if current.1.queue_count > accum.1.queue_count {
            current
        } else {
            accum
        }
    })
        .expect("Failed to find a valid compute queue");

    let graphics = (graphics_queue.0 as u32, graphics_queue.1.queue_count);
    let transfer = (transfer_queue.0 as u32, transfer_queue.1.queue_count);
    let compute = (compute_queue.0 as u32, compute_queue.1.queue_count);

    DeviceQueueIndices {
        graphics,
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
    let device_memory_properties = unsafe { context.instance.get_physical_device_memory_properties(*device) };
    let heap_info = &device_memory_properties.memory_heaps;

    heap_info.iter().enumerate().map(|heap| {
        if heap.0 >= device_memory_properties.memory_heap_count as usize {
            0u64
        }
        else {
            if heap.1.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                heap.1.size as u64
            }
            else {
                0u64
            }
        }
    })
        .collect::<Vec<u64>>()
        .iter()
        .sum()
}