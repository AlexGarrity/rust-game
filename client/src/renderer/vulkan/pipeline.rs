use crate::renderer::vulkan::{Device, Surface};
use ash::vk;
use byteorder::{LittleEndian, ReadBytesExt};
use std::ffi::CString;
use std::rc::{Rc, Weak};
use tracing::{debug, debug_span, warn};

pub struct Pipeline {
    device: Weak<ash::Device>,
    layout: vk::PipelineLayout,
    cache: vk::PipelineCache,
    pub render_pass: vk::RenderPass,
    pub(crate) pipeline: vk::Pipeline,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
}

impl Pipeline {
    /// Constructs a new graphics `Pipeline` using the provided shaders.
    /// Note that the recommended way to create a pipeline is through [`Device::create_pipeline()`]) rather than using `Pipeline::new()` directly
    ///
    /// # Arguments
    ///
    /// * `device`: The `Device` to construct the `Pipeline` on
    /// * `surface`: The `Surface` that the `Pipeline` should render to
    /// * `vertex_shader_path`: A `Path` which references a compiled SPIR-V vertex shader, relative to the application executable
    /// * `fragment_shader_path`: A `Path` which references a compiled SPIR-V vertex shader, relative to the application executable
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
    /// let pipeline = Pipeline::new(&surface, Path::new("vertex_shader.spv"), Path::new("fragment_shader.spv"), String::from("my_shader"));
    /// ```
    pub fn new(
        device: &Device,
        surface: &Surface,
        vertex_shader_path: &std::path::Path,
        fragment_shader_path: &std::path::Path,
    ) -> Self {
        let vertex_shader_module = load_shader(device, vertex_shader_path)
            .expect("The vertex shader either wasn't found, or was invalid");
        let fragment_shader_module = load_shader(device, fragment_shader_path)
            .expect("The vertex shader either wasn't found, or was invalid");

        let shader_entry_point: CString = CString::new("main").unwrap();

        let vertex_shader_state_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .name(shader_entry_point.as_c_str())
            .module(vertex_shader_module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .build();

        let fragment_shader_state_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .name(shader_entry_point.as_c_str())
            .module(fragment_shader_module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let pipeline_layout = create_pipeline_layout(device);
        let pipeline_cache = create_pipeline_cache(device);
        let render_pass = create_render_pass(device, surface);
        let graphics_pipeline = create_graphics_pipeline(
            device,
            surface,
            &pipeline_layout,
            &render_pass,
            &pipeline_cache,
            vertex_shader_state_create_info,
            fragment_shader_state_create_info,
        );

        Pipeline {
            device: Rc::downgrade(&device.logical_device),
            layout: pipeline_layout,
            cache: pipeline_cache,
            render_pass,
            pipeline: graphics_pipeline,
            vertex_shader: vertex_shader_module,
            fragment_shader: fragment_shader_module,
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let span = debug_span!("Vulkan/~Pipeline");
        let _guard = span.enter();

        let device = self.device.upgrade().expect("Device should still exist");

        debug!("Destroying pipeline");
        unsafe { device.destroy_pipeline(self.pipeline, None) };
        debug!("Destroying render pass");
        unsafe { device.destroy_render_pass(self.render_pass, None) };
        debug!("Destroying pipeline cache");
        unsafe { device.destroy_pipeline_cache(self.cache, None) };
        debug!("Destroying pipeline layout");
        unsafe { device.destroy_pipeline_layout(self.layout, None) };
        debug!("Destroying vertex shader module");
        unsafe { device.destroy_shader_module(self.vertex_shader, None) };
        debug!("Destroying fragment shader module");
        unsafe { device.destroy_shader_module(self.fragment_shader, None) };
    }
}

/// Constructs an `ash::vk::PipelineLayout` with default parameters
///
/// # Arguments
///
/// * `device`: The `Device` to create the pipeline layout for
///
fn create_pipeline_layout(device: &Device) -> vk::PipelineLayout {
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder().build();

    unsafe {
        device
            .logical_device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
    }
    .expect("Failed to create Vulkan pipeline")
}

/// Constructs an `ash::vk::PipelineCache` with default parameters
///
/// # Arguments
///
/// * `device`: The `Device` to create the pipeline layout for
///
fn create_pipeline_cache(device: &Device) -> vk::PipelineCache {
    let pipeline_cache_create_info = vk::PipelineCacheCreateInfo::builder().build();

    unsafe {
        device
            .logical_device
            .create_pipeline_cache(&pipeline_cache_create_info, None)
    }
    .expect("Failed to create Vulkan pipeline cache")
}

/// Constructs an `ash::vk::RenderPass` with default parameters
///
/// # Arguments
///
/// * `device`: The `Device` to create the pipeline layout for
/// * `surface`: The `Surface` that the render pass should render to
///
fn create_render_pass(device: &Device, surface: &Surface) -> vk::RenderPass {
    let colour_attachment = vk::AttachmentDescription::builder()
        .format(
            surface
                .swapchain_parameters
                .as_ref()
                .unwrap()
                .surface_format
                .format,
        )
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();

    let colour_attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&[colour_attachment_reference])
        .build();

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build();

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&[colour_attachment])
        .subpasses(&[subpass])
        .dependencies(&[subpass_dependency])
        .build();

    unsafe {
        device
            .logical_device
            .create_render_pass(&render_pass_create_info, None)
    }
    .expect("Failed to create Vulkan render pass")
}

/// Constructs an `ash::vk::Pipeline` with default parameters, using the provided shaders
///
/// # Arguments
///
/// * `device`: The `Device` to create the pipeline on
/// * `surface`: The `Surface` that the pipeline should render to
/// * `pipeline_layout`: The pipeline layout to make the pipeline according to
/// * `render_pass`: The render pass the pipeline should use
/// * `pipeline_cache`: The pipeline cache that the pipeline should use
/// * `vertex_shader`: The `PipelineShaderStageCreateInfo` for the vertex shader that the pipeline should use
/// * `fragment_shader`: The `PipelineShaderStageCreateInfo` for the fragment shader that the pipeline should use
///
/// # Examples
///
/// ```
/// use ash::vk;
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
///
/// let vertex_shader_module = load_shader(device, vertex_shader_path).unwrap();
/// let fragment_shader_module = load_shader(device, fragment_shader_path).unwrap();
///
/// let shader_entry_point: CString = CString::new("main").unwrap();
///
/// let vertex_shader_state_create_info = vk::PipelineShaderStageCreateInfo::builder()
///     .name(shader_entry_point.as_c_str())
///     .module(vertex_shader_module)
///     .stage(vk::ShaderStageFlags::VERTEX)
///     .build();
///
/// let fragment_shader_state_create_info = vk::PipelineShaderStageCreateInfo::builder()
///     .name(shader_entry_point.as_c_str())
///     .module(fragment_shader_module)
///     .stage(vk::ShaderStageFlags::FRAGMENT)
///     .build();
///
/// let pipeline_layout = create_pipeline_layout(&device);
/// let pipeline_cache = create_pipeline_cache(&device);
/// let render_pass = create_render_pass(&device, &surface);
/// let graphics_pipeline = create_graphics_pipeline(
///     &device,
///     &surface,
///     &pipeline_layout,
///     &render_pass,
///     &pipeline_cache,
///     vertex_shader_state_create_info,
///     fragment_shader_state_create_info
/// );
/// ```
fn create_graphics_pipeline(
    device: &Device,
    surface: &Surface,
    pipeline_layout: &vk::PipelineLayout,
    render_pass: &vk::RenderPass,
    pipeline_cache: &vk::PipelineCache,
    vertex_shader: vk::PipelineShaderStageCreateInfo,
    fragment_shader: vk::PipelineShaderStageCreateInfo,
) -> vk::Pipeline {
    // let vertex_input_attribute_description = vk::VertexInputAttributeDescription::builder()
    //     .format(surface.swapchain_parameters.surface_format.format)
    //     .location(0)
    //     .build();

    // let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
    //     .binding(0)
    //     .build();

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[])
        .build();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(surface.swapchain_parameters.as_ref().unwrap().extent.width as f32)
        .height(surface.swapchain_parameters.as_ref().unwrap().extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build();

    let scissor = vk::Rect2D::builder()
        .extent(surface.swapchain_parameters.as_ref().unwrap().extent)
        .offset(vk::Offset2D::builder().x(0).y(0).build())
        .build();

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .scissors(&[scissor])
        .viewports(&[viewport])
        .build();

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .depth_bias_enable(false)
        .build();

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false)
        .build();

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder().build();

    let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(true)
        .color_blend_op(vk::BlendOp::ADD)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .build();

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&[color_blend_attachment_state])
        .build();

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT])
        .build();

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&[fragment_shader, vertex_shader])
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(*pipeline_layout)
        .render_pass(*render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null())
        .build();

    *unsafe {
        device.logical_device.create_graphics_pipelines(
            *pipeline_cache,
            &[pipeline_create_info],
            None,
        )
    }
    .expect("Failed to create Vulkan graphics pipeline")
    .first()
    .expect("Pipeline creation was successful, but returned no pipeline object")
}

/// Attempts to load a shader file from the `Path` provided, and creates a shader module using it.
///
///
/// If the file existed, contained valid bytecode, and was compiled successfully, returns `Some<ash::vk::ShaderModule>`
///
/// If the file doesn't exist, the bytecode is invalid, or compilation fails, returns `None`
///
/// # Arguments
///
/// * `device`: The `Device` to create the shader module on
/// * `relative_file_path`: A `Path` referencing a compiled SPIR-V shader file, relative to the application executable
///
/// # Examples
///
/// ```
/// use client::renderer::vulkan::{Context, Device};
///
/// let context = new Context("my-application", (1.4.2));
/// let device = Device::new(&context);
///
/// let vertex_shader_module = load_shader(&device, Path::new("vertex_shader.spv"))
///     .expect("Something went wrong whilst trying to load the shader");
///
/// ```
fn load_shader(device: &Device, relative_file_path: &std::path::Path) -> Option<vk::ShaderModule> {
    let current_exe = std::env::current_exe();
    let joined_file_path = current_exe
        .unwrap()
        .parent()
        .unwrap()
        .join(relative_file_path);
    let absolute_file_path = joined_file_path.as_path();

    if !absolute_file_path.exists() {
        warn!(
            "Tried to load a shader at {:?} but it does not exist",
            absolute_file_path
        );
        None
    } else {
        let code_as_bytes = std::fs::read(absolute_file_path).expect("Failed to read file");

        let mut cursor = std::io::Cursor::new(&code_as_bytes);
        let mut code = Vec::<u32>::new();
        code.resize(code_as_bytes.len() / 4, 0);
        let _res = cursor.read_u32_into::<LittleEndian>(code.as_mut_slice());

        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(code.as_slice())
            .build();

        let shader_module = unsafe {
            device
                .logical_device
                .create_shader_module(&shader_module_create_info, None)
        }
        .expect("Failed to create shader module");

        Some(shader_module)
    }
}
