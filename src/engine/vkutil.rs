use ash::vk::{
    self, ImageAspectFlags, ImageLayout, ImageTiling, ImageType, QueueFlags, SampleCountFlags,
};
use winit::event::ElementState;

use std::fs::File;
use std::path::Path;

use std::{ffi::CStr, ffi::CString, u64};

use vk_mem::{self, Alloc}; // vma allocator for buffers/textures

const VKAPP_NAME: &str = "FoxyGfx";
const VKVALID_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct Queues {
    pub gfx_queue_idx: u32,
    pub transfer_queue_idx: u32,
    pub present_queue_idx: u32,
    pub gfx_queue: ash::vk::Queue,
    pub transfer_queue: ash::vk::Queue,
    pub present_queue: ash::vk::Queue,
}

pub struct VkContextData {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device: ash::Device,
    pub queues: Queues,
    pub vma_alloc: vk_mem::Allocator,
}

impl VkContextData {
    pub fn instanciateWithExtensions(
        required_instance_exts: &[*const i8],
        required_device_exts: &[*const i8],
    ) -> Self {
        println!("Initializing VK graphics");
        let entry = ash::Entry::linked();

        let app_info = vk::ApplicationInfo {
            api_version: vk::API_VERSION_1_3,
            p_application_name: VKAPP_NAME.as_ptr().cast(),
            ..Default::default()
        };
        let mut instance_create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        }
        .enabled_extension_names(required_instance_exts);

        for ext in required_instance_exts {
            let ext_str = unsafe { CStr::from_ptr(ext.clone()) };
            println!("Ext : {:#?}", ext_str);
        }

        let available_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Failure to enumerate layers")
        };

        let available_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failure to fetch extensions")
        };
        for (index, ext) in available_extensions.iter().enumerate() {
            println!(
                "Extension {} is {:#?}",
                index,
                ext.extension_name_as_c_str().unwrap()
            );
        }

        let enabled_layer_names = [VKVALID_NAME.as_ptr()];
        if available_layers
            .into_iter()
            .any(|layer| layer.layer_name_as_c_str().unwrap().eq(VKVALID_NAME))
        {
            instance_create_info = instance_create_info.enabled_layer_names(&enabled_layer_names);
        }

        // instance create
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failure to create instance")
        };

        // let surfacekhr_win32_loader = ash::khr::win32_surface::Instance::new(&entry, &instance);

        // physical device create and queue retrieval
        let (physical_device, (gfxqidx, transferqidx, presentqidx), pdfeatures) = unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failure to enumerate PDs");
            let pd = physical_devices[0];

            let pdprop = instance.get_physical_device_properties(pd);
            let pdfeatures = instance.get_physical_device_features(pd);
            let pdqueues = instance.get_physical_device_queue_family_properties(pd);
            let mut gfx: Option<u32> = None;
            let mut transfer: Option<u32> = None;
            let mut present: Option<u32> = None;

            for (index, pdqueue) in pdqueues.iter().enumerate() {
                println!("Found queue : {:#?} at index {}", pdqueue, index);
                if pdqueue
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                    && gfx.is_none()
                {
                    gfx = Some(index.try_into().unwrap());
                }
                if pdqueue.queue_flags.contains(QueueFlags::TRANSFER) && transfer.is_none() {
                    transfer = Some(index.try_into().unwrap());
                }

                /*if (surfacekhr_win32_loader
                .get_physical_device_win32_presentation_support(pd, index.try_into().unwrap())
                && present.is_none())*/
                {
                    present = Some(index.try_into().unwrap());
                }

                if gfx.is_some() && transfer.is_some() && present.is_some() {
                    break;
                }
            }

            gfx.expect("Expected to find graphics queue at this point");
            transfer.expect("Expected to find transfer queue at this point");
            present.expect("Expected to find presentation queue at this point");

            // println!(
            //     "Found GPU : {:#?} and queue indices for gfx ({}) and transfer({}) and present({})",
            //     pdprop,
            //     gfx.unwrap(),
            //     transfer.unwrap(),
            //     present.unwrap(),
            // );
            (
                physical_devices[0],
                (gfx.unwrap(), transfer.unwrap(), present.unwrap()),
                pdfeatures,
            )
        };

        // device create
        let gfx_queue_create_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gfxqidx)
            .queue_priorities(&[1.0])];

        assert!(
            gfxqidx == presentqidx,
            "Currently only supporting graphics queues also handling present"
        );

        // enable dynamic rendering feature
        let mut enable_dynrender =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut enable_bda =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut enable_variablepointers = vk::PhysicalDeviceVariablePointersFeatures::default()
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true);
        let mut enable_shaderint64 = vk::PhysicalDeviceShaderAtomicInt64Features::default()
            .shader_buffer_int64_atomics(true)
            .shader_shared_int64_atomics(true);

        let mut physicalfeatures2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enable_dynrender)
            .push_next(&mut enable_bda)
            .push_next(&mut enable_variablepointers)
            .push_next(&mut enable_shaderint64)
            .features(pdfeatures);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&gfx_queue_create_info)
            .enabled_extension_names(&required_device_exts)
            .push_next(&mut physicalfeatures2);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failure to create device")
        };

        let gfx_queue = unsafe { device.get_device_queue(gfxqidx, 0) };
        let present_queue = unsafe { device.get_device_queue(presentqidx, 0) };
        // let transfer_queue = unsafe { device.get_device_queue(1, 1) };

        let queues = Queues {
            gfx_queue_idx: gfxqidx,
            transfer_queue_idx: transferqidx,
            present_queue_idx: presentqidx,
            gfx_queue: gfx_queue,
            transfer_queue: gfx_queue,
            present_queue: present_queue,
        };

        let mut vma_alloc_create_info =
            vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        vma_alloc_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        vma_alloc_create_info.vulkan_api_version = vk::API_VERSION_1_3;

        let vma_alloc = unsafe {
            vk_mem::Allocator::new(vma_alloc_create_info).expect("Failure to instanciate VMA")
        };

        VkContextData {
            entry,
            instance,
            physical_device,
            device,
            queues,
            vma_alloc,
        }
    }

    pub fn allocate_and_begin_commandbuffer(
        &self,
        pool: &ash::vk::CommandPool,
    ) -> ash::vk::CommandBuffer {
        // allocate and begin command buffer work on frame
        let commandbuffer = unsafe {
            //allocate cmdbuf
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(*pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmdbuf = self
                .device
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate commandbuffer")
                .into_iter()
                .nth(0)
                .unwrap();

            //begin cmdbuf
            self.device
                .begin_command_buffer(cmdbuf, &vk::CommandBufferBeginInfo::default())
                .expect("Failure to start recording cmdbuf");

            cmdbuf
        };
        commandbuffer
    }
}

enum ShaderType {
    FRAGMENT,
    VERTEX,
    COMPUTE,
    MESH,
}

struct Shader {
    name: String,
    shader_type: ShaderType,
    shader_module: vk::ShaderModule,
}

impl Shader {
    fn loadFromNameAndType(
        name: &String,
        shader_type: ShaderType,
        context: &VkContextData,
    ) -> Self {
        let prefix = match shader_type {
            ShaderType::FRAGMENT => "fs",
            ShaderType::VERTEX => "vs",
            ShaderType::COMPUTE => "comp",
            ShaderType::MESH => "mesh",
        };
        let shader_path_str = format!("shaders/{}{}.o", prefix, name);
        let shader_path = Path::new(&shader_path_str);

        let mut file = File::open(shader_path).expect("Failure to open shader");
        let spv_words = ash::util::read_spv(&mut file).expect("Failure to read file");

        let shader_module = unsafe {
            let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
            context
                .device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        };

        Shader {
            name: name.clone(),
            shader_type,
            shader_module,
        }
    }
}

pub struct Pipeline {
    name: String,
    pub vk_pipeline: vk::Pipeline,
}

impl Pipeline {
    pub fn load_gfx_pipeline_from_name(name: &String, context: &VkContextData) -> Self {
        let vs = Shader::loadFromNameAndType(&name, ShaderType::VERTEX, context);
        let fs = Shader::loadFromNameAndType(&name, ShaderType::FRAGMENT, context);

        let shadername = CString::new("main").unwrap();

        let opaque_mesh_layout = unsafe {
            let push_constant_ranges = [
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .offset(0)
                    .size(16), // 2 buffer device addresses for mesh and object data
            ];
            let layout_create_info =
                vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
            context
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .expect("Failure to create pipeline layout")
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs.shader_module)
                .name(&shadername.as_c_str()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs.shader_module)
                .name(&shadername.as_c_str()),
        ];

        let vtx_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = [vk::Viewport::default()
            .max_depth(1.0f32)
            .width(512 as f32)
            .height(512 as f32)];
        let scissor =
            [vk::Rect2D::default().extent(vk::Extent2D::default().width(512).height(512))];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewport)
            .scissors(&scissor);

        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::FRONT)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0f32);

        let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        // opaque draw
        let color_blend_attachments =
            [vk::PipelineColorBlendAttachmentState::default().blend_enable(false).color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let mut rendering_state = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&[vk::Format::B8G8R8A8_SRGB])
            .depth_attachment_format(vk::Format::D32_SFLOAT)
            .stencil_attachment_format(vk::Format::UNDEFINED);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vtx_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster_state)
            .multisample_state(&msaa_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(opaque_mesh_layout)
            .push_next(&mut rendering_state);

        let gfxpipe = unsafe {
            context
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .expect("Failure to generate pipelines")
        }[0];

        Pipeline {
            name: name.clone(),
            vk_pipeline: gfxpipe,
        }
    }
}

pub struct Buffer {
    pub size: usize,
    pub mem_alloc: vk_mem::Allocation,
    pub vk_buffer: vk::Buffer,
    pub buffer_address: u64, //buffer_device_address allocation
}

impl Buffer {
    pub fn new_from_size_and_flags(
        size: usize,
        flags: vk::BufferUsageFlags,
        vma_flags: vk_mem::AllocationCreateFlags,
        context: &VkContextData,
    ) -> Self {
        let mut local_flags = flags;
        if local_flags.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
            | local_flags.contains(vk::BufferUsageFlags::INDEX_BUFFER)
        {
            local_flags |= vk::BufferUsageFlags::TRANSFER_DST;
            local_flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }

        if local_flags.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            local_flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(local_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vma_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vma_flags,
            ..Default::default()
        };

        let (buf, alloc) = unsafe {
            context
                .vma_alloc
                .create_buffer(&buffer_create_info, &vma_create_info)
                .expect("Failure to create buffer")
        };

        let mut buf_addr = 0;
        if local_flags.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            buf_addr = unsafe {
                let bda_info = vk::BufferDeviceAddressInfo::default().buffer(buf);

                context.device.get_buffer_device_address(&bda_info)
            };
        }

        Buffer {
            size: size,
            mem_alloc: alloc,
            vk_buffer: buf,
            buffer_address: buf_addr,
        }
    }

    pub fn destroy(&mut self, context: &VkContextData) {
        unsafe {
            context
                .vma_alloc
                .destroy_buffer(self.vk_buffer, &mut self.mem_alloc);
        }
    }
}

pub struct Texture {
    pub extent: vk::Extent2D,
    pub mem_alloc: vk_mem::Allocation,
    pub vk_image: vk::Image,
    pub vk_imageview: vk::ImageView,
}

impl Texture {
    pub fn new_from_extent_format_and_flags(
        extent: vk::Extent2D,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        vma_flags: vk_mem::AllocationCreateFlags,
        context: &VkContextData,
    ) -> Self {
        let (initial_layout, aspect) = if format == vk::Format::D32_SFLOAT {
            (
                ImageLayout::UNDEFINED,
                ImageAspectFlags::DEPTH,
            )
        } else {
            (ImageLayout::UNDEFINED, ImageAspectFlags::COLOR)
        };

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(SampleCountFlags::TYPE_1)
            .tiling(ImageTiling::OPTIMAL)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(initial_layout);

        let vma_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vma_flags,
            ..Default::default()
        };

        let (image, alloc) = unsafe {
            context
                .vma_alloc
                .create_image(&image_create_info, &vma_create_info)
                .expect("Failure to create image")
        };

        let img_view_create = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .layer_count(1)
                    .level_count(1),
            );

        let img_view = unsafe {
            context
                .device
                .create_image_view(&img_view_create, None)
                .expect("Failure to create image view")
        };

        Texture {
            extent: extent,
            mem_alloc: alloc,
            vk_image: image,
            vk_imageview: img_view,
        }
    }

    pub fn destroy(&mut self, context: &VkContextData) {
        unsafe {
            context
                .vma_alloc
                .destroy_image(self.vk_image, &mut self.mem_alloc);
        }
    }
}
