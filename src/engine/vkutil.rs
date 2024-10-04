use ash::{
    khr::dynamic_rendering_local_read,
    vk::{
        self, AttachmentLoadOp, CommandBufferLevel, ComponentMapping, DeviceQueueCreateInfo,
        Extent2D, Handle, Queue, QueueFlags, Rect2D,
    },
    Entry,
};

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
    pub vma_alloc: vk_mem::Allocator
}

impl VkContextData {
    pub fn instanciateWithExtensions(required_instance_exts: &[*const i8], required_device_exts : &[*const i8]) -> Self {
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

        let available_layers = unsafe {
            entry.enumerate_instance_layer_properties().expect("Failure to enumerate layers")
        };
        println!("Available layers : {:#?}", available_layers);
        let enabled_layer_names = [VKVALID_NAME.as_ptr()];

        if available_layers.into_iter().any(|layer| layer.layer_name_as_c_str().unwrap().eq(VKVALID_NAME)) {
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
        let (physical_device, (gfxqidx, transferqidx, presentqidx)) = unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failure to enumerate PDs");
            let pd = physical_devices[0];

            let pdprop = instance.get_physical_device_properties(pd);
            let pdqueues = instance.get_physical_device_queue_family_properties(pd);
            let mut gfx: Option<u32> = None;
            let mut transfer: Option<u32> = None;
            let mut present: Option<u32> = None;

            for (index, pdqueue) in pdqueues.iter().enumerate() {
                println!("Found queue : {:#?} at index {}", pdqueue, index);
                if (pdqueue
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                    && gfx.is_none())
                {
                    gfx = Some(index.try_into().unwrap());
                }
                if (pdqueue.queue_flags.contains(QueueFlags::TRANSFER) && transfer.is_none()) {
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

            println!(
                "Found GPU : {:#?} and queue indices for gfx ({}) and transfer({}) and present({})",
                pdprop,
                gfx.unwrap(),
                transfer.unwrap(),
                present.unwrap(),
            );
            (
                physical_devices[0],
                (gfx.unwrap(), transfer.unwrap(), present.unwrap()),
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

        let mut physicalfeatures2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enable_dynrender)
            .push_next(&mut enable_bda);

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

        let mut vma_alloc_create_info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
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
            vma_alloc
        }
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
                    .size(16) // 2 buffer device addresses for mesh and object data
            ];
            let layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(&push_constant_ranges);
            context.device.create_pipeline_layout(&layout_create_info, None).expect("Failure to create pipeline layout")
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

        let viewport_state = vk::PipelineViewportStateCreateInfo::default();

        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0f32);

        let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        // opaque draw
        let color_blend_attachments = [
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
        ];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachments);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let mut rendering_state = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&[vk::Format::B8G8R8A8_SRGB])
            .depth_attachment_format(vk::Format::D24_UNORM_S8_UINT)
            .stencil_attachment_format(vk::Format::D24_UNORM_S8_UINT);

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
    size : usize,
    pub mem_alloc : vk_mem::Allocation,
    pub vk_buffer : vk::Buffer,
}

impl Buffer {
    pub fn new_from_size_and_flags(size : usize, flags : vk::BufferUsageFlags, context: &VkContextData) -> Self {
        let mut local_flags = flags;
        if(local_flags.contains(vk::BufferUsageFlags::STORAGE_BUFFER) | local_flags.contains(vk::BufferUsageFlags::INDEX_BUFFER)) {
            local_flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(local_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vma_create_info = vk_mem::AllocationCreateInfo {
            usage : vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (buf,alloc) = unsafe {
            context.vma_alloc.create_buffer(&buffer_create_info, &vma_create_info).expect("Failure to create buffer")
        };

        Buffer {
            size: size,
            mem_alloc: alloc,
            vk_buffer : buf
        }
    }

    pub fn destroy(&mut self, context: &VkContextData) {
        unsafe {
            context.vma_alloc.destroy_buffer(self.vk_buffer, &mut self.mem_alloc);
        }
    }
}