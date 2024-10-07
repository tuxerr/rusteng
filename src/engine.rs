pub mod object;
pub mod vkutil;

const REQUIRED_DEVICE_EXTENSIONS: [*const i8; 2] = [
    ash::khr::swapchain::NAME.as_ptr(),
    ash::khr::dynamic_rendering::NAME.as_ptr(),
];
const MAX_FRAMES_IN_FLIGHT: u32 = 3;

use ash::vk::{self, Buffer, PipelineLayout};

use gltf;
use object::Object;
use std::cmp;
use std::rc::Rc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};

pub struct Engine {
    frame_index: i64,
    context: vkutil::VkContextData,
    surface: Option<vk::SurfaceKHR>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    commandpools: Vec<vk::CommandPool>,
    semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    objects: Vec<Object>,
    opaque_layout: PipelineLayout,
    global_ibo: vkutil::Buffer,
    global_vbo: vkutil::Buffer,
    global_scene: vkutil::Buffer,
    staging_buf: vkutil::Buffer,
}

struct FrameData<'a> {
    commandpool: vk::CommandPool,
    semaphores: &'a [vk::Semaphore],
    fence: vk::Fence,
}

impl<'a> FrameData<'_> {
    fn new(engine: &'a Engine, indexer: usize) -> FrameData<'a> {
        FrameData {
            commandpool: engine.commandpools[1 + indexer],
            semaphores: &engine.semaphores[1 + indexer * 2..indexer * 2 + 3],
            fence: engine.fences[1 + indexer],
        }
    }
}

impl Engine {
    pub fn new(disp_handle: RawDisplayHandle) -> Self {
        let required_instance_extensions = ash_window::enumerate_required_extensions(disp_handle)
            .expect("Failure to enumerate required extensions!");

        let context = vkutil::VkContextData::instanciateWithExtensions(
            &required_instance_extensions,
            &REQUIRED_DEVICE_EXTENSIONS,
        );

        let global_ibo = vkutil::Buffer::new_from_size_and_flags(
            16 * 1024 * 1024,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );
        let global_vbo = vkutil::Buffer::new_from_size_and_flags(
            16 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );
        let global_scene = vkutil::Buffer::new_from_size_and_flags(
            1 * 1024 * 1024 * MAX_FRAMES_IN_FLIGHT as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );
        let staging_buf = vkutil::Buffer::new_from_size_and_flags(
            8 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            &context,
        );

        //default engine-wide push constant layouts
        let opaque_mesh_pushconstant_layout = unsafe {
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

        let mut eng = Engine {
            frame_index: 0,
            context: context,
            surface: None,
            swapchain: None,
            swapchain_extent: vk::Extent2D::default(),
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            commandpools: Vec::new(),
            semaphores: Vec::new(),
            fences: Vec::new(),
            objects: Vec::new(),
            opaque_layout: opaque_mesh_pushconstant_layout,
            global_ibo,
            global_vbo,
            global_scene,
            staging_buf,
        };

        eng.render_init();
        eng.scene_init();
        eng
    }

    fn render_init(&mut self) {
        println!("Initializing engine datastructures");
        let structure_depth = MAX_FRAMES_IN_FLIGHT + 1;
        for _ in 0..structure_depth {
            let cmdpool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.context.queues.gfx_queue_idx)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);

            let cmdpool = unsafe {
                self.context
                    .device
                    .create_command_pool(&cmdpool_create_info, None)
                    .expect("Error allocating command pool")
            };
            self.commandpools.push(cmdpool);

            // 2 semaphores and a fence per in-flight frame
            unsafe {
                self.semaphores.push(
                    self.context
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .expect("Failure to create semaphore"),
                );
                self.semaphores.push(
                    self.context
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .expect("Failure to create semaphore"),
                );
                self.fences.push(
                    self.context
                        .device
                        .create_fence(
                            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                        .expect("Failure to create fence"),
                );
            }
        }
    }

    fn scene_init(&mut self) {
        let fox_shader_name = String::from("main");
        let fox_shader_pipeline = Rc::new(vkutil::Pipeline::load_gfx_pipeline_from_name(
            &fox_shader_name,
            &self.context,
        ));

        let fox_obj =
            object::Object::loadObjectInEngine(self, String::from("Fox"), fox_shader_pipeline);
        self.objects.push(fox_obj);
    }

    pub fn window_init(&mut self, window: &winit::window::Window) {
        let surface_loader =
            ash::khr::surface::Instance::new(&self.context.entry, &self.context.instance);

        let surface = unsafe {
            ash_window::create_surface(
                &self.context.entry,
                &self.context.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failure to instanciate VK_KHR_Surface for rendering")
        };

        let (surf_capabilities, surf_formats, surf_present_modes) = unsafe {
            (
                surface_loader
                    .get_physical_device_surface_capabilities(self.context.physical_device, surface)
                    .unwrap(),
                surface_loader
                    .get_physical_device_surface_formats(self.context.physical_device, surface)
                    .unwrap(),
                surface_loader
                    .get_physical_device_surface_present_modes(
                        self.context.physical_device,
                        surface,
                    )
                    .unwrap(),
            )
        };

        println!(
            "Surface capabilities : {:#?}, format {:#?}, presents {:#?}",
            surf_capabilities, surf_formats, surf_present_modes
        );

        let mut format = *surf_formats.first().unwrap();
        for form in &surf_formats {
            if form.format == vk::Format::B8G8R8A8_SRGB
                && form.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                format = form.clone();
            }
        }

        let mut present_mode = vk::PresentModeKHR::FIFO;
        /*for p in surf_present_modes {
            if p == vk::PresentModeKHR::MAILBOX {
                present_mode = vk::PresentModeKHR::MAILBOX;
            }
        }*/

        assert!(
            surf_capabilities
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT),
            "must support compute and gfx passes into swapchain!"
        );

        println!("Surface extent : {:?}", surf_capabilities.current_extent);
        println!("Surface capabilities : {:#?}", surf_capabilities);

        let pixel_size = window.inner_size();
        let swapchain_size = vk::Extent2D::default()
            .width(pixel_size.width.clamp(
                surf_capabilities.min_image_extent.width,
                surf_capabilities.max_image_extent.width,
            ))
            .height(pixel_size.height.clamp(
                surf_capabilities.min_image_extent.height,
                surf_capabilities.max_image_extent.height,
            ));

        let min_image_count = 3.clamp(
            surf_capabilities.min_image_count,
            cmp::max(surf_capabilities.max_image_count, 32),
        );

        let swap_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(swapchain_size)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let device_swapchain_loader =
            ash::khr::swapchain::Device::new(&self.context.instance, &self.context.device);
        let swapchain = unsafe {
            device_swapchain_loader
                .create_swapchain(&swap_create_info, None)
                .expect("Failure to create swap")
        };

        unsafe {
            self.swapchain_images = device_swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap();
        }

        for img in &self.swapchain_images {
            let img_view_create = vk::ImageViewCreateInfo::default()
                .image(*img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                );

            let img_view = unsafe {
                self.context
                    .device
                    .create_image_view(&img_view_create, None)
                    .expect("Failure to create image view")
            };
            self.swapchain_image_views.push(img_view);
        }

        self.swapchain = Some(swapchain);
        self.surface = Some(surface);
        self.swapchain_extent = swapchain_size;
    }

    pub fn render(&mut self) {
        self.frame_index += 1;
        if (self.frame_index > 5) {
            return;
        }

        let frame_indexer = (self.frame_index % MAX_FRAMES_IN_FLIGHT as i64) as usize;
        let dynrend_loader =
            ash::khr::dynamic_rendering::Device::new(&self.context.instance, &self.context.device);
        let swapchain_loader =
            ash::khr::swapchain::Device::new(&self.context.instance, &self.context.device);

        let frame_data = FrameData::new(self, frame_indexer);

        // wait on completion of previous frame in flight
        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame_data.fence], true, u64::MAX)
                .expect("Failure to wait on start-of-frame fence");
            self.context
                .device
                .reset_fences(&[frame_data.fence])
                .unwrap();
        }

        // swapchain acquire
        let (acquire_res, _) = unsafe {
            swapchain_loader
                .acquire_next_image(
                    self.swapchain.unwrap(),
                    u64::MAX,
                    frame_data.semaphores[0],
                    vk::Fence::null(),
                )
                .expect("Failure to acquire image")
        };
        let image_view_acquired = self.swapchain_image_views[acquire_res as usize];
        let image_acquired = self.swapchain_images[acquire_res as usize];
        println!(
            "Acquired image {} : {:?} in frame {}",
            acquire_res, image_view_acquired, self.frame_index
        );

        unsafe {
            self.context
                .device
                .reset_command_pool(
                    frame_data.commandpool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failure to free commandpool");
        }

        let commandbuffer = self
            .context
            .allocate_and_begin_commandbuffer(&frame_data.commandpool);

        // transitioning resources
        unsafe {
            let image_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .image(image_acquired)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

            self.context.device.cmd_pipeline_barrier(
                commandbuffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );
        }

        // start dynamic rendering
        unsafe {
            let color_attachments = [vk::RenderingAttachmentInfoKHR::default()
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.4, 0.4, 1.0],
                    },
                })
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image_view(image_view_acquired)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)];

            dynrend_loader.cmd_begin_rendering(
                commandbuffer,
                &vk::RenderingInfoKHR::default()
                    .color_attachments(&color_attachments)
                    .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
                    .layer_count(1),
            )
        }

        // actual render work
        self.render_scene(&commandbuffer);

        // end dynamic rendering
        unsafe {
            dynrend_loader.cmd_end_rendering(commandbuffer);
        }

        // transitioning resources back to present
        unsafe {
            let image_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .image(image_acquired)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

            self.context.device.cmd_pipeline_barrier(
                commandbuffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );
        }

        //end command buffer, submit work and present swapchain on queue
        unsafe {
            self.context
                .device
                .end_command_buffer(commandbuffer)
                .expect("Failure to end commandbuffer recording");

            let commandbuffers_submit = [commandbuffer];
            let framestart_sem = [frame_data.semaphores[0]];
            let framestart_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let frameend_sem = [frame_data.semaphores[1]];

            let queue_submit_info = vk::SubmitInfo::default()
                .command_buffers(&commandbuffers_submit)
                .wait_semaphores(&framestart_sem)
                .wait_dst_stage_mask(&framestart_stages)
                .signal_semaphores(&frameend_sem);

            self.context
                .device
                .queue_submit(
                    self.context.queues.gfx_queue,
                    &[queue_submit_info],
                    frame_data.fence,
                )
                .expect("Failure to submit commandbuffer into gfx queue");

            swapchain_loader
                .queue_present(
                    self.context.queues.present_queue,
                    &vk::PresentInfoKHR::default()
                        .image_indices(&[acquire_res])
                        .swapchains(&[self.swapchain.unwrap()])
                        .wait_semaphores(&frameend_sem),
                )
                .expect("Failure to present queue");
        }
    }

    fn render_scene(&self, commandbuffer: &vk::CommandBuffer) {
        //scene rendering
        let viewport = vk::Viewport::default()
            .max_depth(1.0f32)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32);
        let scissor = vk::Rect2D::default().extent(self.swapchain_extent);

        pub fn convert(data: &[u32; 4]) -> [u8; 16] {
            unsafe { std::mem::transmute(*data) }
        }

        unsafe {
            self.context
                .device
                .cmd_set_viewport(*commandbuffer, 0, &[viewport]);
            self.context
                .device
                .cmd_set_scissor(*commandbuffer, 0, &[scissor]);
            self.context.device.cmd_bind_index_buffer(
                *commandbuffer,
                self.global_ibo.vk_buffer,
                0,
                vk::IndexType::UINT32,
            );

            let push_constant_addresses = [
                self.global_vbo.buffer_address,
                self.global_scene.buffer_address
            ];
            let push_constant_addresses_u8 : [u8; 16] = std::mem::transmute(push_constant_addresses);

            self.context.device.cmd_push_constants(
                *commandbuffer,
                self.opaque_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &push_constant_addresses_u8[..],
            );
        }

        for obj in &self.objects {
            unsafe {
                self.context.device.cmd_bind_pipeline(
                    *commandbuffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    obj.pipeline.vk_pipeline,
                );

                let index_count = obj.ibo_slice.size / std::mem::size_of::<u32>() as u32;
                self.context
                    .device
                    .cmd_draw_indexed(*commandbuffer, index_count, 1, 0, 0, 0);
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.global_ibo.destroy(&self.context);
        self.global_vbo.destroy(&self.context);
        self.staging_buf.destroy(&self.context);
    }
}
