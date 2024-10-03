pub mod vkutil;

const REQUIRED_DEVICE_EXTENSIONS: [*const i8; 3] = [
    ash::khr::swapchain::NAME.as_ptr(),
    ash::khr::dynamic_rendering::NAME.as_ptr(),
    ash::ext::shader_object::NAME.as_ptr(),
];
const MAX_FRAMES_IN_FLIGHT: u32 = 3;

use ash::vk::{self, Buffer};

use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use gltf;

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
    global_ibo: vkutil::Buffer,
    global_vbo: vkutil::Buffer,
}

#[derive(Default)]
struct BufferSlice {
    size : u32,
    offset : u32
}

struct Object {
    pub name : String,
    pub transform : cgmath::Matrix4<f32>,
    pub gltf_document : gltf::Document,
    ibo_slice : BufferSlice,
    vbo_slice : BufferSlice
}

struct FrameData<'a> {
    commandpool: vk::CommandPool,
    semaphores: &'a [vk::Semaphore],
    fence: vk::Fence,
}

impl<'a> FrameData<'_> {
    fn new(engine: &'a Engine, indexer: usize) -> FrameData<'a> {
        FrameData {
            commandpool: engine.commandpools[indexer],
            semaphores: &engine.semaphores[indexer * 2..indexer * 2 + 2],
            fence: engine.fences[indexer],
        }
    }
}

impl Engine {
    pub fn new(disp_handle: RawDisplayHandle) -> Self {
        let required_instance_extensions = ash_window::enumerate_required_extensions(disp_handle)
            .expect("Failure to enumerate required extensions!");

        let context = vkutil::VkContextData::instanciateWithExtensions(&required_instance_extensions, &REQUIRED_DEVICE_EXTENSIONS);
    
        let global_ibo = vkutil::Buffer::new_from_size_and_flags(16 * 1024 * 1024, vk::BufferUsageFlags::INDEX_BUFFER, &context);
        let global_vbo = vkutil::Buffer::new_from_size_and_flags(16 * 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER, &context);

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
            global_ibo,
            global_vbo
        };

        eng.render_init();
        eng.scene_init();
        eng
    }

    fn render_init(&mut self) {
        println!("Initializing engine datastructures");
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
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
        let _fox_shader_pipeline = vkutil::Pipeline::load_gfx_pipeline_from_name(&fox_shader_name, &self.context);
    }

    pub fn window_init(&mut self, window: &winit::window::Window) {
        let surface_loader =
            ash::khr::surface::Instance::new(&self.context.entry, &self.context.instance);

        let surface = unsafe {
            ash_window::create_surface(
                &self.context.entry,
                &self.context.instance,
                window
                    .display_handle()
                    .unwrap()
                    .as_raw(),
                window
                    .window_handle()
                    .unwrap()
                    .as_raw(),
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

        let mut format = *surf_formats.first().unwrap();
        for form in &surf_formats {
            if form.format == vk::Format::B8G8R8A8_SRGB
                && form.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                format = form.clone();
            }
        }

        let mut present_mode = vk::PresentModeKHR::FIFO;
        for p in surf_present_modes {
            if p == vk::PresentModeKHR::MAILBOX {
                present_mode = vk::PresentModeKHR::MAILBOX;
            }
        }

        assert!(
            surf_capabilities.min_image_count <= 3 && surf_capabilities.max_image_count >= 3,
            "3 must be supported as swapchain size!"
        );
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

        let swap_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(3)
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
        // println!(
        //     "Acquired image {} : {:?} in frame {}",
        //     acquire_res, image_view_acquired, self.frame_index
        // );

        // allocate and begin command buffer work on frame
        let commandbuffer = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(frame_data.commandpool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmdbuf = self
                .context
                .device
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate commandbuffer")
                .into_iter()
                .nth(0)
                .unwrap();
            self.context
                .device
                .begin_command_buffer(cmdbuf, &vk::CommandBufferBeginInfo::default())
                .expect("Failure to start recording cmdbuf");

            cmdbuf
        };

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

        //end command buffer, submit work and present swapchain on queue
        unsafe {
            self.context
                .device
                .end_command_buffer(commandbuffer)
                .expect("Failure to end commandbuffer recording");

            let commandbuffers_submit = [commandbuffer];
            let framestart_sem = [frame_data.semaphores[0]];
            let frameend_sem = [frame_data.semaphores[1]];

            let queue_submit_info = vk::SubmitInfo::default()
                .command_buffers(&commandbuffers_submit)
                .wait_semaphores(&framestart_sem)
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

        unsafe {
            self.context
                .device
                .reset_command_pool(
                    frame_data.commandpool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failure to free commandpool");
        }
    }

    fn render_scene(&self, commandbuffer: &vk::CommandBuffer) {
        //scene rendering
    }

}

impl Drop for Engine {
    fn drop(&mut self) {
        self.global_ibo.destroy(&self.context);
        self.global_vbo.destroy(&self.context);
    }
}