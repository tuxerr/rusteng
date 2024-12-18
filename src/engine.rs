pub mod object;
pub mod shader_struct;
pub mod vkutil;

const REQUIRED_DEVICE_EXTENSIONS: [*const i8; 3] = [
    ash::khr::swapchain::NAME.as_ptr(),
    ash::khr::dynamic_rendering::NAME.as_ptr(),
    ash::ext::mesh_shader::NAME.as_ptr(),
];
const MAX_FRAMES_IN_FLIGHT: u32 = 3;
const MAX_BINDLESS_TEXTURES: u32 = 100000;

use ash::vk::{self, PipelineLayout};

use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3};
use object::Object;
use shader_struct::VertexEntry;
use vkutil::PipelineType;
use std::collections::HashMap;
use std::rc::Rc;
use std::{cmp, u32};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};

pub struct Engine {
    frame_index: i64,
    context: vkutil::VkContextData,
    surface: Option<vk::SurfaceKHR>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    depth_buffer: Option<vkutil::Texture>,
    commandpools: Vec<vk::CommandPool>,
    semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    objects: Vec<Object>,
    view_matrix: cgmath::Matrix4<f32>,
    bindless_texture_descriptorset: vk::DescriptorSet,
    bindless_handle: u32,
    shader_map: HashMap<String, vkutil::Pipeline>,
    global_ibo: vkutil::Buffer,
    global_vbo: vkutil::Buffer,
    global_meshlets: vkutil::Buffer,
    global_primitives: vkutil::Buffer,
    staging_buf: vkutil::Buffer,
    indirect_draw_buf: vkutil::Buffer,
    render_mode: RenderMode,
}

struct FrameData {
    commandpool: vk::CommandPool,
    semaphores: Vec<vk::Semaphore>,
    fence: vk::Fence,
}

impl FrameData {
    fn new(engine: &Engine, indexer: usize) -> FrameData {
        FrameData {
            commandpool: engine.commandpools[1 + indexer],
            semaphores: Vec::from(&engine.semaphores[1 + indexer * 2..indexer * 2 + 3]),
            fence: engine.fences[1 + indexer],
        }
    }
}

#[derive(PartialEq)]
enum RenderMode {
    Draw,
    DrawIndirect,
    Meshlet,
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

        let global_meshlets = vkutil::Buffer::new_from_size_and_flags(
            16 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );

        let global_primitives = vkutil::Buffer::new_from_size_and_flags(
            16 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );

        let staging_buf = vkutil::Buffer::new_from_size_and_flags(
            64 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            &context,
        );

        // buffer for vkCmdDrawIndexedIndirectCount
        let indirect_draw_buf = vkutil::Buffer::new_from_size_and_flags(
            1 * 1024 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &context,
        );

        let mut shader_map = HashMap::new();

        let opaque_mesh_shader_name = String::from("main");
        let opaque_meshlet_compute_shader =
            vkutil::Pipeline::load_mesh_pipeline_from_name(&opaque_mesh_shader_name, &context);

        shader_map.insert(opaque_mesh_shader_name, opaque_meshlet_compute_shader);

        let borrowed_opaque = &shader_map["main"];

        let descriptor_pool_sizes = [vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_BINDLESS_TEXTURES)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes);

        let descriptor_set = unsafe {
            let pool = context
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failure to create descriptor pool!");

            let bindless_texture_descriptorset_layouts =
                [borrowed_opaque.vk_descriptorset_layouts[0]];
            let descriptor_set_allocinfo = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&bindless_texture_descriptorset_layouts);

            context
                .device
                .allocate_descriptor_sets(&descriptor_set_allocinfo)
                .expect("Failure to allocate bindless descriptor set")[0]
        };

        let mut eng = Engine {
            frame_index: 0,
            context: context,
            surface: None,
            swapchain: None,
            swapchain_extent: vk::Extent2D::default(),
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            depth_buffer: None,
            commandpools: Vec::new(),
            semaphores: Vec::new(),
            fences: Vec::new(),
            objects: Vec::new(),
            view_matrix: cgmath::Matrix4::identity(),
            bindless_texture_descriptorset: descriptor_set,
            bindless_handle: 0,
            shader_map,
            global_ibo,
            global_vbo,
            global_meshlets,
            global_primitives,
            staging_buf,
            indirect_draw_buf,
            render_mode: RenderMode::Meshlet,
        };

        eng.render_init();
        eng.scene_init();
        eng
    }

    fn get_pipeline(&mut self, pipeline_name : &str, pipeline_type : PipelineType) -> vkutil::Pipeline {
        let pipe_name = String::from(pipeline_name);
        if !self.shader_map.contains_key(&pipe_name)
        {
            let new_shader = match pipeline_type {
                PipelineType::MESH => vkutil::Pipeline::load_mesh_pipeline_from_name(&pipe_name, &self.context),
                PipelineType::VERTEX => vkutil::Pipeline::load_vertex_pipeline_from_name(&pipe_name, &self.context),
                PipelineType::COMPUTE => vkutil::Pipeline::load_compute_pipeline_from_name(&pipe_name, &self.context),
            };
            self.shader_map.insert(pipe_name.clone(), new_shader);
        }

        self.shader_map[&pipe_name].clone()
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
        let opaque_pbr_shader_name = String::from("main");
        let opaque_pbr_shader = Rc::new(vkutil::Pipeline::load_mesh_pipeline_from_name(
            &opaque_pbr_shader_name,
            &self.context
        ));

        /*let mut helmet_obj = object::Object::loadObjectInEngine(
            self,
            String::from("DamagedHelmet.glb"),
            opaque_pbr_shader.clone(),
        );
        helmet_obj.transform = helmet_obj.transform * Matrix4::from_angle_x(Deg(90.0));
        self.objects.push(helmet_obj); */

        let mut fox_obj = object::Object::loadObjectInEngine(
            self,
            String::from("Sponza/Sponza.gltf"),
            //String::from("ToyCar.glb"),
            opaque_pbr_shader.clone(),
        );
        //fox_obj.transform = fox_obj.transform * Matrix4::from_angle_x(Deg(90.0));

        fox_obj.transform = fox_obj.transform * Matrix4::from_scale(0.02f32);

        let cam_transform = Matrix4::look_at_rh(
            Point3::new(-27.0f32, 2.3f32, -1.5f32),
            Point3::new(15.0f32, 8.0f32, 0.0f32),
            Vector3::new(0.0f32, -1.0f32, 0.0f32),
        );
        /*let cam_transform = Matrix4::look_at_rh(
            Point3::new(-15.0f32, 2.3f32, -1.5f32),
            Point3::new(0.0f32, 0.0f32, 0.0f32),
            Vector3::new(0.0f32, -1.0f32, 0.0f32),
        );*/

        self.view_matrix = cam_transform;
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

        let present_mode = vk::PresentModeKHR::FIFO;

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
        self.depth_buffer = Some(vkutil::Texture::new_from_extent_format_and_flags(
            swapchain_size,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            &self.context,
        ));
    }

    pub fn execute_synchronous_on_queue<F>(&self, queue: ash::vk::Queue, mut closure: F)
    where
        F: FnMut(ash::vk::CommandBuffer) -> (),
    {
        let commandbuffer = self
            .context
            .allocate_and_begin_commandbuffer(&self.commandpools[0]);

        closure(commandbuffer);

        unsafe {
            self.context
                .device
                .end_command_buffer(commandbuffer)
                .expect("Failure to end commandbuffer recording");

            self.context.device.reset_fences(&[self.fences[0]]).unwrap();

            let commandbuffers_submit = [commandbuffer];
            let queue_submit_info =
                vk::SubmitInfo::default().command_buffers(&commandbuffers_submit);
            self.context
                .device
                .queue_submit(
                    self.context.queues.gfx_queue,
                    &[queue_submit_info],
                    self.fences[0],
                )
                .expect("Failure to submit commandbuffer into gfx queue");

            self.context
                .device
                .wait_for_fences(&[self.fences[0]], true, u64::MAX)
                .expect("Failure to wait for staging fence");
        }
    }

    //    pub fn allocate_primitiveid

    pub fn render(&mut self) {
        self.frame_index += 1;

        let frame_indexer = (self.frame_index % MAX_FRAMES_IN_FLIGHT as i64) as usize;

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&self.context.instance, &self.context.device);

        let frame_data = FrameData::new(self, frame_indexer);

        // wait on completion of previous frame in flight
        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame_data.fence], true, u64::MAX)
                //.wait_for_fences(self.fences.as_slice(), true, u64::MAX)
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

            let depth_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .image(self.depth_buffer.as_ref().unwrap().vk_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

            self.context.device.cmd_pipeline_barrier(
                commandbuffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier, depth_barrier],
            );
        }

        // actual render work
        self.render_scene(&commandbuffer, &image_view_acquired, &frame_data);

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

    fn render_scene(
        &mut self,
        commandbuffer: &vk::CommandBuffer,
        swapchain_image_view: &vk::ImageView,
        frame_data: &FrameData,
    ) {
        let dynrend_loader =
            ash::khr::dynamic_rendering::Device::new(&self.context.instance, &self.context.device);
        let mesh_loader =
            ash::ext::mesh_shader::Device::new(&self.context.instance, &self.context.device);

        // set scissor and viewports
        let viewport = vk::Viewport::default()
            .max_depth(1.0f32)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32);
        let scissor = vk::Rect2D::default().extent(self.swapchain_extent);

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
        };

        // calculate per-object frame structure
        let mut object_ssbo: Vec<shader_struct::PrimitiveEntry> = Vec::new();

        // projection matrix
        let aspect = self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32;
        let proj_matrix = cgmath::perspective(Deg(45.0f32), 1280.0 / 800.0, 0.1f32, 200.0f32);

        //let obj_rotation = Matrix4::from_angle_y(Deg(1.0f32 * self.frame_index as f32));
        let obj_rotation = Matrix4::identity();

        for obj in &self.objects {
            for prim in obj.primitives.iter() {
                let obj_matrix = obj_rotation * obj.transform;
                let obj_pos = obj_matrix.z;
                let mvp = proj_matrix * self.view_matrix * obj_matrix;

                let obj_entry = shader_struct::PrimitiveEntry {
                    model_view_projection: mvp,
                    position: obj_pos,
                    //sphere_size: 1.0f32,
                    ibo_offset: prim.ibo_slice.offset as u32 / std::mem::size_of::<u32>() as u32,
                    index_count: prim.ibo_slice.size as u32 / std::mem::size_of::<u32>() as u32,
                    vbo_offset: prim.vbo_slice.offset as u32
                        / std::mem::size_of::<VertexEntry>() as u32,
                    albedo_handle: prim.base_color_tex,
                    metallic_roughness_handle: prim.metallic_roughness_tex,
                    occlusion_handle: prim.occlusion_tex,
                    normal_handle: prim.normal_tex,
                    emissive_handle: prim.emissive_tex,
                };
                object_ssbo.push(obj_entry);
            }
        }

        let staging_alloc_ptr = self
            .context
            .vma_alloc
            .get_allocation_info(&self.staging_buf.mem_alloc)
            .mapped_data;
        unsafe {
            std::ptr::copy_nonoverlapping(
                object_ssbo.as_ptr(),
                staging_alloc_ptr as *mut shader_struct::PrimitiveEntry,
                object_ssbo.len(),
            );
            let object_ssbo_size =
                object_ssbo.len() * std::mem::size_of::<shader_struct::PrimitiveEntry>();
            let staging_region = [ash::vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(object_ssbo_size as u64)];
            self.context.device.cmd_copy_buffer(
                *commandbuffer,
                self.staging_buf.vk_buffer,
                self.global_primitives.vk_buffer,
                &staging_region,
            );
            self.global_primitives.enqueue_barrier(
                &self.context,
                commandbuffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::MESH_SHADER_EXT,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
        }

        let object_count: u64 = match self.render_mode {
            RenderMode::DrawIndirect => object_ssbo.len().try_into().unwrap(),
            RenderMode::Draw => self.objects.len() as u64,
            RenderMode::Meshlet => {
                (self.global_meshlets.allocated_size()
                    / std::mem::size_of::<shader_struct::MeshletEntry>()) as u64
            }
        };

        // send over push constants and descriptors
        let push_constants = shader_struct::PushConstants {
            object_count: object_count,
            vbo: self.global_vbo.buffer_address,
            objects: self.global_primitives.buffer_address,
            meshlets: self.global_meshlets.buffer_address,
            drawbuf: self.indirect_draw_buf.buffer_address,
        };

        let meshlet_compute_shader = self.get_pipeline("meshlet_submit", PipelineType::COMPUTE);
        unsafe {
            let push_constant_addresses_u8: [u8; 40] = std::mem::transmute(push_constants);
            self.context.device.cmd_push_constants(
                *commandbuffer,
                meshlet_compute_shader.vk_layout,
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::MESH_EXT,
                0,
                &push_constant_addresses_u8[..],
            );
        }

        // compute shader that fills in the indirect buffer
        unsafe {
            self.context.device.cmd_fill_buffer(
                *commandbuffer,
                self.indirect_draw_buf.vk_buffer,
                0,
                1024,
                0,
            );
            self.indirect_draw_buf.enqueue_barrier(
                &self.context,
                commandbuffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );

            self.context.device.cmd_bind_pipeline(
                *commandbuffer,
                vk::PipelineBindPoint::COMPUTE,
                meshlet_compute_shader.vk_pipeline,
            );
            
            let dispatch_size = u32::div_ceil(object_count as u32, 64);
            //let dispatch_size = 1;
            self.context
                .device
                .cmd_dispatch(*commandbuffer, dispatch_size, 1, 1);

            self.indirect_draw_buf.enqueue_barrier(
                &self.context,
                commandbuffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::INDIRECT_COMMAND_READ,
            );
            // draw indirect command on the above buffer
            self.context.device.cmd_bind_pipeline(
                *commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.objects[0].pipeline.vk_pipeline,
            );

            // bind bindless descriptorset
            self.context.device.cmd_bind_descriptor_sets(
                *commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.objects[0].pipeline.vk_layout,
                0,
                &[self.bindless_texture_descriptorset],
                &[],
            );
        }

        // start dynamic rendering
        unsafe {
            let color_attachments = [vk::RenderingAttachmentInfoKHR::default()
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                })
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image_view(*swapchain_image_view)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let depth_attachment = vk::RenderingAttachmentInfoKHR::default()
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0f32,
                        stencil: 0,
                    },
                })
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image_view(self.depth_buffer.as_ref().unwrap().vk_imageview)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE);
            dynrend_loader.cmd_begin_rendering(
                *commandbuffer,
                &vk::RenderingInfoKHR::default()
                    .color_attachments(&color_attachments)
                    .depth_attachment(&depth_attachment)
                    .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
                    .layer_count(1),
            )
        }

        unsafe {
            mesh_loader.cmd_draw_mesh_tasks(*commandbuffer, object_count as u32, 1, 1);
        }

        // end dynamic rendering
        unsafe {
            dynrend_loader.cmd_end_rendering(*commandbuffer);
        }

        // let mut current_instance_id = 0;
        // for obj in &self.objects {
        //     for prim in obj.primitives.iter() {
        //         unsafe {
        //             self.context.device.cmd_bind_pipeline(
        //                 *commandbuffer,
        //                 vk::PipelineBindPoint::GRAPHICS,
        //                 obj.pipeline.vk_pipeline,
        //             );

        //             let index_count =
        //                 prim.ibo_slice.size as u32 / std::mem::size_of::<u32>() as u32;
        //             let first_index =
        //                 prim.ibo_slice.offset as u32 / std::mem::size_of::<u32>() as u32;
        //             self.context.device.cmd_draw_indexed(
        //                 *commandbuffer,
        //                 index_count,
        //                 1,
        //                 first_index,
        //                 0,
        //                 current_instance_id,
        //             );
        //             current_instance_id += 1;
        //         }
        //     }
        // }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .wait_for_fences(self.fences.as_slice(), true, u64::MAX)
                .expect("Failure to wait on end-of-engine fence");
        }

        self.global_ibo.destroy(&self.context);
        self.global_vbo.destroy(&self.context);
        self.global_meshlets.destroy(&self.context);
        self.global_primitives.destroy(&self.context);
        self.staging_buf.destroy(&self.context);
        self.indirect_draw_buf.destroy(&self.context);

        if let Some(depth_buffer) = &mut self.depth_buffer {
            depth_buffer.destroy(&self.context);
        }

        self.objects.iter_mut().for_each(|o| {
            o.textures.iter_mut().for_each(|t| t.destroy(&self.context));
        });
    }
}
