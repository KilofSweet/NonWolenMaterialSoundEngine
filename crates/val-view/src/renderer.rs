use crate::shaders;
use std::sync::Arc;
use val_core::config::AppConfig;
use val_felt::VoxelMedium;
use half::f16;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Diagnostic traces are sampled independently of the full packet count so
/// ray visualization remains useful without allocating a path for every packet.
pub const DEBUG_RAYS_PER_EMITTER: u32 = 64;
pub const DEBUG_MAX_SEGMENTS_PER_RAY: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DebugRaySegment {
    /// xyz = segment start in world metres, w = remaining energy / initial packet energy.
    start: [f32; 4],
    /// xyz = segment end in world metres, w = 1 when this segment was written.
    end: [f32; 4],
    /// Linear RGBA emitter diagnostic color.
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SolveParams {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    spacing: f32,
    src: [f32; 4],
    source_dir: [f32; 4],
    max_steps: u32,
    num_rays: u32,
    material_count: u32,
    seed: u32,
    source_energy: f32,
    air_speed_mps: f32,
    air_density_kg_m3: f32,
    air_absorption_m_inv: f32,
    time_bin_us: f32,
    time_bin_count: u32,
    energy_scale: f32,
    _padding: u32,
    debug_ray_count: u32,
    debug_segment_count: u32,
    debug_segment_offset: u32,
    _debug_padding: u32,
    debug_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DimUniform {
    dims: [u32; 4],
    spacing: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CamUniform {
    pub pos: [f32; 4],
    pub look_at: [f32; 4],
    pub up: [f32; 4],
    pub fov: f32,
    pub aspect: f32,
    pub screen: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlayUniform {
    pub t_us: f32,
    pub freq: f32,
    pub shell_thickness_m: f32,
    pub fade_us: f32,
    pub source: [f32; 4],
    /// x=air gain, y=material energy gain, z=base opacity multiplier, w=source glow gain
    pub visual: [f32; 4],
    /// xyz=emission direction, w=cosine of emission-cone half-angle
    pub emission: [f32; 4],
    /// x=energy time-bin width (us), z=inverse fixed-point scale, w=max felt alpha
    pub history: [f32; 4],
    /// xyz=energy hot color (linear RGB), w=max air-wave alpha
    pub visual2: [f32; 4],
    /// Exact integer histogram dimensions; kept out of f32 to avoid count precision loss.
    pub history_counts: [u32; 4],
}

impl PlayUniform {
    pub fn from_config(
        t_us: f32,
        frequency_hz: f32,
        source: [f32; 4],
        emitter_index: usize,
        config: &AppConfig,
    ) -> Self {
        let emitter = &config.emitters[emitter_index.min(config.emitters.len() - 1)];
        let direction = emitter.direction;
        let length = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt()
            .max(1e-6);
        let cone_cos = emitter.cone_half_angle_degrees.to_radians().cos();
        Self {
            t_us,
            freq: frequency_hz,
            shell_thickness_m: config.visualization.wave_shell_thickness_mm * 0.001,
            fade_us: config.visualization.voxel_fade_ms * 1000.0,
            source,
            visual: [
                config.visualization.air_wave_gain,
                config.visualization.felt_energy_gain,
                config.visualization.felt_base_opacity_multiplier,
                config.visualization.source_glow_gain,
            ],
            emission: [
                direction[0] / length,
                direction[1] / length,
                direction[2] / length,
                cone_cos,
            ],
            history: [
                config.simulation.energy_time_bin_us,
                0.0,
                1.0 / 100_000_000.0,
                config.visualization.felt_energy_max_opacity,
            ],
            visual2: [
                config.visualization.energy_hot_color[0],
                config.visualization.energy_hot_color[1],
                config.visualization.energy_hot_color[2],
                config.visualization.air_wave_max_opacity,
            ],
            history_counts: [config.simulation.energy_time_bins, 0, 0, 0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialBoundsGpu {
    /// xyz = inclusive minimum voxel coordinate
    min: [u32; 4],
    /// xyz = exclusive max voxel coordinate
    max: [u32; 4],
    /// x = effective density, y = mean absorption, z = mean scattering, w = mean sound speed
    properties: [f32; 4],
}

const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

struct SceneTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    render_bind: wgpu::BindGroup,
    energy_display_pipeline: wgpu::ComputePipeline,
    energy_display_bind: wgpu::BindGroup,
    _energy_display_texture: wgpu::Texture,
    energy_display_workgroups: [u32; 3],
    last_energy_time_bits: Option<u32>,
    _debug_segment_buffer: wgpu::Buffer,
    debug_segment_bind: wgpu::BindGroup,
    debug_ray_pipeline: wgpu::RenderPipeline,
    debug_vertex_count: u32,
    blit_pipeline: wgpu::RenderPipeline,
    scene_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    scene_target: SceneTarget,
    cam_buf: wgpu::Buffer,
    play_buf: wgpu::Buffer,
    resolution_scale: f32,
    max_render_width: u32,
    max_render_height: u32,
    egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub fn new(window: Arc<Window>, medium: &VoxelMedium, app_config: &AppConfig) -> Self {
        let initial_size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .unwrap();
        let adapter_limits = adapter.limits();
        // Request the adapter's real limits instead of wgpu's conservative
        // defaults. This avoids arbitrary prototype caps while retaining the
        // device's own hard texture and storage-buffer boundaries.
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("val-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
            },
            None,
        ))
        .unwrap();

        let device_limits = device.limits();
        if medium.dim.iter().any(|d| *d > device_limits.max_texture_dimension_3d) {
            panic!(
                "VAL allocation cannot fit this GPU: world texture {:?} exceeds max_texture_dimension_3d={}. Reduce world dimensions or increase voxel_size_mm.",
                medium.dim, device_limits.max_texture_dimension_3d
            );
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: initial_size.width.max(1),
            height: initial_size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let med_data = pack_medium_texture(medium)
            .unwrap_or_else(|error| panic!("VAL host allocation failed while packing material texture: {error}"));
        let med_row_pitch = align_to(medium.dim[0] * 8, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let med_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("val_medium_rgba16f"),
            size: wgpu::Extent3d {
                width: medium.dim[0],
                height: medium.dim[1],
                depth_or_array_layers: medium.dim[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("VAL GPU allocation failed for {:?} RGBA16F medium texture ({:.1} MiB): {error}", medium.dim, med_data.len() as f64 / 1_048_576.0);
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &med_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &med_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(med_row_pitch),
                rows_per_image: Some(medium.dim[1]),
            },
            wgpu::Extent3d {
                width: medium.dim[0],
                height: medium.dim[1],
                depth_or_array_layers: medium.dim[2],
            },
        );
        let med_view = med_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let med_smp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("medium_nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // One temporal energy histogram spans the complete world, including air.
        // This is essential: the visible air front is reconstructed from the same
        // packet paths that traverse material interfaces, not from a free sphere.
        let mut material_bounds_data = Vec::with_capacity(medium.material_regions.len());
        for region in &medium.material_regions {
            material_bounds_data.push(MaterialBoundsGpu {
                min: [region.min[0], region.min[1], region.min[2], 0],
                max: [region.max[0], region.max[1], region.max[2], 0],
                properties: [
                    region.effective_density_kg_m3,
                    region.sigma_a_mean,
                    region.sigma_s_mean,
                    region.sound_speed_mean,
                ],
            });
        }
        let world_cells = (medium.dim[0] as u64)
            .checked_mul(medium.dim[1] as u64)
            .and_then(|v| v.checked_mul(medium.dim[2] as u64))
            .expect("VAL world voxel count overflow");
        let energy_bins = app_config.simulation.energy_time_bins as u64;
        let energy_entries = world_cells
            .checked_mul(energy_bins)
            .expect("VAL energy history element count overflow");
        let energy_bytes = energy_entries
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .expect("VAL energy history byte count overflow");
        let max_storage_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buffer_size = device.limits().max_buffer_size;
        if energy_bytes > max_storage_binding || energy_bytes > max_buffer_size {
            panic!(
                "VAL energy-history allocation cannot fit this GPU: requested {:.1} MiB ({} bytes); max storage binding is {:.1} MiB and max buffer is {:.1} MiB. These are reported hardware/API limits, not VAL caps. Reduce world dimensions, increase voxel_size_mm, or reduce energy_time_bins.",
                energy_bytes as f64 / 1_048_576.0,
                energy_bytes,
                max_storage_binding as f64 / 1_048_576.0,
                max_buffer_size as f64 / 1_048_576.0
            );
        }
        let energy_buf = create_zeroed_storage_buffer(
            &device,
            energy_bytes,
            "world_air_and_material_energy_time_histogram",
        ).unwrap_or_else(|error| panic!("{error}"));

        // Per-frame energy reduction texture. The fragment shader reads one
        // value per voxel rather than looping over histogram bins for every
        // ray-march sample (which multiplied storage reads by pixels × steps).
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let energy_display_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("time_filtered_world_energy_r32f"),
            size: wgpu::Extent3d {
                width: medium.dim[0],
                height: medium.dim[1],
                depth_or_array_layers: medium.dim[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("VAL GPU allocation failed for time-filtered energy texture {:?}: {error}", medium.dim);
        }
        let energy_display_view = energy_display_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let energy_display_workgroups = [
            (medium.dim[0] + 3) / 4,
            (medium.dim[1] + 3) / 4,
            (medium.dim[2] + 3) / 4,
        ];

        let ray_count = app_config.simulation.ray_count;
        let ray_workgroups = ray_count / 256 + u32::from(ray_count % 256 != 0);
        if ray_workgroups > device_limits.max_compute_workgroups_per_dimension {
            panic!(
                "VAL transport dispatch cannot fit this GPU: {ray_count} packets require {ray_workgroups} workgroups, device limit is {}. Reduce simulation.ray_count.",
                device_limits.max_compute_workgroups_per_dimension
            );
        }
        let material_bounds_bytes = (medium.material_regions.len() as u64)
            .checked_mul(std::mem::size_of::<MaterialBoundsGpu>() as u64)
            .expect("material bounds buffer size overflow");
        let material_visuals_bytes = (medium.material_regions.len() as u64)
            .checked_mul(std::mem::size_of::<[f32; 4]>() as u64)
            .expect("material visuals buffer size overflow");
        for (label, bytes) in [
            ("material bounds", material_bounds_bytes),
            ("material visuals", material_visuals_bytes),
        ] {
            if bytes > device_limits.max_storage_buffer_binding_size as u64 {
                panic!(
                    "VAL {label} allocation requires {bytes} bytes, but this GPU's storage binding limit is {} bytes.",
                    device_limits.max_storage_buffer_binding_size
                );
            }
        }
        // Allocate a bounded diagnostic sample, not a path buffer for every
        // packet. The full configured ray count still contributes to the energy
        // histogram; only the first sample rays per emitter retain line geometry.
        let debug_segment_bytes = std::mem::size_of::<DebugRaySegment>() as u64;
        let debug_capacity_entries = ((device_limits.max_storage_buffer_binding_size as u64)
            .min(device_limits.max_buffer_size)
            / debug_segment_bytes)
            .min((u32::MAX / 2) as u64);
        let emitter_count_u64 = app_config.emitters.len() as u64;
        let entries_per_emitter = debug_capacity_entries / emitter_count_u64.max(1);
        let debug_segments_per_ray = (app_config.simulation.max_steps
            .min(DEBUG_MAX_SEGMENTS_PER_RAY)
            .max(1) as u64)
            .min(entries_per_emitter.max(1)) as u32;
        let debug_rays_per_emitter = if entries_per_emitter == 0 {
            0
        } else {
            (ray_count.min(DEBUG_RAYS_PER_EMITTER) as u64)
                .min((entries_per_emitter / debug_segments_per_ray as u64).max(1)) as u32
        };
        let debug_entry_count = emitter_count_u64
            .checked_mul(debug_rays_per_emitter as u64)
            .and_then(|value| value.checked_mul(debug_segments_per_ray as u64))
            .expect("VAL debug-ray buffer entry count overflow");
        let debug_buffer_entries = debug_entry_count.max(1);
        let debug_buffer_bytes = debug_buffer_entries
            .checked_mul(debug_segment_bytes)
            .expect("VAL debug-ray buffer byte size overflow");
        if debug_buffer_bytes > device_limits.max_storage_buffer_binding_size as u64
            || debug_buffer_bytes > device_limits.max_buffer_size
        {
            panic!("VAL debug-ray buffer exceeds the selected GPU's storage-buffer limits");
        }
        let debug_segment_buffer = create_zeroed_storage_buffer(
            &device,
            debug_buffer_bytes,
            "sampled_transport_debug_ray_segments",
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let debug_vertex_count = (debug_entry_count.min((u32::MAX / 2) as u64) as u32) * 2;
        if debug_rays_per_emitter < ray_count.min(DEBUG_RAYS_PER_EMITTER) {
            eprintln!(
                "VAL debug-ray sample reduced to {} rays/emitter × {} segments by GPU storage limits",
                debug_rays_per_emitter, debug_segments_per_ray
            );
        }

        let step_m = app_config.simulation.step_mm * 0.001;
        let total_amplitude = app_config.emitters.iter()
            .map(|emitter| emitter.amplitude as f64)
            .sum::<f64>();
        // Keep the aggregate fixed-point histogram in a stable visual range as
        // emitters are added. Relative amplitude ratios remain intact.
        let energy_scale = (100_000_000.0_f64 / total_amplitude.max(1.0)) as f32;
        let solve_param_buffers: Vec<wgpu::Buffer> = app_config
            .emitters
            .iter()
            .enumerate()
            .map(|(index, emitter)| {
                let direction = emitter.direction;
                let direction_len = (direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2])
                    .sqrt()
                    .max(1e-6);
                let params = SolveParams {
                    dim_x: medium.dim[0],
                    dim_y: medium.dim[1],
                    dim_z: medium.dim[2],
                    spacing: medium.spacing,
                    src: [
                        emitter.position_mm[0] * 0.001,
                        emitter.position_mm[1] * 0.001,
                        emitter.position_mm[2] * 0.001,
                        step_m,
                    ],
                    source_dir: [
                        direction[0] / direction_len,
                        direction[1] / direction_len,
                        direction[2] / direction_len,
                        emitter.cone_half_angle_degrees.to_radians().cos(),
                    ],
                    max_steps: app_config.simulation.max_steps,
                    num_rays: ray_count,
                    material_count: medium.material_regions.len() as u32,
                    seed: ((app_config.world.seed as u32)
                        ^ ((app_config.world.seed >> 32) as u32)
                        ^ (index as u32).wrapping_mul(0x9E37_79B9)),
                    source_energy: emitter.amplitude,
                    air_speed_mps: app_config.world.air_speed_mps,
                    air_density_kg_m3: app_config.world.air_density_kg_m3,
                    air_absorption_m_inv: app_config.world.air_absorption_m_inv,
                    time_bin_us: app_config.simulation.energy_time_bin_us,
                    time_bin_count: app_config.simulation.energy_time_bins,
                    energy_scale,
                    _padding: 0,
                    debug_ray_count: debug_rays_per_emitter,
                    debug_segment_count: debug_segments_per_ray,
                    debug_segment_offset: (index as u32)
                        .wrapping_mul(debug_rays_per_emitter)
                        .wrapping_mul(debug_segments_per_ray),
                    _debug_padding: 0,
                    debug_color: debug_emitter_color(index),
                };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("val_transport_solve_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform"),
            size: std::mem::size_of::<CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let play_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("playback_visual_uniform"),
            size: std::mem::size_of::<PlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dim_uniform = DimUniform {
            dims: [medium.dim[0], medium.dim[1], medium.dim[2], 0],
            spacing: [
                medium.spacing,
                app_config.world.air_speed_mps,
                app_config.world.air_density_kg_m3,
                app_config.world.air_absorption_m_inv,
            ],
        };
        let dim_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("world_dimensions"),
            contents: bytemuck::bytes_of(&dim_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let material_visuals: Vec<[f32; 4]> = medium
            .material_regions
            .iter()
            .map(|region| region.visual)
            .collect();
        let material_bounds = material_bounds_data;
        let material_visuals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_visual_settings"),
            contents: bytemuck::cast_slice(&material_visuals),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_bounds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_region_bounds"),
            contents: bytemuck::cast_slice(&material_bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let solve_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packet_transport_compute"),
            source: wgpu::ShaderSource::Wgsl(shaders::SOLVE.into()),
        });
        let render_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_felt_renderer"),
            source: wgpu::ShaderSource::Wgsl(shaders::RENDER.into()),
        });
        let energy_display_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("time_filtered_energy_field"),
            source: wgpu::ShaderSource::Wgsl(shaders::ENERGY_DISPLAY.into()),
        });

        let solve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transport_compute_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, false),
                uniform_layout_entry(3, wgpu::ShaderStages::COMPUTE),
                storage_layout_entry(4, wgpu::ShaderStages::COMPUTE, true),
                storage_layout_entry(5, wgpu::ShaderStages::COMPUTE, false),
            ],
        });
        let solve_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transport_compute_pipeline_layout"),
            bind_group_layouts: &[&solve_bgl],
            push_constant_ranges: &[],
        });
        let solve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("one_shot_packet_transport"),
            layout: Some(&solve_pipeline_layout),
            module: &solve_mod,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let solve_binds: Vec<wgpu::BindGroup> = solve_param_buffers
            .iter()
            .map(|params| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("val_transport_compute_bind_group"),
                    layout: &solve_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&med_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&med_smp) },
                        wgpu::BindGroupEntry { binding: 2, resource: energy_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: material_bounds_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: debug_segment_buffer.as_entire_binding() },
                    ],
                })
            })
            .collect();

        let energy_display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("time_filtered_energy_layout"),
            entries: &[
                storage_layout_entry(0, wgpu::ShaderStages::COMPUTE, true),
                uniform_layout_entry(1, wgpu::ShaderStages::COMPUTE),
                uniform_layout_entry(2, wgpu::ShaderStages::COMPUTE),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let energy_display_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("time_filtered_energy_pipeline_layout"),
            bind_group_layouts: &[&energy_display_bgl],
            push_constant_ranges: &[],
        });
        let energy_display_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("time_filtered_energy_update"),
            layout: Some(&energy_display_layout),
            module: &energy_display_mod,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let energy_display_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("time_filtered_energy_bind_group"),
            layout: &energy_display_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: energy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dim_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: play_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&energy_display_view) },
            ],
        });

        // The packet transport is precomputed once. Playback does not dispatch
        // compute work; it only changes the time-dependent visualization.
        let solve_start = std::time::Instant::now();
        let mut solve_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("initial_transport_solve_encoder"),
        });
        for (emitter_index, solve_bind) in solve_binds.iter().enumerate() {
            let mut pass = solve_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emitter_transport_solve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&solve_pipeline);
            pass.set_bind_group(0, solve_bind, &[]);
            pass.dispatch_workgroups(ray_workgroups, 1, 1);
            eprintln!("VAL emitter {} transport dispatch queued: {} packets", emitter_index + 1, ray_count);
        }
        queue.submit(Some(solve_encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        eprintln!(
            "VAL transport solve complete: {} emitters × {ray_count} packets, {} total packets × {} max segments across air/materials ({:.2?}); world histogram {:.1} MiB",
            app_config.emitters.len(),
            app_config.emitters.len() as u64 * ray_count as u64,
            app_config.simulation.max_steps,
            solve_start.elapsed(),
            energy_bytes as f64 / (1024.0 * 1024.0)
        );

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume_render_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform_layout_entry(1, wgpu::ShaderStages::FRAGMENT),
                uniform_layout_entry(2, wgpu::ShaderStages::FRAGMENT),
                uniform_layout_entry(3, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_layout_entry(5, wgpu::ShaderStages::FRAGMENT, true),
                storage_layout_entry(6, wgpu::ShaderStages::FRAGMENT, true),
            ],
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume_render_pipeline_layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume_raymarch_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_mod,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &render_mod,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        let render_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume_render_bind_group"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&energy_display_view) },
                wgpu::BindGroupEntry { binding: 1, resource: cam_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dim_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: play_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&med_view) },
                wgpu::BindGroupEntry { binding: 5, resource: material_visuals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: material_bounds_buf.as_entire_binding() },
            ],
        });

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_blit_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_upscale_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let blit_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::BLIT.into()),
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_blit_pipeline_layout"),
            bind_group_layouts: &[&scene_bgl],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_mod,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_mod,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let debug_ray_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transport_debug_ray_lines"),
            source: wgpu::ShaderSource::Wgsl(shaders::DEBUG_RAYS.into()),
        });
        let debug_ray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transport_debug_ray_render_layout"),
            entries: &[
                storage_layout_entry(0, wgpu::ShaderStages::VERTEX, true),
                uniform_layout_entry(1, wgpu::ShaderStages::VERTEX),
            ],
        });
        let debug_ray_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transport_debug_ray_render_bind"),
            layout: &debug_ray_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: debug_segment_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cam_buf.as_entire_binding() },
            ],
        });
        let debug_ray_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transport_debug_ray_pipeline_layout"),
            bind_group_layouts: &[&debug_ray_bgl],
            push_constant_ranges: &[],
        });
        let debug_ray_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transport_debug_ray_line_pipeline"),
            layout: Some(&debug_ray_layout),
            vertex: wgpu::VertexState {
                module: &debug_ray_mod,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &debug_ray_mod,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let scene_target = Self::create_scene_target(
            &device,
            &scene_bgl,
            &blit_sampler,
            config.width,
            config.height,
            app_config.render.resolution_scale,
            app_config.render.max_width,
            app_config.render.max_height,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            render_bind,
            energy_display_pipeline,
            energy_display_bind,
            _energy_display_texture: energy_display_texture,
            energy_display_workgroups,
            last_energy_time_bits: None,
            _debug_segment_buffer: debug_segment_buffer,
            debug_segment_bind: debug_ray_bind,
            debug_ray_pipeline,
            debug_vertex_count,
            blit_pipeline,
            scene_bgl,
            blit_sampler,
            scene_target,
            cam_buf,
            play_buf,
            resolution_scale: app_config.render.resolution_scale,
            max_render_width: app_config.render.max_width,
            max_render_height: app_config.render.max_height,
            egui_renderer,
        }
    }

    fn create_scene_target(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        surface_width: u32,
        surface_height: u32,
        scale: f32,
        max_width: u32,
        max_height: u32,
    ) -> SceneTarget {
        let mut width = ((surface_width as f32 * scale).round() as u32).max(1);
        let mut height = ((surface_height as f32 * scale).round() as u32).max(1);
        let fit = (max_width as f32 / width as f32)
            .min(max_height as f32 / height as f32)
            .min(1.0);
        width = ((width as f32 * fit).round() as u32).max(1);
        height = ((height as f32 * fit).round() as u32).max(1);
        let limit = device.limits().max_texture_dimension_2d;
        width = width.min(limit);
        height = height.min(limit);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reduced_resolution_volume_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduced_resolution_volume_target_bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        SceneTarget { _texture: texture, view, bind_group, width, height }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.scene_target = Self::create_scene_target(
            &self.device,
            &self.scene_bgl,
            &self.blit_sampler,
            width,
            height,
            self.resolution_scale,
            self.max_render_width,
            self.max_render_height,
        );
    }

    pub fn render(
        &mut self,
        cam: &CamUniform,
        play: &PlayUniform,
        ui_primitives: &[egui::ClippedPrimitive],
        ui_textures: &egui::TexturesDelta,
        pixels_per_point: f32,
        show_debug_rays: bool,
    ) {
        let mut internal_cam = *cam;
        internal_cam.screen = [self.scene_target.width as f32, self.scene_target.height as f32];
        internal_cam.aspect = self.scene_target.width as f32 / self.scene_target.height.max(1) as f32;
        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&internal_cam));
        self.queue.write_buffer(&self.play_buf, 0, bytemuck::bytes_of(play));

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::Timeout | wgpu::SurfaceError::OutOfMemory) => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });
        // Camera-only redraws while paused reuse the last filtered 3D field.
        // The full-world compute reduction runs only when timeline time changes.
        let update_energy_field = self.last_energy_time_bits != Some(play.t_us.to_bits());
        if update_energy_field {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("time_filtered_energy_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.energy_display_pipeline);
            pass.set_bind_group(0, &self.energy_display_bind, &[]);
            pass.dispatch_workgroups(
                self.energy_display_workgroups[0],
                self.energy_display_workgroups[1],
                self.energy_display_workgroups[2],
            );
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("felt_volume_raymarch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        if show_debug_rays && self.debug_vertex_count > 0 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transport_debug_ray_overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.debug_ray_pipeline);
            pass.set_bind_group(0, &self.debug_segment_bind, &[]);
            pass.draw(0..self.debug_vertex_count, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_upscale_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.scene_target.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        for (texture_id, image_delta) in &ui_textures.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *texture_id, image_delta);
        }
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point,
        };
        let callback_commands = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            ui_primitives,
            &screen_descriptor,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("val_egui_overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer.render(
                &mut pass,
                ui_primitives,
                &screen_descriptor,
            );
        }
        self.queue.submit(callback_commands.into_iter().chain(std::iter::once(encoder.finish())));
        for texture_id in &ui_textures.free {
            self.egui_renderer.free_texture(texture_id);
        }
        if update_energy_field {
            self.last_energy_time_bits = Some(play.t_us.to_bits());
        }
        frame.present();
    }
}

fn debug_emitter_color(index: usize) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 8] = [
        [0.20, 0.86, 1.00, 0.92],
        [1.00, 0.48, 0.18, 0.92],
        [0.72, 0.52, 1.00, 0.92],
        [0.42, 0.94, 0.48, 0.92],
        [1.00, 0.78, 0.24, 0.92],
        [1.00, 0.42, 0.70, 0.92],
        [0.34, 0.72, 1.00, 0.92],
        [0.78, 0.90, 0.90, 0.92],
    ];
    PALETTE[index % PALETTE.len()]
}

fn create_zeroed_storage_buffer(
    device: &wgpu::Device,
    size: u64,
    label: &str,
) -> Result<wgpu::Buffer, String> {
    if size == 0 {
        return Err(format!("VAL requested a zero-byte storage buffer for '{label}'"));
    }
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    device.poll(wgpu::Maintain::Wait);
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!(
            "VAL GPU allocation failed for '{label}' ({:.1} MiB): {error}",
            size as f64 / 1_048_576.0
        ));
    }
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.fill(0);
    }
    buffer.unmap();
    Ok(buffer)
}

fn storage_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}

fn pack_medium_texture(medium: &VoxelMedium) -> Result<Vec<u8>, String> {
    let width = medium.dim[0] as usize;
    let height = medium.dim[1] as usize;
    let depth = medium.dim[2] as usize;
    let row_bytes = width.checked_mul(8).ok_or_else(|| "medium row byte count overflow".to_string())?;
    let padded_row_bytes = align_to(row_bytes as u32, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
    let byte_len = padded_row_bytes
        .checked_mul(height)
        .and_then(|v| v.checked_mul(depth))
        .ok_or_else(|| "medium texture upload size overflow".to_string())?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len)
        .map_err(|error| format!("unable to reserve {:.1} MiB for medium upload: {error}", byte_len as f64 / 1_048_576.0))?;
    bytes.resize(byte_len, 0u8);

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = z * height * width + y * width + x;
                let values = [
                    f16::from_f32(medium.sigma_a[index]).to_bits(),
                    f16::from_f32(medium.sigma_s[index]).to_bits(),
                    f16::from_f32(medium.g[index]).to_bits(),
                    f16::from_f32(medium.c[index]).to_bits(),
                ];
                let source = bytemuck::cast_slice(&values);
                let destination = (z * height + y) * padded_row_bytes + x * 8;
                bytes[destination..destination + 8].copy_from_slice(source);
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod uniform_layout_tests {
    use super::PlayUniform;

    #[test]
    fn playback_uniform_matches_sixteen_byte_wgsl_lanes() {
        assert_eq!(std::mem::size_of::<PlayUniform>(), 112);
    }
}


#[cfg(test)]
mod debug_layout_tests {
    use super::{DebugRaySegment, SolveParams};

    #[test]
    fn debug_ray_gpu_struct_layouts_match_wgsl_vec4_strides() {
        assert_eq!(std::mem::size_of::<DebugRaySegment>(), 48);
        assert_eq!(std::mem::size_of::<SolveParams>(), 128);
    }
}
