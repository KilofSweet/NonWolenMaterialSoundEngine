mod camera;
mod renderer;
mod shaders;
mod state;
mod ui;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::WindowBuilder;

use val_core::config::{AppConfig, MaterialConfig, RenderConfig};
use val_felt::{ClumpNoise, FeltVolume, GradientDensity, MediumLaw, MediumProperties};

use camera::OrbitCamera;
use renderer::{CamUniform, PlayUniform, Renderer};
use state::PlaybackState;
use ui::{UiAction, UiState};

fn material_seed(world_seed: u64, material: &MaterialConfig, index: usize) -> u64 {
    material.seed.unwrap_or_else(|| {
        world_seed
            .wrapping_add((index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    })
}

fn build_medium(config: &AppConfig, world_dims: [u32; 3]) -> Result<val_felt::VoxelMedium, String> {
    let voxel_mm = config.world.voxel_size_mm;
    let spacing_m = voxel_mm * 0.001;
    let mut volumes = Vec::new();
    volumes
        .try_reserve_exact(config.materials.len())
        .map_err(|error| format!("unable to reserve material instances: {error}"))?;

    for (index, material) in config.materials.iter().enumerate() {
        let min: [u32; 3] = std::array::from_fn(|axis| {
            (material.bounds_min_mm[axis] / voxel_mm).round() as u32
        });
        let max: [u32; 3] = std::array::from_fn(|axis| {
            (material.bounds_max_mm[axis] / voxel_mm).round() as u32
        });
        let local_dims = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

        let mut laws: Vec<Box<dyn MediumLaw>> = Vec::new();
        if material.gradient_strength.abs() > f32::EPSILON {
            laws.push(Box::new(GradientDensity {
                axis: material.gradient_axis,
                strength: material.gradient_strength,
                center: local_dims[material.gradient_axis.min(2)] as f32 * spacing_m * 0.5,
            }));
        }
        if material.clump_strength.abs() > f32::EPSILON {
            laws.push(Box::new(ClumpNoise {
                scale: material.clump_scale_per_m,
                strength: material.clump_strength,
            }));
        }

        let properties = MediumProperties {
            sigma_a: material.sigma_a_m_inv,
            sigma_s: material.sigma_s_m_inv,
            anisotropy_g: material.anisotropy_g,
            sound_speed_mps: material.sound_speed_mps,
        };
        let local_medium = val_felt::generate_medium_with_properties(
            local_dims,
            spacing_m,
            material_seed(config.world.seed, material, index),
            properties,
            &laws,
        )
        .map_err(|error| format!("material '{}' allocation failed: {error}", material.name))?;

        volumes.push(FeltVolume {
            name: material.name.clone(),
            origin: min,
            medium: local_medium,
            visual: [
                material.visual_color[0],
                material.visual_color[1],
                material.visual_color[2],
                material.visual_opacity,
            ],
            density_kg_m3: material.density_kg_m3,
            effective_density_kg_m3: material.effective_density_kg_m3,
            porosity: material.porosity,
        });
    }

    val_felt::compose_world(world_dims, spacing_m, config.world.air_speed_mps, volumes)
}

#[cfg(target_os = "linux")]
fn available_host_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn available_host_memory_bytes() -> Option<u64> {
    None
}

fn report_startup_allocation(config: &AppConfig, world_dims: [u32; 3]) -> Result<(), String> {
    let world_cells = (world_dims[0] as u64)
        .checked_mul(world_dims[1] as u64)
        .and_then(|v| v.checked_mul(world_dims[2] as u64))
        .ok_or_else(|| "world voxel count overflow; use smaller dimensions or larger voxels".to_string())?;
    let history_bytes = world_cells
        .checked_mul(config.simulation.energy_time_bins as u64)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| "energy-history size overflow; reduce dimensions/bins or increase voxel size".to_string())?;
    // Approximate peak host construction: four world f32 arrays, occupancy, and
    // local material arrays. This is a feasibility estimate, never a fixed cap.
    let estimated_host_bytes = world_cells
        .checked_mul(33)
        .ok_or_else(|| "estimated host allocation size overflows u64".to_string())?;

    if let Some(available) = available_host_memory_bytes() {
        if estimated_host_bytes > available {
            return Err(format!(
                "host-memory allocation rejected by the operating system's current memory report: estimated peak {:.1} MiB, MemAvailable {:.1} MiB. This is not a VAL size limit. Reduce voxel resolution/material volume or close other memory-heavy applications.",
                estimated_host_bytes as f64 / 1_048_576.0,
                available as f64 / 1_048_576.0
            ));
        }
        println!(
            "allocation request: {:.1} MiB GPU energy history; estimated peak {:.1} MiB host construction memory; OS MemAvailable {:.1} MiB",
            history_bytes as f64 / 1_048_576.0,
            estimated_host_bytes as f64 / 1_048_576.0,
            available as f64 / 1_048_576.0
        );
    } else {
        println!(
            "allocation request: {:.1} MiB GPU energy history; estimated peak {:.1} MiB host construction memory; OS available-memory query unavailable, fallible allocations enabled",
            history_bytes as f64 / 1_048_576.0,
            estimated_host_bytes as f64 / 1_048_576.0
        );
    }
    Ok(())
}

fn make_camera_uniform(
    camera: &OrbitCamera,
    screen: (f32, f32),
    render: &RenderConfig,
) -> CamUniform {
    let surface_width = screen.0.max(1.0);
    let surface_height = screen.1.max(1.0);
    let mut target_width = (surface_width * render.resolution_scale).round().max(1.0);
    let mut target_height = (surface_height * render.resolution_scale).round().max(1.0);
    let fit = (render.max_width.max(1) as f32 / target_width)
        .min(render.max_height.max(1) as f32 / target_height)
        .min(1.0);
    target_width = (target_width * fit).round().max(1.0);
    target_height = (target_height * fit).round().max(1.0);

    let pos = camera.position();
    CamUniform {
        pos: [pos.x, pos.y, pos.z, 0.0],
        look_at: [camera.target.x, camera.target.y, camera.target.z, 0.0],
        up: [0.0, 1.0, 0.0, 0.0],
        fov: 0.9,
        // Match the reduced-resolution target's projection aspect. The target
        // is then upscaled to the window, so the editor's 3D overlays need the
        // same aspect to sit directly on top of the volume render.
        aspect: target_width / target_height,
        screen: [screen.0, screen.1],
    }
}

/// Test a winit physical-pixel cursor position against egui's logical-point
/// viewport rectangle. Egui's SidePanel occupies part of the window, so the
/// whole window must not be treated as the camera viewport.
fn cursor_over_scene_viewport(
    cursor_pos: (f64, f64),
    pixels_per_point: f64,
    viewport: Option<egui::Rect>,
) -> bool {
    let Some(viewport) = viewport else { return false };
    let scale = pixels_per_point.max(f64::EPSILON);
    let point = egui::pos2(
        (cursor_pos.0 / scale) as f32,
        (cursor_pos.1 / scale) as f32,
    );
    viewport.contains(point)
}

fn main() {
    let mut config = AppConfig::load_or_default("val.toml");
    let world_dims = config.validate().unwrap_or_else(|error| {
        eprintln!("VAL configuration error: {error}");
        std::process::exit(2);
    });
    report_startup_allocation(&config, world_dims).unwrap_or_else(|error| {
        eprintln!("VAL allocation preflight failed: {error}");
        std::process::exit(2);
    });
    let medium = build_medium(&config, world_dims).unwrap_or_else(|error| {
        eprintln!("VAL material setup error: {error}");
        std::process::exit(2);
    });

    println!("VAL: {} felt regions + explicit air medium", medium.material_regions.len());
    println!("world: {:?} voxels at {:.3} mm spacing", world_dims, config.world.voxel_size_mm);
    for (index, material) in medium.material_regions.iter().enumerate() {
        let settings = &config.materials[index];
        println!(
            "material {}: {:?}..{:?} voxels ({}), sigma_a={:.2} m^-1, sigma_s={:.2} m^-1",
            material.id, material.min, material.max, material.name,
            settings.sigma_a_m_inv, settings.sigma_s_m_inv,
        );
    }
    println!(
        "{} emitters × {} packets; max {} spatial segments/packet; render {} FPS cap",
        config.emitters.len(), config.simulation.ray_count, config.simulation.max_steps, config.render.target_fps
    );
    println!(
        "time history: {} bins × {:.1} us = {:.0} us",
        config.simulation.energy_time_bins,
        config.simulation.energy_time_bin_us,
        config.simulation.energy_time_bins as f32 * config.simulation.energy_time_bin_us,
    );
    println!("controls: drag=orbit · wheel=zoom · WASD/QE=move · space=pause · [ ]=speed · , .=scrub · R=reset");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("VAL — Virtual Acoustic Laboratory")
            .with_inner_size(winit::dpi::LogicalSize::new(1180u32, 900u32))
            .build(&event_loop)
            .unwrap(),
    );

    let mut renderer = Some(Renderer::new(window.clone(), &medium, &config));
    // The GPU now owns the uploaded medium; release the temporary host-side voxel arrays.
    drop(medium);
    let mut camera = OrbitCamera::from_config(&config.camera);
    let mut playback = PlaybackState::new(&config.emitters[0]);
    let mut draft = config.clone();
    let mut ui_state = UiState::default();
    let mut status = String::from("Ready · edit parameters, then Apply & rebuild");
    let mut egui_context = egui::Context::default();
    let mut egui_state = egui_winit::State::new(
        egui_context.clone(),
        egui::ViewportId::ROOT,
        window.as_ref(),
        Some(window.scale_factor() as f32),
        None,
    );

    let size = window.inner_size();
    let mut screen = (size.width as f32, size.height as f32);
    let mut cursor_pos = (0.0f64, 0.0f64);
    let mut cursor_known = false;
    let mut orbit_dragging = false;
    let mut held: HashSet<String> = HashSet::new();
    let mut frame_interval = Duration::from_secs_f64(1.0 / config.render.target_fps as f64);
    let mut last_frame = Instant::now();
    let mut next_frame = last_frame;
    let mut last_title = String::new();
    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            let mut egui_consumed = false;
            let mut egui_repaint = false;
            if let Event::WindowEvent { window_id, event } = &event {
                if *window_id == window.id() {
                    let response = egui_state.on_window_event(&window, event);
                    egui_consumed = response.consumed;
                    egui_repaint = response.repaint;
                }
            }
            if egui_repaint {
                window.request_redraw();
            }

            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    screen = (size.width as f32, size.height as f32);
                    renderer.as_mut().expect("renderer is initialized").resize(size.width, size.height);
                    window.request_redraw();
                }
                Event::WindowEvent { event: WindowEvent::Focused(false), .. } => {
                    held.clear();
                    orbit_dragging = false;
                    cursor_known = false;
                    if config.render.pause_when_unfocused && !playback.paused {
                        playback.toggle_pause();
                        window.request_redraw();
                    }
                }
                Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                    if orbit_dragging && cursor_known {
                        // Once a drag starts in the scene viewport, keep capture
                        // until release even if the cursor crosses over the UI.
                        let scale = window.scale_factor().max(f64::EPSILON);
                        camera.orbit(
                            (position.x - cursor_pos.0) / scale,
                            (position.y - cursor_pos.1) / scale,
                        );
                        window.request_redraw();
                    }
                    cursor_pos = (position.x, position.y);
                    cursor_known = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::MouseInput { state, button: MouseButton::Left, .. },
                    ..
                } => {
                    match state {
                        ElementState::Pressed => {
                            // Route by the actual scene viewport, not egui's
                            // generic consumed flag. The CentralPanel also
                            // spans the scene and may report pointer interest.
                            orbit_dragging = cursor_known && cursor_over_scene_viewport(
                                cursor_pos,
                                window.scale_factor(),
                                ui_state.viewport_rect,
                            );
                        }
                        ElementState::Released => orbit_dragging = false,
                    }
                }
                Event::WindowEvent { event: WindowEvent::CursorLeft { .. }, .. } => {
                    orbit_dragging = false;
                    cursor_known = false;
                }
                Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => {
                    if cursor_known && cursor_over_scene_viewport(
                        cursor_pos,
                        window.scale_factor(),
                        ui_state.viewport_rect,
                    ) {
                        let amount = match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => y * 0.01,
                            winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.001,
                        };
                        camera.zoom(amount);
                        window.request_redraw();
                    }
                }
                Event::WindowEvent { event: WindowEvent::KeyboardInput { event: ke, .. }, .. } => {
                    let pressed = ke.state == ElementState::Pressed;
                    let name = match &ke.logical_key {
                        Key::Character(c) => c.to_string().to_lowercase(),
                        Key::Named(NamedKey::Space) => "space".into(),
                        Key::Named(NamedKey::ArrowLeft) => ",".into(),
                        Key::Named(NamedKey::ArrowRight) => ".".into(),
                        _ => String::new(),
                    };
                    match name.as_str() {
                        "w" | "a" | "s" | "d" | "q" | "e" => {
                            if pressed {
                                // Do not move the camera while the user is
                                // typing into an editor field. Always process
                                // releases, however, to prevent stuck keys.
                                if !egui_consumed {
                                    held.insert(name.clone());
                                }
                            } else {
                                held.remove(&name);
                            }
                            window.request_redraw();
                        }
                        _ if !egui_consumed => match name.as_str() {
                            "space" if pressed && !ke.repeat => playback.toggle_pause(),
                            "r" if pressed && !ke.repeat => playback.reset(),
                            "[" if pressed && !ke.repeat => playback.adjust_speed(1.0 / 1.5),
                            "]" if pressed && !ke.repeat => playback.adjust_speed(1.5),
                            "," if pressed && !ke.repeat => playback.scrub(-config.simulation.energy_time_bin_us, &config.simulation),
                            "." if pressed && !ke.repeat => playback.scrub(config.simulation.energy_time_bin_us, &config.simulation),
                            "-" if pressed && !ke.repeat => playback.adjust_frequency(1.0 / 1.25),
                            "=" if pressed && !ke.repeat => playback.adjust_frequency(1.25),
                            _ => {}
                        },
                        _ => {}
                    }
                    if pressed && !ke.repeat && !egui_consumed {
                        window.request_redraw();
                    }
                }
                Event::AboutToWait => {
                    let animate = !playback.paused || !held.is_empty();
                    if animate {
                        let now = Instant::now();
                        if now >= next_frame {
                            window.request_redraw();
                            next_frame = now + frame_interval;
                        }
                        elwt.set_control_flow(ControlFlow::WaitUntil(next_frame));
                    } else {
                        // A paused scene sleeps; egui pointer/key events request redraws.
                        elwt.set_control_flow(ControlFlow::Wait);
                    }
                }
                Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32().min(frame_interval.as_secs_f32());
                    last_frame = now;
                    playback.advance(dt, &config.simulation);

                    let camera_speed = config.camera.movement_speed_mps * dt;
                    let forward = (held.contains("w") as i8 - held.contains("s") as i8) as f32;
                    let right = (held.contains("d") as i8 - held.contains("a") as i8) as f32;
                    let up = (held.contains("e") as i8 - held.contains("q") as i8) as f32;
                    if forward != 0.0 || right != 0.0 || up != 0.0 {
                        camera.move_local(forward * camera_speed, right * camera_speed, up * camera_speed);
                    }
                    let preview_camera = make_camera_uniform(&camera, screen, &config.render);

                    let raw_input = egui_state.take_egui_input(&window);
                    let mut ui_action = UiAction::None;
                    let mut egui_context_reset = false;
                    let full_output = egui_context.run(raw_input, |ctx| {
                        ui_action = ui::draw(
                            ctx,
                            &mut draft,
                            &mut playback,
                            &mut ui_state,
                            &status,
                            &preview_camera,
                        );
                    });
                    egui_state.handle_platform_output(&window, full_output.platform_output);

                    match ui_action {
                        UiAction::None => {}
                        UiAction::Save => match draft.validate() {
                            Ok(_) => match draft.save_to("val.toml") {
                                Ok(()) => {
                                    status = "Saved draft to val.toml".into();
                                    window.request_redraw();
                                }
                                Err(error) => {
                                    status = format!("Save failed: {error}");
                                    window.request_redraw();
                                }
                            },
                            Err(error) => {
                                status = format!("Cannot save invalid configuration: {error}");
                                window.request_redraw();
                            }
                        },
                        UiAction::Apply => match draft.validate() {
                            Err(error) => {
                                status = format!("Apply rejected: {error}");
                                window.request_redraw();
                            }
                            Ok(new_dims) => match build_medium(&draft, new_dims) {
                                Err(error) => {
                                    status = format!("Material rebuild failed: {error}");
                                    window.request_redraw();
                                }
                                Ok(new_medium) => {
                                    // Avoid holding two complete GPU worlds during a rebuild.
                                    drop(renderer.take());
                                    renderer = Some(Renderer::new(window.clone(), &new_medium, &draft));
                                    drop(new_medium);
                                    // The new egui-wgpu renderer has an empty managed texture store.
                                    // Recreate the Context so its font atlas is uploaded again, while
                                    // carrying over Memory to preserve panel state, selection focus,
                                    // scroll offsets, and other widget interaction state.
                                    let preserved_memory = egui_context.memory(|memory| memory.clone());
                                    egui_context = egui::Context::default();
                                    egui_context.memory_mut(|memory| {
                                        *memory = preserved_memory;
                                        // An editor TextEdit may have held focus
                                        // when Apply was clicked. Do not carry
                                        // that keyboard capture into the new UI
                                        // integration or it will swallow WASD.
                                        if let Some(focused) = memory.focused() {
                                            memory.surrender_focus(focused);
                                        }
                                        memory.stop_text_input();
                                    });
                                    egui_state = egui_winit::State::new(
                                        egui_context.clone(),
                                        egui::ViewportId::ROOT,
                                        window.as_ref(),
                                        Some(window.scale_factor() as f32),
                                        None,
                                    );
                                    egui_context_reset = true;
                                    config = draft.clone();
                                    held.clear();
                                    orbit_dragging = false;
                                    camera = OrbitCamera::from_config(&config.camera);
                                    let selected = ui_state.selected_emitter.min(config.emitters.len() - 1);
                                    ui_state.selected_emitter = selected;
                                    playback.time_us = 0.0;
                                    playback.frequency_hz = config.emitters[selected].frequency_hz;
                                    playback.paused = false;
                                    frame_interval = Duration::from_secs_f64(1.0 / config.render.target_fps as f64);
                                    last_frame = Instant::now();
                                    next_frame = last_frame;
                                    status = format!(
                                        "Applied · {} material region(s), {} emitter(s); transport history rebuilt",
                                        config.materials.len(), config.emitters.len()
                                    );
                                    draft = config.clone();
                                    window.request_redraw();
                                }
                            }
                        }
                    }

                    // Apply may reset the camera from the newly applied config; compute
                    // the final camera uniform after processing the editor action.
                    let cam = make_camera_uniform(&camera, screen, &config.render);
                    let selected = ui_state.selected_emitter.min(config.emitters.len() - 1);
                    let emitter = &config.emitters[selected];
                    let source = [
                        emitter.position_mm[0] * 0.001,
                        emitter.position_mm[1] * 0.001,
                        emitter.position_mm[2] * 0.001,
                        emitter.radius_mm * 0.001,
                    ];
                    let play = PlayUniform::from_config(
                        playback.time_us,
                        playback.frequency_hz,
                        source,
                        selected,
                        &config,
                    );
                    // The Apply frame's egui output belongs to the old Context/texture
                    // store. Skip its overlay once; the requested next redraw runs the
                    // new Context, which uploads its font atlas into the rebuilt renderer.
                    let paint_jobs = if egui_context_reset {
                        Vec::new()
                    } else {
                        egui_context.tessellate(full_output.shapes, full_output.pixels_per_point)
                    };
                    let empty_textures_delta = egui::TexturesDelta::default();
                    let textures_delta = if egui_context_reset {
                        &empty_textures_delta
                    } else {
                        &full_output.textures_delta
                    };
                    renderer.as_mut().expect("renderer is initialized").render(
                        &cam,
                        &play,
                        &paint_jobs,
                        textures_delta,
                        full_output.pixels_per_point,
                        ui_state.debug_show_rays,
                    );

                    let title = format!(
                        "VAL — Virtual Acoustic Laboratory | t={:7.1} µs | {:6.1} kHz | speed {:5.2}× | {}",
                        playback.time_us,
                        playback.frequency_hz / 1000.0,
                        playback.speed,
                        if playback.paused { "PAUSED" } else { "play" }
                    );
                    if title != last_title {
                        window.set_title(&title);
                        last_title = title;
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}

#[cfg(test)]
mod input_tests {
    use super::cursor_over_scene_viewport;

    #[test]
    fn camera_input_is_routed_only_to_scene_viewport() {
        let viewport = egui::Rect::from_min_max(
            egui::pos2(390.0, 0.0),
            egui::pos2(1180.0, 900.0),
        );

        assert!(cursor_over_scene_viewport((800.0, 300.0), 1.0, Some(viewport)));
        assert!(!cursor_over_scene_viewport((200.0, 300.0), 1.0, Some(viewport)));
        assert!(cursor_over_scene_viewport((1600.0, 600.0), 2.0, Some(viewport)));
        assert!(!cursor_over_scene_viewport((800.0, 300.0), 1.0, None));
    }
}
