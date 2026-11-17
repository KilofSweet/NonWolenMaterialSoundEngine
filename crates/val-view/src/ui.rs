//! Runtime editor for the serializable VAL configuration.
//!
//! Widgets mutate a draft `AppConfig`; they never allocate GPU resources. The
//! caller validates the draft and rebuilds the material volume/transport field
//! only after the user presses Apply.

use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use glam::Vec3;
use val_core::config::{AppConfig, EmitterConfig, MaterialConfig};

use crate::renderer::CamUniform;
use crate::state::PlaybackState;

/// UI-only color themes. These values affect egui styling, never the GPU scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiTheme {
    Light,
    #[default]
    Dark,
    Ocean,
    TealCoral,
    Crimson,
}

impl UiTheme {
    pub const ALL: [Self; 5] = [
        Self::Light,
        Self::Dark,
        Self::Ocean,
        Self::TealCoral,
        Self::Crimson,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Light => "Light",
            Self::Dark => "Dark",
            Self::Ocean => "Palette 1 — Ocean / Orange",
            Self::TealCoral => "Palette 2 — Teal / Coral",
            Self::Crimson => "Palette 3 — Crimson",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
            Self::Ocean => "ocean",
            Self::TealCoral => "teal-coral",
            Self::Crimson => "crimson",
        }
    }

    fn from_key(key: &str) -> Self {
        match key.trim() {
            "light" => Self::Light,
            "ocean" => Self::Ocean,
            "teal-coral" => Self::TealCoral,
            "crimson" => Self::Crimson,
            _ => Self::Dark,
        }
    }

    fn colors(
        self,
    ) -> (
        egui::Color32,
        egui::Color32,
        egui::Color32,
        egui::Color32,
        egui::Color32,
    ) {
        use egui::Color32 as C;
        match self {
            Self::Light => (
                C::from_rgb(246, 247, 249),
                C::from_rgb(255, 255, 255),
                C::from_rgb(35, 42, 54),
                C::from_rgb(50, 115, 175),
                C::from_rgb(220, 226, 234),
            ),
            Self::Dark => (
                C::from_rgb(23, 26, 36),
                C::from_rgb(29, 34, 47),
                C::from_rgb(225, 231, 240),
                C::from_rgb(65, 155, 210),
                C::from_rgb(57, 66, 84),
            ),
            Self::Ocean => (
                C::from_rgb(0, 27, 43),
                C::from_rgb(12, 52, 66),
                C::from_rgb(255, 236, 210),
                C::from_rgb(255, 128, 0),
                C::from_rgb(23, 97, 107),
            ),
            Self::TealCoral => (
                C::from_rgb(9, 38, 41),
                C::from_rgb(16, 57, 58),
                C::from_rgb(240, 237, 225),
                C::from_rgb(240, 45, 53),
                C::from_rgb(33, 124, 120),
            ),
            Self::Crimson => (
                C::from_rgb(41, 44, 64),
                C::from_rgb(54, 59, 80),
                C::from_rgb(238, 242, 243),
                C::from_rgb(240, 32, 67),
                C::from_rgb(145, 157, 176),
            ),
        }
    }

    pub fn apply(self, ctx: &egui::Context) {
        use egui::{Color32, Stroke, Visuals};
        let (panel, window, text, accent, muted) = self.colors();
        let mut visuals = if self == Self::Light {
            Visuals::light()
        } else {
            Visuals::dark()
        };
        visuals.panel_fill = panel;
        visuals.window_fill = window;
        visuals.extreme_bg_color = if self == Self::Light {
            Color32::from_rgb(235, 238, 242)
        } else {
            Color32::from_rgb(8, 13, 22)
        };
        visuals.faint_bg_color = muted;
        visuals.code_bg_color = window;
        visuals.override_text_color = Some(text);
        visuals.hyperlink_color = accent;
        visuals.selection.bg_fill = accent;
        visuals.selection.stroke = Stroke::new(1.0, text);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text);
        visuals.widgets.noninteractive.bg_fill = panel;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text);
        visuals.widgets.inactive.bg_fill = muted;
        visuals.widgets.inactive.weak_bg_fill = muted;
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, text);
        visuals.widgets.hovered.bg_fill = accent;
        visuals.widgets.hovered.weak_bg_fill = accent;
        visuals.widgets.active.fg_stroke = Stroke::new(1.0, text);
        visuals.widgets.active.bg_fill = accent;
        visuals.widgets.active.weak_bg_fill = accent;
        visuals.widgets.open.fg_stroke = Stroke::new(1.0, text);
        visuals.widgets.open.bg_fill = muted;
        visuals.widgets.open.weak_bg_fill = muted;
        ctx.set_visuals(visuals);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiAction {
    None,
    Apply,
    Save,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewportMode {
    /// Existing GPU packet-path debug overlay.
    Paths,
    /// Existing energy-volume rendering.
    #[default]
    WaveField,
    /// UI mode placeholder; the current renderer does not yet isolate a material.
    MaterialInterior,
}

impl UiTheme {
    fn load() -> Self {
        std::fs::read_to_string("val-theme.txt")
            .map(|value| Self::from_key(&value))
            .unwrap_or_default()
    }

    fn save(self) {
        // Kept separate from val.toml so GUI preferences do not modify engine configuration.
        let _ = std::fs::write("val-theme.txt", self.key());
    }
}

pub struct UiState {
    pub selected_emitter: usize,
    pub selected_material: usize,
    /// Last scene viewport rectangle in egui points. Input routing uses this
    /// instead of egui's generic consumed flag so camera drags work in the
    /// viewport without stealing interactions from the editor panel.
    pub viewport_rect: Option<egui::Rect>,
    /// Draw the world frame and material AABB wireframes in the scene overlay.
    pub show_boundaries: bool,
    /// Draw the sampled packet paths recorded by the transport compute shader.
    pub debug_show_rays: bool,
    /// Active viewport presentation tab. Only Paths and WaveField are currently rendered.
    pub viewport_mode: ViewportMode,
    pub theme: UiTheme,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            selected_emitter: 0,
            selected_material: 0,
            viewport_rect: None,
            show_boundaries: true,
            debug_show_rays: false,
            viewport_mode: ViewportMode::WaveField,
            theme: UiTheme::load(),
        }
    }
}

pub fn draw(
    ctx: &egui::Context,
    config: &mut AppConfig,
    playback: &mut PlaybackState,
    state: &mut UiState,
    status: &str,
    camera: &CamUniform,
) -> UiAction {
    let mut action = UiAction::None;
    state.theme.apply(ctx);

    // 1) Application menu and command toolbar (Blender / CAD-style layout).
    egui::TopBottomPanel::top("val_main_toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.strong("VAL");
            ui.separator();
            ui.menu_button("File", |ui| {
                if ui.button("Save configuration…").clicked() {
                    action = UiAction::Save;
                    ui.close_menu();
                }
            });
            ui.menu_button("Edit", |ui| {
                ui.label("Configuration editing is in the Properties panel");
            });
            ui.menu_button("View", |ui| {
                ui.label("Viewport layouts");
                ui.separator();
                ui.menu_button("Theme", |ui| {
                    for theme in UiTheme::ALL {
                        if ui
                            .selectable_label(state.theme == theme, theme.label())
                            .clicked()
                        {
                            state.theme = theme;
                            state.theme.save();
                            state.theme.apply(ctx);
                            ui.close_menu();
                        }
                    }
                });
            });
            ui.menu_button("Simulation", |ui| {
                if ui.button("Apply & rebuild").clicked() {
                    action = UiAction::Apply;
                    ui.close_menu();
                }
            });
            ui.menu_button("Help", |ui| {
                ui.label("Virtual Acoustic Laboratory");
            });
            ui.separator();
            if ui.button("▶ Run / Pause").clicked() {
                playback.toggle_pause();
            }
            if ui.button("↺ Reset").clicked() {
                playback.reset();
            }
            ui.separator();
            if ui.button("Apply & rebuild").clicked() {
                action = UiAction::Apply;
            }
            if ui.button("Save").clicked() {
                action = UiAction::Save;
            }
        });
    });

    // 2) Scene Explorer: selection and hierarchy only; detailed editing stays on the right.
    egui::SidePanel::left("val_scene_explorer")
        .default_width(220.0)
        .min_width(170.0)
        .max_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Scene");
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::CollapsingHeader::new(format!("Emitters ({})", config.emitters.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        for (index, emitter) in config.emitters.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    state.selected_emitter == index,
                                    format!("◉ {}", emitter.name),
                                )
                                .clicked()
                            {
                                state.selected_emitter = index;
                                playback.frequency_hz = emitter.frequency_hz;
                            }
                        }
                    });
                egui::CollapsingHeader::new(format!("Materials ({})", config.materials.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        for (index, material) in config.materials.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    state.selected_material == index,
                                    format!("▧ {}", material.name),
                                )
                                .clicked()
                            {
                                state.selected_material = index;
                            }
                        }
                    });
                egui::CollapsingHeader::new("Environment")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(format!(
                            "World: {:.0} × {:.0} × {:.0} mm",
                            config.world.dimensions_mm[0],
                            config.world.dimensions_mm[1],
                            config.world.dimensions_mm[2]
                        ));
                        ui.label(format!("Voxel: {:.2} mm", config.world.voxel_size_mm));
                    });
            });
        });

    // 3) Properties Inspector: existing configuration editors, moved to the right.
    egui::SidePanel::right("val_properties_inspector")
        .default_width(360.0)
        .min_width(290.0)
        .max_width(500.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.heading("Properties");
                    ui.label("Selected scene object and experiment settings");
                    ui.horizontal(|ui| {
                        ui.label("Interface theme");
                        let mut selected_theme = state.theme;
                        egui::ComboBox::from_id_source("val_theme_properties")
                            .selected_text(selected_theme.label())
                            .show_ui(ui, |ui| {
                                for theme in UiTheme::ALL {
                                    ui.selectable_value(&mut selected_theme, theme, theme.label());
                                }
                            });
                        if selected_theme != state.theme {
                            state.theme = selected_theme;
                            state.theme.save();
                            state.theme.apply(ctx);
                        }
                    });
                    ui.add_space(6.0);
                    egui::CollapsingHeader::new("Viewport overlays")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut state.show_boundaries, "Show world/material boundaries");
                            ui.small("Use the Packet paths tab to show the sampled GPU packet trajectories.");
                            ui.small("Ray view draws a deterministic sample of actual GPU packet paths; the energy solve still uses every configured packet.");
                            ui.small("Line color identifies the emitter; brightness follows the packet's remaining energy.");
                            ui.small(format!(
                                "Debug sample: up to {} rays per emitter × {} segments per ray.",
                                crate::renderer::DEBUG_RAYS_PER_EMITTER,
                                crate::renderer::DEBUG_MAX_SEGMENTS_PER_RAY,
                            ));
                        });

                    egui::CollapsingHeader::new("Sound emitters")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("＋ Add emitter").clicked() {
                                    let mut emitter = EmitterConfig::default();
                                    emitter.name = unique_name("emitter", config.emitters.iter().map(|item| item.name.as_str()));
                                    emitter.position_mm = [
                                        18.0_f32.min(config.world.dimensions_mm[0] * 0.25),
                                        (config.world.dimensions_mm[1] * 0.25).max(config.world.voxel_size_mm),
                                        config.world.dimensions_mm[2] * 0.5,
                                    ];
                                    emitter.position_mm[0] = (emitter.position_mm[0] / config.world.voxel_size_mm).round()
                                        * config.world.voxel_size_mm;
                                    config.emitters.push(emitter);
                                    state.selected_emitter = config.emitters.len() - 1;
                                }
                                if ui.add_enabled(config.emitters.len() > 1, egui::Button::new("Remove")).clicked() {
                                    if state.selected_emitter < config.emitters.len() {
                                        config.emitters.remove(state.selected_emitter);
                                    }
                                    state.selected_emitter = state.selected_emitter.min(config.emitters.len().saturating_sub(1));
                                }
                            });

                            if !config.emitters.is_empty() {
                                state.selected_emitter = state.selected_emitter.min(config.emitters.len() - 1);
                                let previous_selection = state.selected_emitter;
                                egui::ComboBox::from_id_source("selected_emitter")
                                    .selected_text(&config.emitters[state.selected_emitter].name)
                                    .show_ui(ui, |ui| {
                                        for (index, emitter) in config.emitters.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut state.selected_emitter,
                                                index,
                                                &emitter.name,
                                            );
                                        }
                                    });
                                if state.selected_emitter != previous_selection {
                                    playback.frequency_hz = config.emitters[state.selected_emitter].frequency_hz;
                                }

                                let emitter = &mut config.emitters[state.selected_emitter];
                                ui.label("Identity");
                                ui.text_edit_singleline(&mut emitter.name);
                                ui.label("Position (mm)");
                                vec3_drag(ui, "X", &mut emitter.position_mm[0], 0.5);
                                vec3_drag(ui, "Y", &mut emitter.position_mm[1], 0.5);
                                vec3_drag(ui, "Z", &mut emitter.position_mm[2], 0.5);
                                scalar(ui, "Source radius (mm)", &mut emitter.radius_mm, 0.1, 0.1..=50.0);
                                scalar(ui, "Frequency (Hz)", &mut emitter.frequency_hz, 100.0, 1.0..=200_000.0);
                                scalar(ui, "Relative amplitude", &mut emitter.amplitude, 0.01, 0.0..=100.0);
                                ui.label("Emission direction");
                                vec3_drag(ui, "X##dir", &mut emitter.direction[0], 0.05);
                                vec3_drag(ui, "Y##dir", &mut emitter.direction[1], 0.05);
                                vec3_drag(ui, "Z##dir", &mut emitter.direction[2], 0.05);
                                scalar(
                                    ui,
                                    "Cone half-angle (degrees)",
                                    &mut emitter.cone_half_angle_degrees,
                                    1.0,
                                    1.0..=180.0,
                                );
                            }
                            ui.small("Emitter changes require a transport rebuild. Amplitudes are relative, not calibrated watts.");
                        });

                    egui::CollapsingHeader::new("Felt materials")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("＋ Add material").clicked() {
                                    let mut material = MaterialConfig::default();
                                    material.name = unique_name("felt", config.materials.iter().map(|item| item.name.as_str()));
                                    let voxel = config.world.voxel_size_mm;
                                    let x_start = (config.emitters.first().map(|e| e.position_mm[0] + e.radius_mm + voxel * 3.0).unwrap_or(32.0)
                                        / voxel).ceil() * voxel;
                                    let x_size = (32.0 / voxel).round().max(1.0) * voxel;
                                    material.bounds_min_mm = [
                                        x_start,
                                        (16.0_f32.min(config.world.dimensions_mm[1] * 0.1) / voxel).floor() * voxel,
                                        (16.0_f32.min(config.world.dimensions_mm[2] * 0.1) / voxel).floor() * voxel,
                                    ];
                                    material.bounds_max_mm = [
                                        (x_start + x_size).min(config.world.dimensions_mm[0]),
                                        (config.world.dimensions_mm[1] * 0.85 / voxel).floor() * voxel,
                                        (config.world.dimensions_mm[2] * 0.85 / voxel).floor() * voxel,
                                    ];
                                    material.visual_color = [0.18, 0.72, 0.82];
                                    config.materials.push(material);
                                    state.selected_material = config.materials.len() - 1;
                                }
                                if ui.add_enabled(config.materials.len() > 1, egui::Button::new("Remove")).clicked() {
                                    if state.selected_material < config.materials.len() {
                                        config.materials.remove(state.selected_material);
                                    }
                                    state.selected_material = state.selected_material.min(config.materials.len().saturating_sub(1));
                                }
                            });

                            if !config.materials.is_empty() {
                                state.selected_material = state.selected_material.min(config.materials.len() - 1);
                                egui::ComboBox::from_id_source("selected_material")
                                    .selected_text(&config.materials[state.selected_material].name)
                                    .show_ui(ui, |ui| {
                                        for (index, material) in config.materials.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut state.selected_material,
                                                index,
                                                &material.name,
                                            );
                                        }
                                    });

                                let material = &mut config.materials[state.selected_material];
                                ui.label("Identity");
                                ui.text_edit_singleline(&mut material.name);
                                let mut custom_seed = material.seed.is_some();
                                if ui.checkbox(&mut custom_seed, "Override world-derived seed").changed() {
                                    material.seed = if custom_seed { Some(1) } else { None };
                                }
                                if let Some(seed) = material.seed.as_mut() {
                                    ui.add(egui::DragValue::new(seed).prefix("Material seed "));
                                }
                                ui.label("Bounds · millimetres · half-open AABB");
                                for axis in 0..3 {
                                    ui.horizontal(|ui| {
                                        ui.label(["X", "Y", "Z"][axis]);
                                        ui.add(egui::DragValue::new(&mut material.bounds_min_mm[axis]).speed(0.5).suffix(" min"));
                                        ui.add(egui::DragValue::new(&mut material.bounds_max_mm[axis]).speed(0.5).suffix(" max"));
                                    });
                                }
                                ui.separator();
                                ui.label("Transport properties");
                                scalar(ui, "Absorption σa (m⁻¹)", &mut material.sigma_a_m_inv, 0.5, 0.0..=10_000.0);
                                scalar(ui, "Scattering σs (m⁻¹)", &mut material.sigma_s_m_inv, 0.5, 0.0..=10_000.0);
                                scalar(ui, "Anisotropy g", &mut material.anisotropy_g, 0.01, -0.99..=0.99);
                                scalar(ui, "Sound speed (m/s)", &mut material.sound_speed_mps, 1.0, 40.0..=343.0);
                                scalar(ui, "Bulk density (kg/m³)", &mut material.density_kg_m3, 1.0, 0.1..=10_000.0);
                                scalar(ui, "Effective density (kg/m³)", &mut material.effective_density_kg_m3, 0.05, 0.1..=100.0);
                                scalar(ui, "Porosity", &mut material.porosity, 0.01, 0.0..=1.0);
                                ui.separator();
                                ui.label("Structure laws");
                                egui::ComboBox::from_id_source("gradient_axis")
                                    .selected_text(["X", "Y", "Z"][material.gradient_axis.min(2)])
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut material.gradient_axis, 0, "X");
                                        ui.selectable_value(&mut material.gradient_axis, 1, "Y");
                                        ui.selectable_value(&mut material.gradient_axis, 2, "Z");
                                    });
                                scalar(ui, "Gradient strength", &mut material.gradient_strength, 0.01, 0.0..=5.0);
                                scalar(ui, "Clump scale (m⁻¹)", &mut material.clump_scale_per_m, 1.0, 0.1..=10_000.0);
                                scalar(ui, "Clump strength", &mut material.clump_strength, 0.01, 0.0..=1.0);
                                ui.separator();
                                ui.label("Appearance");
                                ui.color_edit_button_rgb(&mut material.visual_color);
                                scalar(ui, "Base opacity", &mut material.visual_opacity, 0.001, 0.0..=1.0);
                            }
                            ui.small("Material geometry and transport coefficients are regenerated on Apply.");
                        });

                    egui::CollapsingHeader::new("World and air")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label("World dimensions (mm)");
                            vec3_drag(ui, "X##world", &mut config.world.dimensions_mm[0], 1.0);
                            vec3_drag(ui, "Y##world", &mut config.world.dimensions_mm[1], 1.0);
                            vec3_drag(ui, "Z##world", &mut config.world.dimensions_mm[2], 1.0);
                            scalar(ui, "Voxel size (mm)", &mut config.world.voxel_size_mm, 0.1, 0.1..=20.0);
                            scalar(ui, "Air sound speed (m/s)", &mut config.world.air_speed_mps, 1.0, 100.0..=500.0);
                            scalar(ui, "Air density (kg/m³)", &mut config.world.air_density_kg_m3, 0.01, 0.1..=10.0);
                            scalar(ui, "Air absorption (m⁻¹)", &mut config.world.air_absorption_m_inv, 0.01, 0.0..=100.0);
                            ui.add(egui::DragValue::new(&mut config.world.seed).prefix("Seed "));
                        });

                    egui::CollapsingHeader::new("Camera")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label("Look-at target (mm)");
                            vec3_drag(ui, "X##camera", &mut config.camera.target_mm[0], 1.0);
                            vec3_drag(ui, "Y##camera", &mut config.camera.target_mm[1], 1.0);
                            vec3_drag(ui, "Z##camera", &mut config.camera.target_mm[2], 1.0);
                            scalar(ui, "Yaw (rad)", &mut config.camera.yaw_radians, 0.01, -6.283..=6.283);
                            scalar(ui, "Pitch (rad)", &mut config.camera.pitch_radians, 0.01, -1.5..=1.5);
                            scalar(ui, "Distance (mm)", &mut config.camera.distance_mm, 1.0, 1.0..=100_000.0);
                            scalar(ui, "Minimum distance", &mut config.camera.min_distance_mm, 1.0, 1.0..=100_000.0);
                            scalar(ui, "Maximum distance", &mut config.camera.max_distance_mm, 1.0, 1.0..=100_000.0);
                            scalar(ui, "Movement speed (m/s)", &mut config.camera.movement_speed_mps, 0.01, 0.001..=10.0);
                        });

                    egui::CollapsingHeader::new("Transport and time")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.add(egui::DragValue::new(&mut config.simulation.ray_count).speed(1024).prefix("Packets / emitter "));
                            ui.add(egui::DragValue::new(&mut config.simulation.max_steps).speed(8).prefix("Max segments / packet "));
                            scalar(ui, "Step length (mm)", &mut config.simulation.step_mm, 0.05, 0.01..=10.0);
                            scalar(ui, "Timeline rate (µs/s)", &mut config.simulation.timeline_rate_us_per_second, 1.0, 0.1..=10_000.0);
                            scalar(ui, "Time-bin width (µs)", &mut config.simulation.energy_time_bin_us, 1.0, 0.1..=10_000.0);
                            ui.add(egui::DragValue::new(&mut config.simulation.energy_time_bins).speed(1).prefix("Time bins "));
                        });

                    egui::CollapsingHeader::new("Visualization")
                        .default_open(false)
                        .show(ui, |ui| {
                            scalar(ui, "Shell thickness (mm)", &mut config.visualization.wave_shell_thickness_mm, 0.1, 0.1..=20.0);
                            scalar(ui, "Air energy gain", &mut config.visualization.air_wave_gain, 10.0, 0.0..=100_000.0);
                            scalar(ui, "Material energy gain", &mut config.visualization.felt_energy_gain, 10.0, 0.0..=100_000.0);
                            scalar(ui, "Source glow", &mut config.visualization.source_glow_gain, 0.05, 0.0..=10.0);
                            scalar(ui, "Material base opacity multiplier", &mut config.visualization.felt_base_opacity_multiplier, 0.05, 0.0..=10.0);
                            scalar(ui, "Visual fade (ms)", &mut config.visualization.voxel_fade_ms, 0.001, 0.001..=10.0);
                            scalar(ui, "Material max opacity", &mut config.visualization.felt_energy_max_opacity, 0.005, 0.0..=1.0);
                            scalar(ui, "Air max opacity", &mut config.visualization.air_wave_max_opacity, 0.005, 0.0..=1.0);
                            ui.color_edit_button_rgb(&mut config.visualization.energy_hot_color);
                        });

                    egui::CollapsingHeader::new("Rendering")
                        .default_open(false)
                        .show(ui, |ui| {
                            scalar(ui, "Resolution scale", &mut config.render.resolution_scale, 0.01, 0.1..=1.0);
                            ui.add(egui::DragValue::new(&mut config.render.max_width).speed(8).prefix("Max width "));
                            ui.add(egui::DragValue::new(&mut config.render.max_height).speed(8).prefix("Max height "));
                            ui.add(egui::Slider::new(&mut config.render.target_fps, 5..=60).text("FPS cap"));
                            ui.checkbox(&mut config.render.pause_when_unfocused, "Pause on focus loss");
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Apply & rebuild").clicked() {
                            action = UiAction::Apply;
                        }
                        if ui.button("Save val.toml").clicked() {
                            action = UiAction::Save;
                        }
                    });
                    if !status.is_empty() {
                        ui.add_space(4.0);
                        ui.label(status);
                    }
                    ui.add_space(4.0);
                    ui.small("Apply rebuilds the material field and precomputed GPU transport history. Save writes the current draft to val.toml.");
                });
        });

    // 4) Timeline and 5) diagnostics/status dock.
    egui::TopBottomPanel::bottom("val_timeline_diagnostics")
        .resizable(true)
        .default_height(142.0)
        .min_height(105.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Timeline");
                ui.separator();
                if ui
                    .button(if playback.paused {
                        "▶ Play"
                    } else {
                        "Ⅱ Pause"
                    })
                    .clicked()
                {
                    playback.toggle_pause();
                }
                if ui.button("↺ Reset").clicked() {
                    playback.reset();
                }
                ui.add(egui::Slider::new(&mut playback.speed, 0.05..=16.0).text("Speed"));
                ui.add(
                    egui::Slider::new(&mut playback.frequency_hz, 1.0..=200_000.0)
                        .logarithmic(true)
                        .text("Frequency (Hz)"),
                );
            });
            let history_end =
                config.simulation.energy_time_bin_us * config.simulation.energy_time_bins as f32;
            ui.add(egui::Slider::new(&mut playback.time_us, 0.0..=history_end).text("Time (µs)"));
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.strong("Diagnostics");
                ui.label(format!(
                    "{}",
                    if playback.paused { "Paused" } else { "Playing" }
                ));
                ui.separator();
                ui.label(format!("{} emitters", config.emitters.len()));
                ui.label(format!("{} material regions", config.materials.len()));
                ui.label(format!("{} packets / emitter", config.simulation.ray_count));
                ui.separator();
                ui.label(status);
            });
        });

    // Draw scene-space previews in the central viewport. The three tabs are a
    // presentation shell: Paths and WaveField map to existing renderer features;
    // MaterialInterior is intentionally marked as not yet backed by a renderer.
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Viewport");
                ui.separator();
                ui.selectable_value(&mut state.viewport_mode, ViewportMode::Paths, "① Packet paths");
                ui.selectable_value(&mut state.viewport_mode, ViewportMode::WaveField, "② Wave field");
                ui.selectable_value(&mut state.viewport_mode, ViewportMode::MaterialInterior, "③ Inside material");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Perspective · 3D");
                });
            });
            ui.separator();
            state.debug_show_rays = state.viewport_mode == ViewportMode::Paths;
            if state.viewport_mode == ViewportMode::MaterialInterior {
                ui.colored_label(Color32::from_rgb(235, 190, 95), "Material interior view: UI tab is prepared; dedicated material-volume rendering is not implemented yet.");
            }
            let viewport = ui.available_rect_before_wrap();
            state.viewport_rect = Some(viewport);
            draw_scene_preview(ui.painter(), viewport, camera, config, state);
        });

    action
}

#[derive(Clone, Copy)]
struct SceneProjection {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    aspect: f32,
    fov: f32,
    rect: Rect,
}

impl SceneProjection {
    fn new(camera: &CamUniform, rect: Rect) -> Option<Self> {
        let origin = Vec3::new(camera.pos[0], camera.pos[1], camera.pos[2]);
        let target = Vec3::new(camera.look_at[0], camera.look_at[1], camera.look_at[2]);
        let world_up = Vec3::new(camera.up[0], camera.up[1], camera.up[2]);
        let forward_delta = target - origin;
        if forward_delta.length_squared() < 1.0e-10 {
            return None;
        }
        let forward = forward_delta.normalize();
        let right_delta = forward.cross(world_up);
        if right_delta.length_squared() < 1.0e-10 {
            return None;
        }
        let right = right_delta.normalize();
        let up = right.cross(forward).normalize();
        // The HUD docks reduce the actual scene viewport. Its projection must
        // use the central panel's aspect ratio, not the full window's ratio.
        let viewport_aspect = rect.width() / rect.height().max(1.0);
        if !viewport_aspect.is_finite()
            || viewport_aspect <= 0.0
            || !camera.fov.is_finite()
            || camera.fov <= 0.0
        {
            return None;
        }
        Some(Self {
            origin,
            forward,
            right,
            up,
            aspect: viewport_aspect,
            fov: camera.fov,
            rect,
        })
    }

    fn camera_point(&self, world: Vec3) -> Vec3 {
        let relative = world - self.origin;
        Vec3::new(
            relative.dot(self.right),
            relative.dot(self.up),
            relative.dot(self.forward),
        )
    }

    fn project_camera_point(&self, point: Vec3) -> Option<Pos2> {
        if point.z <= 0.001 || !point.is_finite() {
            return None;
        }
        let ndc_x = point.x / (point.z * self.aspect);
        let ndc_y = point.y / (point.z * self.fov);
        let screen = Pos2::new(
            self.rect.center().x + ndc_x * self.rect.width() * 0.5,
            self.rect.center().y - ndc_y * self.rect.height() * 0.5,
        );
        (screen.x.is_finite() && screen.y.is_finite()).then_some(screen)
    }

    fn project(&self, world: Vec3) -> Option<(Pos2, f32)> {
        let camera_point = self.camera_point(world);
        Some((self.project_camera_point(camera_point)?, camera_point.z))
    }

    /// Clip a 3D edge against the near plane before projecting it. This keeps
    /// a box outline useful when the camera moves partly inside an object.
    fn project_segment(&self, a: Vec3, b: Vec3) -> Option<[Pos2; 2]> {
        let near = 0.001;
        let mut a = self.camera_point(a);
        let mut b = self.camera_point(b);
        if a.z <= near && b.z <= near {
            return None;
        }
        if a.z <= near {
            let t = (near - a.z) / (b.z - a.z);
            a = a.lerp(b, t.clamp(0.0, 1.0));
        }
        if b.z <= near {
            let t = (near - b.z) / (a.z - b.z);
            b = b.lerp(a, t.clamp(0.0, 1.0));
        }
        Some([self.project_camera_point(a)?, self.project_camera_point(b)?])
    }
}

const BOX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
];

fn box_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}

fn rgb8(rgb: [f32; 3]) -> [u8; 3] {
    rgb.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn draw_wire_box(
    painter: &egui::Painter,
    projection: &SceneProjection,
    min: Vec3,
    max: Vec3,
    rgb: [f32; 3],
    selected: bool,
    label: &str,
) {
    if !min.is_finite() || !max.is_finite() {
        return;
    }
    let corners = box_corners(min, max);
    let [r, g, b] = rgb8(rgb);
    let glow = Color32::from_rgba_unmultiplied(r, g, b, if selected { 42 } else { 24 });
    let middle = Color32::from_rgba_unmultiplied(r, g, b, if selected { 125 } else { 78 });
    let core = Color32::from_rgba_unmultiplied(r, g, b, if selected { 245 } else { 165 });

    for (a, b) in BOX_EDGES {
        if let Some(points) = projection.project_segment(corners[a], corners[b]) {
            // Multiple strokes form a soft glow with a sharp centre line.
            painter.line_segment(points, Stroke::new(if selected { 8.0 } else { 5.5 }, glow));
            painter.line_segment(
                points,
                Stroke::new(if selected { 3.2 } else { 2.0 }, middle),
            );
            painter.line_segment(points, Stroke::new(if selected { 1.5 } else { 0.9 }, core));
        }
    }

    if let Some((anchor, _)) = projection.project(corners[0]) {
        painter.text(
            anchor + Vec2::new(7.0, -6.0),
            Align2::LEFT_BOTTOM,
            label,
            FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(r, g, b, if selected { 255 } else { 205 }),
        );
    }
}

fn draw_emitter_marker(
    painter: &egui::Painter,
    projection: &SceneProjection,
    emitter: &EmitterConfig,
    selected: bool,
) {
    let center_world = Vec3::new(
        emitter.position_mm[0] * 0.001,
        emitter.position_mm[1] * 0.001,
        emitter.position_mm[2] * 0.001,
    );
    let Some((center, depth)) = projection.project(center_world) else {
        return;
    };

    let (color, glow) = if selected {
        (
            Color32::from_rgb(255, 211, 126),
            Color32::from_rgba_unmultiplied(255, 165, 65, 85),
        )
    } else {
        (
            Color32::from_rgb(105, 226, 245),
            Color32::from_rgba_unmultiplied(55, 185, 225, 58),
        )
    };
    let radius_m = (emitter.radius_mm * 0.001).max(0.001);
    let radius_px =
        (radius_m / (depth * projection.fov) * projection.rect.height() * 0.5).clamp(4.0, 18.0);

    painter.circle_stroke(center, radius_px + 2.0, Stroke::new(7.0, glow));
    painter.circle_stroke(
        center,
        radius_px,
        Stroke::new(if selected { 2.0 } else { 1.4 }, color),
    );
    painter.circle_stroke(center, radius_px * 0.42, Stroke::new(1.0, color));
    painter.circle_filled(center, if selected { 3.2 } else { 2.4 }, color);

    // A short axis line makes the emitter's configured directivity visible.
    let direction = Vec3::new(
        emitter.direction[0],
        emitter.direction[1],
        emitter.direction[2],
    );
    if direction.length_squared() > 1.0e-8 {
        let direction_end = center_world + direction.normalize() * (radius_m * 3.0).max(0.006);
        if let Some(points) = projection.project_segment(center_world, direction_end) {
            painter.line_segment(points, Stroke::new(4.0, glow));
            painter.line_segment(points, Stroke::new(1.5, color));
            let tip = points[1];
            let delta = points[1] - points[0];
            if delta.length_sq() > 1.0 {
                let unit = delta.normalized();
                let side = Vec2::new(-unit.y, unit.x);
                let back = tip - unit * 5.0;
                painter.line_segment([tip, back + side * 3.0], Stroke::new(1.5, color));
                painter.line_segment([tip, back - side * 3.0], Stroke::new(1.5, color));
            }
        }
    }

    painter.text(
        center + Vec2::new(radius_px + 7.0, -radius_px - 4.0),
        Align2::LEFT_BOTTOM,
        format!(
            "{} · {:.1} kHz",
            emitter.name,
            emitter.frequency_hz / 1000.0
        ),
        FontId::proportional(12.0),
        color,
    );
}

fn draw_scene_preview(
    painter: &egui::Painter,
    rect: Rect,
    camera: &CamUniform,
    config: &AppConfig,
    state: &UiState,
) {
    let Some(projection) = SceneProjection::new(camera, rect) else {
        return;
    };

    if state.show_boundaries {
        // World frame provides a stable coordinate reference while positioning items.
        let world_min = Vec3::ZERO;
        let world_max = Vec3::new(
            config.world.dimensions_mm[0] * 0.001,
            config.world.dimensions_mm[1] * 0.001,
            config.world.dimensions_mm[2] * 0.001,
        );
        draw_wire_box(
            painter,
            &projection,
            world_min,
            world_max,
            [0.28, 0.42, 0.56],
            false,
            "WORLD",
        );

        // These are draft bounds, not only the last applied material field. New
        // regions therefore appear in the viewport as soon as they are created.
        for (index, material) in config.materials.iter().enumerate() {
            let min = Vec3::new(
                material.bounds_min_mm[0] * 0.001,
                material.bounds_min_mm[1] * 0.001,
                material.bounds_min_mm[2] * 0.001,
            );
            let max = Vec3::new(
                material.bounds_max_mm[0] * 0.001,
                material.bounds_max_mm[1] * 0.001,
                material.bounds_max_mm[2] * 0.001,
            );
            draw_wire_box(
                painter,
                &projection,
                min,
                max,
                material.visual_color,
                index == state.selected_material,
                &material.name,
            );
        }
    }

    // All draft emitters remain visible simultaneously. Selection only changes
    // their marker emphasis; it does not move or hide the other source pointers.
    for (index, emitter) in config.emitters.iter().enumerate() {
        draw_emitter_marker(
            painter,
            &projection,
            emitter,
            index == state.selected_emitter,
        );
    }
}

fn unique_name<'a>(prefix: &str, existing: impl Iterator<Item = &'a str>) -> String {
    let existing: Vec<String> = existing.map(|name| name.to_lowercase()).collect();
    let mut suffix = 1u32;
    loop {
        let candidate = format!("{prefix}-{suffix}");
        if !existing
            .iter()
            .any(|name| name == &candidate.to_lowercase())
        {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

fn scalar(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    speed: f64,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).speed(speed).range(range));
    });
}

fn vec3_drag(ui: &mut egui::Ui, label: &str, value: &mut f32, speed: f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).speed(speed));
    });
}

#[cfg(test)]
mod scene_overlay_tests {
    use super::UiState;

    #[test]
    fn boundaries_start_visible_and_ray_debug_starts_disabled() {
        let state = UiState::default();
        assert!(state.show_boundaries);
        assert!(!state.debug_show_rays);
    }
}
