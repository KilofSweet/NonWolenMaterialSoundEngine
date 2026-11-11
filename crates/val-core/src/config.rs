use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AppConfig {
    pub world: WorldConfig,
    pub emitters: Vec<EmitterConfig>,
    pub simulation: SimulationConfig,
    pub render: RenderConfig,
    pub camera: CameraConfig,
    pub visualization: VisualizationConfig,
    pub materials: Vec<MaterialConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            world: WorldConfig::default(),
            emitters: vec![EmitterConfig::default()],
            simulation: SimulationConfig::default(),
            render: RenderConfig::default(),
            camera: CameraConfig::default(),
            visualization: VisualizationConfig::default(),
            materials: vec![MaterialConfig::default()],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct WorldConfig {
    pub dimensions_mm: [f32; 3],
    pub voxel_size_mm: f32,
    pub air_speed_mps: f32,
    /// Ambient-air density used for impedance at material boundaries.
    pub air_density_kg_m3: f32,
    /// Energy attenuation coefficient in air (m^-1).
    pub air_absorption_m_inv: f32,
    pub seed: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            dimensions_mm: [128.0, 96.0, 96.0],
            voxel_size_mm: 1.0,
            air_speed_mps: 343.0,
            air_density_kg_m3: 1.204,
            air_absorption_m_inv: 0.0,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct EmitterConfig {
    pub name: String,
    pub position_mm: [f32; 3],
    pub radius_mm: f32,
    pub frequency_hz: f32,
    /// Relative emitter amplitude used to scale packet weights (not calibrated watts).
    pub amplitude: f32,
    /// Unit direction of the emitter. The packet solver samples a cone around it.
    pub direction: [f32; 3],
    /// Half-angle of the emission cone. 180 degrees means omnidirectional.
    pub cone_half_angle_degrees: f32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            name: "emitter-1".into(),
            position_mm: [18.0, 48.0, 48.0],
            radius_mm: 3.2,
            frequency_hz: 20_000.0,
            amplitude: 1.0,
            direction: [1.0, 0.0, 0.0],
            cone_half_angle_degrees: 25.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SimulationConfig {
    pub ray_count: u32,
    pub max_steps: u32,
    pub step_mm: f32,
    /// Timeline microseconds advanced per real second at 1x playback.
    pub timeline_rate_us_per_second: f32,
    /// Temporal histogram resolution for packet energy in every world voxel (air and materials).
    pub energy_time_bin_us: f32,
    pub energy_time_bins: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            ray_count: 131_072,
            // Maximum spatial segments per packet path; unrelated to animation FPS.
            max_steps: 512,
            step_mm: 0.75,
            timeline_rate_us_per_second: 60.0,
            energy_time_bin_us: 20.0,
            energy_time_bins: 24,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct RenderConfig {
    /// Internal volume-render resolution as a fraction of surface resolution.
    pub resolution_scale: f32,
    pub max_width: u32,
    pub max_height: u32,
    pub target_fps: u32,
    /// Automatically stop timeline redraws when the window loses focus.
    pub pause_when_unfocused: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            resolution_scale: 0.22,
            max_width: 420,
            max_height: 315,
            target_fps: 60,
            pause_when_unfocused: true,
        }
    }
}


#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CameraConfig {
    pub target_mm: [f32; 3],
    pub yaw_radians: f32,
    pub pitch_radians: f32,
    pub distance_mm: f32,
    pub min_distance_mm: f32,
    pub max_distance_mm: f32,
    pub movement_speed_mps: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            target_mm: [64.0, 48.0, 48.0],
            yaw_radians: -0.55,
            pitch_radians: 0.35,
            distance_mm: 240.0,
            min_distance_mm: 40.0,
            max_distance_mm: 750.0,
            movement_speed_mps: 0.06,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct VisualizationConfig {
    pub wave_shell_thickness_mm: f32,
    pub air_wave_gain: f32,
    pub source_glow_gain: f32,
    pub felt_energy_gain: f32,
    pub felt_base_opacity_multiplier: f32,
    /// Exponential energy-display persistence in milliseconds. This is visual
    /// persistence only; it does not alter packet attenuation or residence time.
    pub voxel_fade_ms: f32,
    /// Maximum alpha reached by a material voxel at full normalized packet energy.
    pub felt_energy_max_opacity: f32,
    /// Maximum opacity of energy-bearing air samples.
    pub air_wave_max_opacity: f32,
    /// Linear RGB color approached by felt voxels at high deposited energy.
    pub energy_hot_color: [f32; 3],
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            wave_shell_thickness_mm: 2.5,
            air_wave_gain: 1200.0,
            source_glow_gain: 1.0,
            felt_energy_gain: 1200.0,
            felt_base_opacity_multiplier: 1.0,
            voxel_fade_ms: 0.008,
            felt_energy_max_opacity: 0.10,
            air_wave_max_opacity: 0.22,
            energy_hot_color: [1.0, 0.88, 0.42],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct MaterialConfig {
    pub name: String,
    pub bounds_min_mm: [f32; 3],
    pub bounds_max_mm: [f32; 3],
    /// Prototype transport coefficients (m^-1), not measured material data.
    pub sigma_a_m_inv: f32,
    pub sigma_s_m_inv: f32,
    pub anisotropy_g: f32,
    pub sound_speed_mps: f32,
    /// Bulk density metadata (kg/m^3).
    pub density_kg_m3: f32,
    /// Effective acoustic density used for the approximate impedance model (kg/m^3).
    /// This is not the bulk density; porous media can have a much lower dynamic density.
    pub effective_density_kg_m3: f32,
    pub porosity: f32,
    pub visual_color: [f32; 3],
    pub visual_opacity: f32,
    pub seed: Option<u64>,
    pub gradient_axis: usize,
    pub gradient_strength: f32,
    pub clump_scale_per_m: f32,
    pub clump_strength: f32,
}

impl Default for MaterialConfig {
    fn default() -> Self {
        Self {
            name: "warm-felt".into(),
            bounds_min_mm: [64.0, 16.0, 16.0],
            bounds_max_mm: [128.0, 80.0, 80.0],
            // Demonstration preset: deliberately absorptive; replace with measured data.
            sigma_a_m_inv: 60.0,
            sigma_s_m_inv: 8.0,
            anisotropy_g: 0.55,
            sound_speed_mps: 220.0,
            density_kg_m3: 180.0,
            effective_density_kg_m3: 1.8,
            porosity: 0.82,
            visual_color: [0.95, 0.43, 0.12],
            visual_opacity: 0.012,
            seed: None,
            gradient_axis: 1,
            gradient_strength: 0.5,
            clump_scale_per_m: 150.0,
            clump_strength: 0.8,
        }
    }
}

impl AppConfig {
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match fs::read_to_string(path) {
            Ok(source) => match toml::from_str::<Self>(&source) {
                Ok(config) => config,
                Err(error) => {
                    eprintln!(
                        "VAL config: failed to parse {}: {error}; using built-in defaults",
                        path.display()
                    );
                    Self::default()
                }
            },
            Err(error) => {
                eprintln!(
                    "VAL config: cannot read {}: {error}; using built-in defaults",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Persist the current settings from the runtime editor or a headless config workflow.
    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let serialized = toml::to_string_pretty(self)
            .map_err(|error| format!("failed to serialize VAL configuration: {error}"))?;
        fs::write(path.as_ref(), serialized)
            .map_err(|error| format!("failed to write {}: {error}", path.as_ref().display()))
    }

    pub fn validate(&self) -> Result<[u32; 3], String> {
        if !self.world.voxel_size_mm.is_finite() || self.world.voxel_size_mm <= 0.0 {
            return Err("world.voxel_size_mm must be finite and greater than zero".into());
        }
        if !self.world.air_speed_mps.is_finite() || self.world.air_speed_mps <= 0.0 {
            return Err("world.air_speed_mps must be finite and greater than zero".into());
        }
        if !self.world.air_density_kg_m3.is_finite()
            || !(0.1..=10.0).contains(&self.world.air_density_kg_m3)
            || !self.world.air_absorption_m_inv.is_finite()
            || self.world.air_absorption_m_inv < 0.0
            || self.world.air_absorption_m_inv > 100.0
        {
            return Err("air density or absorption coefficient is outside the supported range".into());
        }

        let mut dims = [0u32; 3];
        for axis in 0..3 {
            let cells = self.world.dimensions_mm[axis] / self.world.voxel_size_mm;
            let rounded = cells.round();
            if !cells.is_finite() || cells <= 0.0 || (cells - rounded).abs() > 0.001 {
                return Err(format!(
                    "world dimension on axis {axis} must be a positive integer multiple of voxel_size_mm"
                ));
            }
            // The config layer does not impose an arbitrary resolution cap.
            // Device texture/storage limits and fallible host/GPU allocations are
            // checked against the actual selected hardware during setup.
            if rounded >= 4_294_967_296.0 {
                return Err(format!("world dimension on axis {axis} exceeds the u32 voxel-coordinate range"));
            }
            dims[axis] = rounded as u32;
        }

        let voxel_count = (dims[0] as u64)
            .checked_mul(dims[1] as u64)
            .and_then(|v| v.checked_mul(dims[2] as u64))
            .ok_or_else(|| "world voxel count overflows u64; reduce dimensions or increase voxel_size_mm".to_string())?;

        if self.emitters.is_empty() {
            return Err("configure at least one [[emitters]] sound source".into());
        }
        if self.materials.is_empty() {
            return Err("configure at least one [[materials]] region".into());
        }
        let mut emitter_names = HashSet::with_capacity(self.emitters.len());
        for emitter in &self.emitters {
            let name = emitter.name.trim();
            if name.is_empty() {
                return Err("every emitter must have a non-empty name".into());
            }
            if !emitter_names.insert(name.to_lowercase()) {
                return Err(format!("emitter name '{}' is duplicated; give each emitter a unique name", emitter.name));
            }
        }
        let total_emitter_amplitude = self.emitters.iter()
            .map(|emitter| emitter.amplitude as f64)
            .sum::<f64>();
        if !total_emitter_amplitude.is_finite() || total_emitter_amplitude <= 0.0 {
            return Err("at least one sound emitter must have a positive amplitude".into());
        }
        let mut material_names = HashSet::with_capacity(self.materials.len());
        for material in &self.materials {
            let name = material.name.trim();
            if name.is_empty() {
                return Err("every felt material region must have a non-empty name".into());
            }
            if !material_names.insert(name.to_lowercase()) {
                return Err(format!("material name '{}' is duplicated; give each region a unique name", material.name));
            }
        }
        let mut bounds: Vec<([u32; 3], [u32; 3])> = Vec::with_capacity(self.materials.len());
        for material in self.materials.iter() {
            let mut min = [0u32; 3];
            let mut max = [0u32; 3];
            for axis in 0..3 {
                let a = material.bounds_min_mm[axis] / self.world.voxel_size_mm;
                let b = material.bounds_max_mm[axis] / self.world.voxel_size_mm;
                if !a.is_finite() || !b.is_finite() || b <= a {
                    return Err(format!(
                        "material '{}' has invalid bounds on axis {}: min={:.3} mm, max={:.3} mm. Set max greater than min and keep both finite.",
                        material.name, axis_name(axis), material.bounds_min_mm[axis], material.bounds_max_mm[axis]
                    ));
                }
                if a < 0.0 || b > dims[axis] as f32 {
                    let world_mm = self.world.dimensions_mm[axis];
                    let size_mm = material.bounds_max_mm[axis] - material.bounds_min_mm[axis];
                    if size_mm <= world_mm {
                        let delta = if a < 0.0 { -material.bounds_min_mm[axis] } else { world_mm - material.bounds_max_mm[axis] };
                        let direction = if delta < 0.0 { "-" } else { "+" };
                        return Err(format!(
                            "material '{}' is outside the world on {} (bounds {:.2}..{:.2} mm, world 0..{:.2} mm). Suggested fix: move this material {:.2} mm along {}{}; new bounds {:.2}..{:.2} mm.",
                            material.name, axis_name(axis), material.bounds_min_mm[axis], material.bounds_max_mm[axis], world_mm,
                            delta.abs(), direction, axis_name(axis), material.bounds_min_mm[axis] + delta, material.bounds_max_mm[axis] + delta
                        ));
                    } else {
                        let shrink_by = size_mm - world_mm;
                        return Err(format!(
                            "material '{}' is {:.2} mm wide on {}, larger than the {:.2} mm world extent. Suggested fix: shrink this material by at least {:.2} mm along {} and place its bounds at {:.2}..{:.2} mm (or enlarge world.dimensions_mm[{}]).",
                            material.name, size_mm, axis_name(axis), world_mm, shrink_by,
                            axis_name(axis), 0.0, world_mm, axis
                        ));
                    }
                }
                if (a - a.round()).abs() > 0.001 || (b - b.round()).abs() > 0.001 {
                    let mut snapped_min_cells = a.round() as u32;
                    let mut snapped_max_cells = b.round() as u32;
                    // Tiny off-grid boxes can round to zero voxels. Suggest a
                    // valid one-cell minimum instead of returning unusable bounds.
                    if snapped_max_cells <= snapped_min_cells {
                        snapped_min_cells = a.floor().max(0.0) as u32;
                        snapped_max_cells = (snapped_min_cells + 1).min(dims[axis]);
                    }
                    let snapped_min = snapped_min_cells as f32 * self.world.voxel_size_mm;
                    let snapped_max = snapped_max_cells as f32 * self.world.voxel_size_mm;
                    return Err(format!(
                        "material '{}' bounds on {} are not aligned to the {:.3} mm voxel grid. Suggested bounds: min={:.3} mm, max={:.3} mm (nearest grid positions, expanded to at least one voxel if necessary).",
                        material.name, axis_name(axis), self.world.voxel_size_mm, snapped_min, snapped_max
                    ));
                }
                min[axis] = a.round() as u32;
                max[axis] = b.round() as u32;
                if max[axis] <= min[axis] {
                    return Err(format!(
                        "material '{}' has no voxel cells on {} after grid alignment. Suggested fix: make its extent at least one voxel ({:.3} mm), for example bounds {:.3}..{:.3} mm.",
                        material.name, axis_name(axis), self.world.voxel_size_mm,
                        min[axis] as f32 * self.world.voxel_size_mm,
                        (min[axis] + 1).min(dims[axis]) as f32 * self.world.voxel_size_mm
                    ));
                }
                if max[axis] > dims[axis] {
                    return Err(format!("material {} exceeds world bounds", material.name));
                }
            }
            if !material.sigma_a_m_inv.is_finite()
                || !material.sigma_s_m_inv.is_finite()
                || !material.anisotropy_g.is_finite()
                || !material.sound_speed_mps.is_finite()
                || !material.density_kg_m3.is_finite()
                || !material.porosity.is_finite()
                || !material.visual_opacity.is_finite()
                || material.sigma_a_m_inv < 0.0
                || material.sigma_s_m_inv < 0.0
                || material.sigma_a_m_inv > 10_000.0
                || material.sigma_s_m_inv > 10_000.0
                || !(-0.99..=0.99).contains(&material.anisotropy_g)
                || !(40.0..=343.0).contains(&material.sound_speed_mps)
                || !(0.0..=1.0).contains(&material.porosity)
                || material.density_kg_m3 <= 0.0
                || !material.effective_density_kg_m3.is_finite()
                || !(0.1..=100.0).contains(&material.effective_density_kg_m3)
                || !(0.0..=1.0).contains(&material.visual_opacity)
                || material.visual_color.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
                || material.clump_strength < 0.0
                || material.clump_strength > 1.0
                || !material.clump_strength.is_finite()
                || !material.clump_scale_per_m.is_finite()
                || material.clump_scale_per_m <= 0.0
                || material.gradient_strength < 0.0
                || material.gradient_strength > 5.0
                || !material.gradient_strength.is_finite()
            {
                return Err(format!("material {} has invalid physical or visual properties", material.name));
            }
            if material.gradient_axis > 2 {
                return Err(format!("material {} gradient_axis must be 0, 1, or 2", material.name));
            }
            for (other_index, (other_min, other_max)) in bounds.iter().enumerate() {
                let overlaps = (0..3).all(|axis| min[axis] < other_max[axis] && max[axis] > other_min[axis]);
                if overlaps {
                    let other = &self.materials[other_index];
                    let mut moves: Vec<(f32, usize, f32)> = Vec::new();
                    for axis in 0..3 {
                        let current_min = material.bounds_min_mm[axis];
                        let current_max = material.bounds_max_mm[axis];
                        let other_min_mm = other_min[axis] as f32 * self.world.voxel_size_mm;
                        let other_max_mm = other_max[axis] as f32 * self.world.voxel_size_mm;
                        // Move the current region just far enough to touch the
                        // other region (half-open bounds permit face contact).
                        let negative_delta = other_min_mm - current_max;
                        let positive_delta = other_max_mm - current_min;
                        let world_extent = self.world.dimensions_mm[axis];
                        if current_min + negative_delta >= 0.0 {
                            moves.push((negative_delta.abs(), axis, negative_delta));
                        }
                        if current_max + positive_delta <= world_extent {
                            moves.push((positive_delta.abs(), axis, positive_delta));
                        }
                    }
                    moves.sort_by(|a, b| a.0.total_cmp(&b.0));
                    if let Some((distance, axis, delta)) = moves.first().copied() {
                        let sign = if delta < 0.0 { "-" } else { "+" };
                        return Err(format!(
                            "materials '{}' and '{}' overlap. Suggested fix: move '{}' by {:.2} mm along {}{}; new bounds on that axis are {:.2}..{:.2} mm and will just touch '{}'.",
                            other.name, material.name, material.name, distance, sign, axis_name(axis),
                            material.bounds_min_mm[axis] + delta, material.bounds_max_mm[axis] + delta, other.name
                        ));
                    }
                    return Err(format!(
                        "materials '{}' and '{}' overlap, and moving '{}' clear on one axis would take it outside the world. Shrink '{}' or enlarge world.dimensions_mm, then separate the blocks by at least one voxel.",
                        other.name, material.name, material.name, material.name
                    ));
                }
            }
            bounds.push((min, max));
        }

        if self.simulation.energy_time_bins == 0
            || !self.simulation.energy_time_bin_us.is_finite()
            || self.simulation.energy_time_bin_us <= 0.0
        {
            return Err("simulation.energy_time_bins and energy_time_bin_us must be greater than zero".into());
        }
        // Checked arithmetic only: no prototype memory budget. The actual
        // histogram size is compared with the adapter's binding/buffer limits,
        // then allocated with an out-of-memory error scope at device setup.
        let histogram_entries = voxel_count
            .checked_mul(self.simulation.energy_time_bins as u64)
            .ok_or_else(|| "energy-history element count overflows u64; reduce dimensions/bins or increase voxel_size_mm".to_string())?;
        histogram_entries
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| "energy-history byte size overflows u64; reduce dimensions/bins or increase voxel_size_mm".to_string())?;
        // WGSL currently addresses the flattened time histogram with u32 indices.
        // Allow exactly 2^32 entries (the final valid index is u32::MAX), but do
        // not let a larger GPU buffer silently wrap indices in the shader.
        if histogram_entries > u32::MAX as u64 + 1 {
            return Err("energy history exceeds the WGSL u32 flattened-index range; reduce world dimensions or energy_time_bins".into());
        }

        if self.simulation.ray_count == 0 {
            return Err("simulation.ray_count must be greater than zero".into());
        }
        if self.simulation.max_steps == 0 {
            return Err("simulation.max_steps must be greater than zero".into());
        }
        if self.simulation.step_mm <= 0.0
            || self.simulation.step_mm > self.world.voxel_size_mm
            || !self.simulation.step_mm.is_finite()
            || self.simulation.timeline_rate_us_per_second <= 0.0
            || !self.simulation.timeline_rate_us_per_second.is_finite()
        {
            return Err("simulation step must be finite, positive and no larger than voxel_size_mm; timeline rate must be finite and positive".into());
        }
        if !(0.1..=1.0).contains(&self.render.resolution_scale)
            || self.render.max_width == 0
            || self.render.max_height == 0
            || !(5..=60).contains(&self.render.target_fps)
        {
            return Err("render settings are outside their supported ranges".into());
        }
        for emitter in &self.emitters {
            if !emitter.radius_mm.is_finite()
                || !emitter.frequency_hz.is_finite()
                || !emitter.amplitude.is_finite()
                || emitter.radius_mm <= 0.0
                || !(1.0..=200_000.0).contains(&emitter.frequency_hz)
                || emitter.amplitude < 0.0
                || emitter.amplitude > 100.0
                || emitter.direction.iter().any(|v| !v.is_finite())
                || !emitter.direction.iter().map(|v| v * v).sum::<f32>().is_finite()
                || emitter.direction.iter().map(|v| v * v).sum::<f32>() < 1e-8
                || !emitter.cone_half_angle_degrees.is_finite()
                || !(1.0..=180.0).contains(&emitter.cone_half_angle_degrees)
            {
                return Err(format!("emitter '{}' has invalid radius, frequency, amplitude or directivity settings", emitter.name));
            }
            for axis in 0..3 {
                let world_extent = self.world.dimensions_mm[axis];
                let voxel = self.world.voxel_size_mm;
                let position = emitter.position_mm[axis];
                if !position.is_finite() {
                    return Err(format!("emitter '{}' has a non-finite {} position", emitter.name, axis_name(axis)));
                }
                let margin = emitter.radius_mm + voxel * 0.5;
                if world_extent <= margin * 2.0 {
                    return Err(format!(
                        "emitter '{}' cannot fit inside the world on {}: world extent {:.2} mm, required diameter plus voxel clearance {:.2} mm. Enlarge world.dimensions_mm[{}] or reduce source radius.",
                        emitter.name, axis_name(axis), world_extent, margin * 2.0, axis
                    ));
                }
                if position < margin || position > world_extent - margin {
                    let target = position.clamp(margin, world_extent - margin);
                    let delta = target - position;
                    let sign = if delta < 0.0 { "-" } else { "+" };
                    return Err(format!(
                        "emitter '{}' is too close to the world boundary on {}: center {:.2} mm, safe interval {:.2}..{:.2} mm. Suggested fix: move it {:.2} mm along {}{} to {:.2} mm so its {:.2} mm radius and half-voxel margin fit inside the world.",
                        emitter.name, axis_name(axis), position, margin, world_extent - margin,
                        delta.abs(), sign, axis_name(axis), target, emitter.radius_mm
                    ));
                }
            }
        }
        if !self.visualization.voxel_fade_ms.is_finite()
            || !self.visualization.wave_shell_thickness_mm.is_finite()
            || !self.visualization.felt_energy_gain.is_finite()
            || !self.visualization.air_wave_gain.is_finite()
            || !self.visualization.source_glow_gain.is_finite()
            || !self.visualization.felt_base_opacity_multiplier.is_finite()
            || !self.visualization.felt_energy_max_opacity.is_finite()
            || !self.visualization.air_wave_max_opacity.is_finite()
            || self.visualization.energy_hot_color.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            || self.visualization.voxel_fade_ms <= 0.0
            || self.visualization.wave_shell_thickness_mm <= 0.0
            || self.visualization.felt_energy_gain < 0.0
            || self.visualization.air_wave_gain < 0.0
            || self.visualization.source_glow_gain < 0.0
            || self.visualization.felt_base_opacity_multiplier < 0.0
            || !(0.0..=1.0).contains(&self.visualization.felt_energy_max_opacity)
            || !(0.0..=1.0).contains(&self.visualization.air_wave_max_opacity)
        {
            return Err("visualization fade, shell thickness and gains are invalid".into());
        }
        if self.camera.target_mm.iter().any(|v| !v.is_finite())
            || !self.camera.yaw_radians.is_finite()
            || !self.camera.pitch_radians.is_finite()
            || !self.camera.distance_mm.is_finite()
            || !self.camera.min_distance_mm.is_finite()
            || !self.camera.max_distance_mm.is_finite()
            || !self.camera.movement_speed_mps.is_finite()
            || self.camera.distance_mm <= 0.0
            || self.camera.min_distance_mm <= 0.0
            || self.camera.max_distance_mm < self.camera.min_distance_mm
            || self.camera.distance_mm < self.camera.min_distance_mm
            || self.camera.distance_mm > self.camera.max_distance_mm
            || self.camera.movement_speed_mps <= 0.0
        {
            return Err("camera distances and movement speed are invalid".into());
        }

        for emitter in &self.emitters {
            for material in &self.materials {
                let mut gaps = [0.0f32; 3];
                for axis in 0..3 {
                    let p = emitter.position_mm[axis];
                    let min = material.bounds_min_mm[axis];
                    let max = material.bounds_max_mm[axis];
                    gaps[axis] = if p < min { min - p } else if p > max { p - max } else { 0.0 };
                }
                let distance_to_region = gaps.iter().map(|gap| gap * gap).sum::<f32>().sqrt();
                let clearance = emitter.radius_mm + self.world.voxel_size_mm;
                if distance_to_region + 1e-4 < clearance {
                    let mut suggestions: Vec<(f32, usize, f32)> = Vec::new();
                    for axis in 0..3 {
                        let other_gap_sq = (0..3)
                            .filter(|other_axis| *other_axis != axis)
                            .map(|other_axis| gaps[other_axis] * gaps[other_axis])
                            .sum::<f32>();
                        let required_gap = (clearance * clearance - other_gap_sq).max(0.0).sqrt();
                        let position = emitter.position_mm[axis];
                        let min = material.bounds_min_mm[axis];
                        let max = material.bounds_max_mm[axis];
                        let mut targets = Vec::with_capacity(2);
                        if position <= min {
                            targets.push(min - required_gap);
                        } else if position >= max {
                            targets.push(max + required_gap);
                        } else {
                            targets.push(min - required_gap);
                            targets.push(max + required_gap);
                        }
                        let world_extent = self.world.dimensions_mm[axis];
                        let margin = emitter.radius_mm + self.world.voxel_size_mm * 0.5;
                        for target in targets {
                            if target >= margin && target <= world_extent - margin {
                                let delta = target - position;
                                suggestions.push((delta.abs(), axis, delta));
                            }
                        }
                    }
                    suggestions.sort_by(|a, b| a.0.total_cmp(&b.0));
                    if let Some((distance, axis, delta)) = suggestions.first().copied() {
                        let new_position = emitter.position_mm[axis] + delta;
                        let sign = if delta < 0.0 { "-" } else { "+" };
                        return Err(format!(
                            "emitter '{}' is too close to material '{}': distance to its AABB is {:.2} mm, required clearance is {:.2} mm (source radius + one voxel). Suggested fix: move it {:.2} mm along {}{} (axis position {:.2} -> {:.2} mm).",
                            emitter.name, material.name, distance_to_region, clearance, distance, sign, axis_name(axis),
                            emitter.position_mm[axis], new_position
                        ));
                    }
                    return Err(format!(
                        "emitter '{}' is too close to material '{}', and no single-axis clearance move fits inside the current world. Move or shrink the material, move the emitter in multiple axes, or enlarge world.dimensions_mm.",
                        emitter.name, material.name
                    ));
                }
            }
        }

        Ok(dims)
    }
}

fn axis_name(axis: usize) -> &'static str {
    match axis { 0 => "X", 1 => "Y", _ => "Z" }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;

    #[test]
    fn built_in_config_is_valid() {
        let config = AppConfig::default();
        assert_eq!(config.validate().unwrap(), [128, 96, 96]);
    }

    #[test]
    fn world_size_is_not_rejected_by_a_prototype_memory_budget() {
        let mut config = AppConfig::default();
        config.world.dimensions_mm = [512.0, 512.0, 512.0];
        assert_eq!(config.validate().unwrap(), [512, 512, 512]);
    }


    #[test]
    fn energy_history_rejects_wgsl_flat_index_overflow() {
        let mut config = AppConfig::default();
        config.simulation.energy_time_bins = u32::MAX;
        let error = config.validate().unwrap_err();
        assert!(error.contains("WGSL u32 flattened-index range"));
    }

    #[test]
    fn defaults_use_a_dense_packet_field_and_limited_scattering() {
        let config = AppConfig::default();
        assert_eq!(config.simulation.ray_count, 131_072);
        assert_eq!(config.materials[0].sigma_a_m_inv, 60.0);
        assert_eq!(config.materials[0].sigma_s_m_inv, 8.0);
    }

    #[test]
    fn visual_energy_alpha_and_color_settings_are_validated() {
        let mut config = AppConfig::default();
        config.visualization.air_wave_max_opacity = 1.1;
        assert!(config.validate().unwrap_err().contains("visualization"));

        let mut config = AppConfig::default();
        config.visualization.energy_hot_color[1] = f32::NAN;
        assert!(config.validate().unwrap_err().contains("visualization"));
    }

    #[test]
    fn overlap_error_suggests_a_direction_and_distance() {
        let mut config = AppConfig::default();
        config.materials[0].bounds_max_mm[0] = 100.0;
        let mut second = config.materials[0].clone();
        second.name = "second".into();
        second.bounds_min_mm[0] = 90.0;
        second.bounds_max_mm[0] = 128.0;
        let err = {
            config.materials.push(second);
            config.validate().unwrap_err()
        };
        assert!(err.contains("move 'second'"));
        assert!(err.contains("mm along"));
    }

    #[test]
    fn tiny_off_grid_material_suggestion_keeps_at_least_one_voxel() {
        let mut config = AppConfig::default();
        config.materials[0].bounds_min_mm[0] = 64.2;
        config.materials[0].bounds_max_mm[0] = 64.4;
        let err = config.validate().unwrap_err();
        assert!(err.contains("expanded to at least one voxel"));
        assert!(err.contains("max=65.000 mm"));
    }

    #[test]
    fn outside_material_error_suggests_exact_translation() {
        let mut config = AppConfig::default();
        config.materials[0].bounds_min_mm[0] = 118.0;
        config.materials[0].bounds_max_mm[0] = 138.0;
        let err = config.validate().unwrap_err();
        assert!(err.contains("move this material 10.00 mm along X-"));
        assert!(err.contains("new bounds 108.00..128.00 mm"));
    }

    #[test]
    fn source_inside_material_error_suggests_a_clearance_move() {
        let mut config = AppConfig::default();
        config.emitters[0].position_mm = [80.0, 48.0, 48.0];
        let err = config.validate().unwrap_err();
        assert!(err.contains("Suggested fix"));
        assert!(err.contains("one voxel"));
    }

    #[test]
    fn multiple_emitters_validate_and_round_trip_through_toml() {
        let mut config = AppConfig::default();
        let mut second = config.emitters[0].clone();
        second.name = "upper-emitter".into();
        second.position_mm = [18.0, 24.0, 48.0];
        second.frequency_hz = 8_000.0;
        second.amplitude = 0.35;
        config.emitters.push(second);

        assert_eq!(config.validate().unwrap(), [128, 96, 96]);
        let serialized = toml::to_string_pretty(&config).unwrap();
        assert!(serialized.contains("[[emitters]]"));
        let decoded: AppConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(decoded.emitters.len(), 2);
        assert_eq!(decoded.emitters[1].name, "upper-emitter");
        assert_eq!(decoded.validate().unwrap(), [128, 96, 96]);
    }

    #[test]
    fn emitter_near_material_boundary_gets_clearance_suggestion() {
        let mut config = AppConfig::default();
        config.emitters[0].position_mm[0] = 60.0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("too close to material 'warm-felt'"));
        assert!(error.contains("Suggested fix"));
        assert!(error.contains("X"));
    }

    #[test]
    fn emitter_names_must_be_unique() {
        let mut config = AppConfig::default();
        let mut second = config.emitters[0].clone();
        second.position_mm = [18.0, 24.0, 48.0];
        config.emitters.push(second);
        assert!(config.validate().unwrap_err().contains("is duplicated"));
    }

    #[test]
    fn adjacent_serial_material_regions_are_valid() {
        let mut config = AppConfig::default();
        let mut first = config.materials[0].clone();
        first.name = "felt-a".into();
        first.bounds_max_mm[0] = 96.0;

        let mut second = first.clone();
        second.name = "felt-b".into();
        second.bounds_min_mm[0] = 96.0;
        second.bounds_max_mm[0] = 128.0;
        second.sound_speed_mps = 190.0;
        second.sigma_a_m_inv = 28.0;

        config.materials = vec![first, second];
        assert_eq!(config.validate().unwrap(), [128, 96, 96]);
    }
}
