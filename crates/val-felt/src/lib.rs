use val_core::{new_rng, Rng, Vec3};

pub mod laws;
pub use laws::*;

fn try_filled_vec<T: Clone>(len: usize, value: T, label: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        format!("unable to allocate {len} elements for {label}: {error}")
    })?;
    values.resize(len, value);
    Ok(values)
}

fn checked_voxel_count(dim: [u32; 3], label: &str) -> Result<usize, String> {
    let count = (dim[0] as u64)
        .checked_mul(dim[1] as u64)
        .and_then(|v| v.checked_mul(dim[2] as u64))
        .ok_or_else(|| format!("{label} voxel count overflows u64"))?;
    usize::try_from(count).map_err(|_| format!("{label} voxel count does not fit host address space"))
}

#[derive(Clone, Copy, Debug)]
pub struct MediumProperties {
    pub sigma_a: f32,
    pub sigma_s: f32,
    pub anisotropy_g: f32,
    pub sound_speed_mps: f32,
}

impl Default for MediumProperties {
    fn default() -> Self {
        Self {
            sigma_a: 60.0,
            sigma_s: 8.0,
            anisotropy_g: 0.55,
            sound_speed_mps: 220.0,
        }
    }
}

/// One non-overlapping felt AABB in world voxel coordinates (half-open bounds).
#[derive(Clone, Debug)]
pub struct MaterialRegion {
    pub name: String,
    pub id: u32,
    pub min: [u32; 3],
    pub max: [u32; 3],
    /// RGB in linear space and per-voxel base opacity in alpha.
    pub visual: [f32; 4],
    /// Bulk material density (kg/m^3), retained as physical metadata.
    pub density_kg_m3: f32,
    /// Effective dynamic density used for the approximate acoustic impedance.
    pub effective_density_kg_m3: f32,
    pub porosity: f32,
    /// Region-average coefficients retained for diagnostics and material summaries.
    pub sigma_a_mean: f32,
    pub sigma_s_mean: f32,
    pub sound_speed_mean: f32,
}

/// World-space voxel material field. Cells outside all regions are air.
pub struct VoxelMedium {
    pub dim: [u32; 3],
    pub spacing: f32,
    pub sigma_a: Vec<f32>,
    pub sigma_s: Vec<f32>,
    pub g: Vec<f32>,
    pub c: Vec<f32>,
    pub material_regions: Vec<MaterialRegion>,
}

pub trait MediumLaw: Send + Sync {
    fn name(&self) -> String;
    fn apply(&self, p: Vec3, sa: &mut f32, ss: &mut f32, g: &mut f32, c: &mut f32, rng: &mut Rng);
}

pub struct FeltVolume {
    pub name: String,
    pub origin: [u32; 3],
    pub medium: VoxelMedium,
    pub visual: [f32; 4],
    pub density_kg_m3: f32,
    pub effective_density_kg_m3: f32,
    pub porosity: f32,
}

pub fn generate_medium(
    dim: [u32; 3],
    spacing: f32,
    seed: u64,
    laws: &[Box<dyn MediumLaw>],
) -> Result<VoxelMedium, String> {
    generate_medium_with_properties(dim, spacing, seed, MediumProperties::default(), laws)
}

pub fn generate_medium_with_properties(
    dim: [u32; 3],
    spacing: f32,
    seed: u64,
    properties: MediumProperties,
    laws: &[Box<dyn MediumLaw>],
) -> Result<VoxelMedium, String> {
    let n = checked_voxel_count(dim, "material")?;
    let mut sigma_a = try_filled_vec(n, 0.0, "material absorption field")?;
    let mut sigma_s = try_filled_vec(n, 0.0, "material scattering field")?;
    let mut g = try_filled_vec(n, 0.0, "material anisotropy field")?;
    let mut c = try_filled_vec(n, 0.0, "material sound-speed field")?;
    let mut rng = new_rng(seed);

    let dx = dim[0] as usize;
    let dy = dim[1] as usize;
    for z in 0..dim[2] as usize {
        for y in 0..dim[1] as usize {
            for x in 0..dim[0] as usize {
                let i = z * dy * dx + y * dx + x;
                let p = Vec3::new(x as f32 * spacing, y as f32 * spacing, z as f32 * spacing);
                let (mut sa, mut ss, mut gg, mut cc) = (
                    properties.sigma_a,
                    properties.sigma_s,
                    properties.anisotropy_g,
                    properties.sound_speed_mps,
                );
                for law in laws {
                    law.apply(p, &mut sa, &mut ss, &mut gg, &mut cc, &mut rng);
                }
                sigma_a[i] = sa.max(0.0);
                sigma_s[i] = ss.max(0.0);
                g[i] = gg.clamp(-0.99, 0.99);
                c[i] = cc.clamp(40.0, 343.0);
            }
        }
    }

    Ok(VoxelMedium {
        dim,
        spacing,
        sigma_a,
        sigma_s,
        g,
        c,
        material_regions: Vec::new(),
    })
}

/// Combines independently generated felt volumes into a common air/felt grid.
/// Overlapping regions are rejected so material ownership is deterministic.
pub fn compose_world(
    world_dim: [u32; 3],
    spacing: f32,
    air_speed: f32,
    volumes: Vec<FeltVolume>,
) -> Result<VoxelMedium, String> {
    if world_dim.iter().any(|v| *v == 0) || !spacing.is_finite() || spacing <= 0.0 {
        return Err("world dimensions and voxel spacing must be positive".into());
    }
    let count = checked_voxel_count(world_dim, "world")?;
    let mut sigma_a = try_filled_vec(count, 0.0, "world absorption field")?;
    let mut sigma_s = try_filled_vec(count, 0.0, "world scattering field")?;
    let mut g = try_filled_vec(count, 0.0, "world anisotropy field")?;
    let mut c = try_filled_vec(count, air_speed, "world sound-speed field")?;
    let mut occupied = try_filled_vec(count, false, "world material occupancy field")?;
    let mut material_regions = Vec::new();
    material_regions.try_reserve_exact(volumes.len())
        .map_err(|error| format!("unable to reserve material-region metadata: {error}"))?;

    for (volume_index, volume) in volumes.into_iter().enumerate() {
        if (volume.medium.spacing - spacing).abs() > 1e-7 {
            return Err(format!("material {} uses a different voxel spacing", volume.name));
        }
        let mut max = [0u32; 3];
        for axis in 0..3 {
            max[axis] = volume.origin[axis]
                .checked_add(volume.medium.dim[axis])
                .ok_or_else(|| format!("material {} bound overflows voxel coordinates", volume.name))?;
            if max[axis] > world_dim[axis] {
                return Err(format!("material {} exceeds world bounds on {}: end {} voxels, world {} voxels; move or shrink the material", volume.name, ["X", "Y", "Z"][axis], max[axis], world_dim[axis]));
            }
        }
        let id = u32::try_from(volume_index + 1)
            .map_err(|_| "too many material regions for u32 material IDs".to_string())?;

        let local_x = volume.medium.dim[0] as usize;
        let local_y = volume.medium.dim[1] as usize;
        let world_x = world_dim[0] as usize;
        let world_y = world_dim[1] as usize;
        for z in 0..volume.medium.dim[2] as usize {
            for y in 0..volume.medium.dim[1] as usize {
                for x in 0..volume.medium.dim[0] as usize {
                    let dst_x = x + volume.origin[0] as usize;
                    let dst_y = y + volume.origin[1] as usize;
                    let dst_z = z + volume.origin[2] as usize;
                    let src_i = z * local_y * local_x + y * local_x + x;
                    let dst_i = dst_z * world_y * world_x + dst_y * world_x + dst_x;
                    if occupied[dst_i] {
                        return Err(format!("material {} overlaps another felt region", volume.name));
                    }
                    occupied[dst_i] = true;
                    sigma_a[dst_i] = volume.medium.sigma_a[src_i];
                    sigma_s[dst_i] = volume.medium.sigma_s[src_i];
                    g[dst_i] = volume.medium.g[src_i];
                    c[dst_i] = volume.medium.c[src_i];
                }
            }
        }

        let local_count = volume.medium.sigma_a.len().max(1) as f32;
        let sigma_a_mean = volume.medium.sigma_a.iter().sum::<f32>() / local_count;
        let sigma_s_mean = volume.medium.sigma_s.iter().sum::<f32>() / local_count;
        let sound_speed_mean = volume.medium.c.iter().sum::<f32>() / local_count;
        material_regions.push(MaterialRegion {
            name: volume.name,
            id,
            min: volume.origin,
            max,
            visual: volume.visual,
            density_kg_m3: volume.density_kg_m3,
            effective_density_kg_m3: volume.effective_density_kg_m3,
            porosity: volume.porosity,
            sigma_a_mean,
            sigma_s_mean,
            sound_speed_mean,
        });
    }

    Ok(VoxelMedium {
        dim: world_dim,
        spacing,
        sigma_a,
        sigma_s,
        g,
        c,
        material_regions,
    })
}

#[cfg(test)]
mod tests {
    use super::{compose_world, generate_medium, FeltVolume};

    fn volume(name: &str, origin: [u32; 3], seed: u64) -> FeltVolume {
        FeltVolume {
            name: name.into(),
            origin,
            medium: generate_medium([2, 2, 2], 0.001, seed, &[]).unwrap(),
            visual: [0.9, 0.3, 0.1, 0.03],
            density_kg_m3: 180.0,
            effective_density_kg_m3: 1.8,
            porosity: 0.82,
        }
    }

    #[test]
    fn composes_multiple_non_overlapping_felt_instances() {
        let world = compose_world(
            [8, 4, 4],
            0.001,
            343.0,
            vec![volume("left", [1, 1, 1], 7), volume("right", [5, 1, 1], 9)],
        ).unwrap();
        assert_eq!(world.material_regions.len(), 2);
        assert_eq!(world.material_regions[1].id, 2);
        assert_eq!(world.c[0], 343.0);
        assert!(world.sigma_a[(1 * 4 * 8 + 1 * 8 + 1) as usize] > 0.0);
    }

    #[test]
    fn rejects_overlapping_material_instances() {
        let result = compose_world(
            [4, 4, 4],
            0.001,
            343.0,
            vec![volume("a", [1, 1, 1], 1), volume("b", [2, 1, 1], 2)],
        );
        assert!(result.is_err());
    }
}
