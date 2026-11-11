use crate::MediumLaw;
use val_core::{Rng, Vec3};

pub struct GradientDensity {
    pub axis: usize,
    pub strength: f32,
    /// Coordinate at which the gradient multiplier is neutral, in local metres.
    pub center: f32,
}
pub struct ClumpNoise {
    pub scale: f32,
    pub strength: f32,
}

fn noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let n = (x * 127.1 + y * 311.7 + z * 74.7).sin() * 43758.5453;
    n - n.floor()
}

impl MediumLaw for GradientDensity {
    fn name(&self) -> String {
        "gradient".into()
    }
    fn apply(
        &self,
        p: Vec3,
        sa: &mut f32,
        ss: &mut f32,
        _g: &mut f32,
        c: &mut f32,
        _rng: &mut Rng,
    ) {
        let coord = match self.axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        };
        let mult = (1.0 + self.strength * (coord - self.center)).max(0.1);
        *sa *= mult;
        *ss *= mult;
        *c /= mult.sqrt(); // Experimental denser-is-slower approximation.
    }
}

impl MediumLaw for ClumpNoise {
    fn name(&self) -> String {
        "clump".into()
    }
    fn apply(
        &self,
        p: Vec3,
        sa: &mut f32,
        ss: &mut f32,
        _g: &mut f32,
        c: &mut f32,
        _rng: &mut Rng,
    ) {
        // Strength now controls the perturbation; zero means a uniform field.
        let n = noise_3d(p.x * self.scale, p.y * self.scale, p.z * self.scale);
        let mult = (1.0 + self.strength * (2.0 * n - 1.0)).max(0.1);
        *sa *= mult;
        *ss *= mult;
        *c /= mult.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::{GradientDensity, MediumLaw};
    use val_core::{new_rng, Vec3};

    #[test]
    fn gradient_is_neutral_at_the_region_local_center() {
        let law = GradientDensity {
            axis: 1,
            strength: 0.8,
            center: 0.048,
        };
        let mut absorption = 10.0;
        let mut scattering = 4.0;
        let mut anisotropy = 0.3;
        let mut speed = 220.0;
        let mut rng = new_rng(7);

        law.apply(
            Vec3::new(0.0, 0.048, 0.0),
            &mut absorption,
            &mut scattering,
            &mut anisotropy,
            &mut speed,
            &mut rng,
        );

        assert!((absorption - 10.0).abs() < 1e-6);
        assert!((scattering - 4.0).abs() < 1e-6);
        assert!((speed - 220.0).abs() < 1e-6);
    }
}
