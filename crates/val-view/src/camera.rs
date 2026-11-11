use val_core::{config::CameraConfig, Vec3};

/// Camera state is independent of wgpu and can be reconstructed from runtime configuration.
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.064, 0.048, 0.048),
            yaw: -0.55,
            pitch: 0.35,
            distance: 0.24,
            min_distance: 0.04,
            max_distance: 0.75,
        }
    }
}

impl OrbitCamera {
    pub fn from_config(config: &CameraConfig) -> Self {
        Self {
            target: Vec3::new(
                config.target_mm[0] * 0.001,
                config.target_mm[1] * 0.001,
                config.target_mm[2] * 0.001,
            ),
            yaw: config.yaw_radians,
            pitch: config.pitch_radians.clamp(-1.5, 1.5),
            distance: config.distance_mm * 0.001,
            min_distance: config.min_distance_mm * 0.001,
            max_distance: config.max_distance_mm * 0.001,
        }
    }

    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.yaw += dx as f32 * 0.005;
        self.pitch = (self.pitch + dy as f32 * 0.005).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance - amount).clamp(self.min_distance, self.max_distance);
    }

    pub fn move_local(&mut self, forward_amount: f32, right_amount: f32, up_amount: f32) {
        let (forward, right, up) = self.basis();
        self.target += forward * forward_amount + right * right_amount + up * up_amount;
    }

    pub fn position(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.pitch.cos() * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * self.pitch.cos() * self.yaw.cos(),
            )
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.target - self.position()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (forward, right, up)
    }
}
