//! Reference equations for the current fluid-equivalent Monte Carlo transport model.
//!
//! These functions are deliberately kept outside rendering code so the model can
//! be experimented with or replaced. The GPU transport shader mirrors these
//! equations for throughput; unit tests here provide a CPU-side reference.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfacePower {
    pub reflected: f32,
    pub transmitted: f32,
    pub refracted_cosine: Option<f32>,
}

/// Approximate power coefficients for a plane acoustic wave at a fluid/fluid
/// boundary. `cos_incidence` is measured relative to the normal pointing from
/// medium A toward medium B. This is not a poroelastic interface model.
pub fn fluid_interface_power(
    sound_speed_a: f32,
    density_a: f32,
    sound_speed_b: f32,
    density_b: f32,
    cos_incidence: f32,
) -> InterfacePower {
    let c_a = sound_speed_a.max(1.0);
    let c_b = sound_speed_b.max(1.0);
    let rho_a = density_a.max(0.001);
    let rho_b = density_b.max(0.001);
    let cos_i = cos_incidence.clamp(0.0, 1.0);
    let eta = c_b / c_a;
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

    if sin2_t >= 1.0 {
        return InterfacePower {
            reflected: 1.0,
            transmitted: 0.0,
            refracted_cosine: None,
        };
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    let z_a = rho_a * c_a;
    let z_b = rho_b * c_b;
    let denominator = (z_b * cos_i + z_a * cos_t).max(0.001);
    let pressure_reflection = (z_b * cos_i - z_a * cos_t) / denominator;
    let reflected = (pressure_reflection * pressure_reflection).clamp(0.0, 1.0);

    InterfacePower {
        reflected,
        transmitted: 1.0 - reflected,
        refracted_cosine: Some(cos_t),
    }
}

/// Beer-Lambert energy attenuation over a homogeneous segment.
/// `sigma_a_m_inv` is an energy attenuation coefficient, not a pressure
/// amplitude coefficient.
pub fn attenuate_energy(energy: f32, sigma_a_m_inv: f32, distance_m: f32) -> f32 {
    energy.max(0.0) * (-sigma_a_m_inv.max(0.0) * distance_m.max(0.0)).exp()
}

/// One-way travel time through a homogeneous segment, in microseconds.
pub fn travel_time_us(distance_m: f32, sound_speed_mps: f32) -> f32 {
    distance_m.max(0.0) / sound_speed_mps.max(1.0) * 1_000_000.0
}

/// One homogeneous section of a piecewise air/material propagation path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSegment {
    pub distance_m: f32,
    pub sound_speed_mps: f32,
    pub sigma_a_m_inv: f32,
}

/// Result of integrating the direct, serial part of a packet path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathState {
    pub travel_time_us: f32,
    pub remaining_energy: f32,
}

/// Integrate consecutive media in path order. Interfaces are handled separately
/// by `fluid_interface_power`; this helper covers in-medium time and absorption.
pub fn integrate_serial_path(initial_energy: f32, segments: &[PathSegment]) -> PathState {
    let mut state = PathState {
        travel_time_us: 0.0,
        remaining_energy: initial_energy.max(0.0),
    };
    for segment in segments {
        state.travel_time_us += travel_time_us(segment.distance_m, segment.sound_speed_mps);
        state.remaining_energy = attenuate_energy(
            state.remaining_energy,
            segment.sigma_a_m_inv,
            segment.distance_m,
        );
    }
    state
}

#[cfg(test)]
mod tests {
    use super::{
        attenuate_energy, fluid_interface_power, integrate_serial_path, travel_time_us, PathSegment,
    };

    #[test]
    fn serial_material_segments_add_their_travel_times() {
        let air_before = travel_time_us(0.046, 343.0);
        let felt = travel_time_us(0.064, 220.0);
        let air_after = travel_time_us(0.020, 343.0);
        let total = air_before + felt + air_after;
        assert!((total - 483.3).abs() < 1.0);
        assert!(total > travel_time_us(0.130, 343.0));
    }

    #[test]
    fn second_layer_cannot_be_reached_before_serial_first_layer_traversal() {
        let air_before = PathSegment { distance_m: 0.046, sound_speed_mps: 343.0, sigma_a_m_inv: 0.0 };
        let first = PathSegment { distance_m: 0.032, sound_speed_mps: 220.0, sigma_a_m_inv: 10.0 };
        let air_gap = PathSegment { distance_m: 0.004, sound_speed_mps: 343.0, sigma_a_m_inv: 0.0 };
        let second = PathSegment { distance_m: 0.032, sound_speed_mps: 190.0, sigma_a_m_inv: 28.0 };

        let at_first_exit = integrate_serial_path(1.0, &[air_before, first]);
        let at_second_entry = integrate_serial_path(1.0, &[air_before, first, air_gap]);
        let through_second = integrate_serial_path(1.0, &[air_before, first, air_gap, second]);

        assert!(at_second_entry.travel_time_us > at_first_exit.travel_time_us);
        assert!((at_second_entry.travel_time_us - at_first_exit.travel_time_us
            - travel_time_us(air_gap.distance_m, air_gap.sound_speed_mps)).abs() < 0.001);
        assert!(through_second.travel_time_us > at_second_entry.travel_time_us);
        assert!(through_second.remaining_energy < at_second_entry.remaining_energy);
    }

    #[test]
    fn absorption_accumulates_through_serial_layers() {
        let start = 1.0;
        let after_a = attenuate_energy(start, 12.0, 0.032);
        let after_gap = attenuate_energy(after_a, 0.0, 0.010);
        let after_b = attenuate_energy(after_gap, 28.0, 0.032);
        assert!(after_b < after_a);
        assert!((after_b - (-1.28_f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn interface_power_is_conservative_without_absorption() {
        let coeff = fluid_interface_power(343.0, 1.204, 220.0, 1.8, 1.0);
        assert!(coeff.reflected >= 0.0 && coeff.reflected <= 1.0);
        assert!(coeff.transmitted >= 0.0 && coeff.transmitted <= 1.0);
        assert!((coeff.reflected + coeff.transmitted - 1.0).abs() < 1e-6);
        assert!(coeff.refracted_cosine.is_some());
    }

    #[test]
    fn total_internal_reflection_has_no_transmitted_branch() {
        let coeff = fluid_interface_power(220.0, 1.8, 343.0, 1.204, 0.2);
        assert_eq!(coeff.reflected, 1.0);
        assert_eq!(coeff.transmitted, 0.0);
        assert_eq!(coeff.refracted_cosine, None);
    }
}
