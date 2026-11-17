//! Frequency-resolved direct-path transport and virtual listener capture.
//!
//! This is a frequency-domain, non-coherent transport approximation. It applies
//! frequency-dependent material attenuation, interface transmission and phase
//! delay along the source-to-listener path. It is not a 3-D pressure FDTD solver:
//! diffraction, modal behaviour and diffuse multipath are not reconstructed.

use val_core::acoustics::fluid_interface_power;
use val_core::audio::AudioClip;
use std::f64::consts::PI;
use std::path::Path;
use val_felt::VoxelMedium;

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn magnitude_squared(self) -> f64 { self.re * self.re + self.im * self.im }
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
    fn scale(self, scale: f64) -> Self { Self::new(self.re * scale, self.im * scale) }
}

fn fft(values: &mut [Complex], inverse: bool) -> Result<(), String> {
    let n = values.len();
    if n < 2 || !n.is_power_of_two() {
        return Err("FFT buffer length must be a power of two of at least 2".into());
    }

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }

    let mut width = 2usize;
    while width <= n {
        let angle = (if inverse { 2.0 } else { -2.0 }) * PI / width as f64;
        let root = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(width) {
            let mut twiddle = Complex::new(1.0, 0.0);
            let half = width / 2;
            for offset in 0..half {
                let even = values[start + offset];
                let odd = values[start + offset + half].mul(twiddle);
                values[start + offset] = Complex::new(even.re + odd.re, even.im + odd.im);
                values[start + offset + half] = Complex::new(even.re - odd.re, even.im - odd.im);
                twiddle = twiddle.mul(root);
            }
        }
        width = width.checked_mul(2).ok_or_else(|| "FFT stage width overflow".to_string())?;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for value in values { *value = value.scale(scale); }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ListenerSpectrum {
    pub frequency_hz: Vec<f32>,
    /// Single-sided mean-square amplitude contribution per frequency bin.
    pub input_power: Vec<f32>,
    /// Single-sided mean-square amplitude contribution captured by the listener.
    pub received_power: Vec<f32>,
    /// Power lost between source and listener, clamped to non-negative values.
    pub lost_power: Vec<f32>,
    /// Received/input power ratio in dB.
    pub transmission_db: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ListenerCapture {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub travel_time_seconds: f32,
    pub total_input_power: f32,
    pub total_received_power: f32,
    pub total_lost_power: f32,
    pub spectrum: ListenerSpectrum,
}

#[derive(Clone, Copy, Debug)]
pub struct ListenerModel {
    pub position_mm: [f32; 3],
    /// Points from the listener face back toward the incoming source.
    pub normal: [f32; 3],
    pub aperture_radius_mm: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct WorldAirModel {
    pub sound_speed_mps: f32,
    pub density_kg_m3: f32,
    pub absorption_m_inv: f32,
    pub absorption_frequency_exponent: f32,
}

#[derive(Clone, Debug)]
struct PathLossTerm {
    region_index: Option<usize>,
    absorption_integral: f64,
    scattering_integral: f64,
    absorption_exponent: f64,
    scattering_exponent: f64,
}

#[derive(Clone, Debug)]
struct PathProfile {
    interface_transmission: f64,
    travel_seconds: f64,
    terms: Vec<PathLossTerm>,
}

impl PathProfile {
    fn energy_transmission(&self, frequency_hz: f64) -> f64 {
        let ratio = (frequency_hz.max(1.0) / 1000.0).clamp(0.001, 1000.0);
        let mut exponent = 0.0f64;
        for term in &self.terms {
            exponent += term.absorption_integral * ratio.powf(term.absorption_exponent);
            exponent += term.scattering_integral * ratio.powf(term.scattering_exponent);
        }
        (self.interface_transmission * (-exponent).exp()).clamp(0.0, 1.0)
    }
}

fn region_at(medium: &VoxelMedium, cell: [u32; 3]) -> Option<usize> {
    medium.material_regions.iter().position(|region| {
        (0..3).all(|axis| cell[axis] >= region.min[axis] && cell[axis] < region.max[axis])
    })
}

fn trace_path_profile(
    source_mm: [f32; 3],
    listener: ListenerModel,
    medium: &VoxelMedium,
    air: WorldAirModel,
) -> Result<PathProfile, String> {
    let source = source_mm.map(|v| v as f64 * 0.001);
    let destination = listener.position_mm.map(|v| v as f64 * 0.001);
    let delta = [destination[0] - source[0], destination[1] - source[1], destination[2] - source[2]];
    let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if !distance.is_finite() || distance <= 1e-6 {
        return Err("source and listener positions must be distinct".into());
    }
    let direction = [delta[0] / distance, delta[1] / distance, delta[2] / distance];
    let normal = listener.normal.map(|v| v as f64);
    let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if normal_len <= 1e-6 {
        return Err("listener normal cannot be zero".into());
    }
    let facing = (direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]) / normal_len;
    if facing > -0.05 {
        return Err("listener normal must face the source; invert listener.normal or move the listener behind the test volume".into());
    }
    if !listener.aperture_radius_mm.is_finite() || listener.aperture_radius_mm <= 0.0 {
        return Err("listener aperture radius must be finite and positive".into());
    }

    let world_max = [
        medium.dim[0] as f64 * medium.spacing as f64,
        medium.dim[1] as f64 * medium.spacing as f64,
        medium.dim[2] as f64 * medium.spacing as f64,
    ];
    for axis in 0..3 {
        if source[axis] < 0.0 || source[axis] >= world_max[axis]
            || destination[axis] < 0.0 || destination[axis] >= world_max[axis]
        {
            return Err("source and listener must be located inside the simulated world".into());
        }
    }

    let aperture_area = PI * (listener.aperture_radius_mm as f64 * 0.001).powi(2);
    // Isotropic point-source spreading, integrated over the listener aperture.
    // It is a received-power fraction, not a calibrated source SPL estimate.
    let geometric_fraction = (aperture_area / (4.0 * PI * distance * distance)).clamp(0.0, 1.0);

    let step = (medium.spacing as f64 * 0.5).max(0.00005);
    let segments = (distance / step).ceil().max(1.0) as usize;
    if segments > 20_000_000 {
        return Err("source-to-listener path is too long for the CPU spectral pass".into());
    }

    let mut interface_transmission = geometric_fraction;
    let mut travel_seconds = 0.0f64;
    let mut previous_region: Option<usize> = None;
    let mut previous_speed = air.sound_speed_mps.max(1.0) as f64;
    let mut previous_density = air.density_kg_m3.max(0.001) as f64;
    let mut terms: Vec<PathLossTerm> = Vec::new();

    for segment in 0..segments {
        let t0 = segment as f64 / segments as f64;
        let t1 = (segment + 1) as f64 / segments as f64;
        let tm = (t0 + t1) * 0.5;
        let position = [
            source[0] + direction[0] * distance * tm,
            source[1] + direction[1] * distance * tm,
            source[2] + direction[2] * distance * tm,
        ];
        let cell = [
            (position[0] / medium.spacing as f64).floor() as u32,
            (position[1] / medium.spacing as f64).floor() as u32,
            (position[2] / medium.spacing as f64).floor() as u32,
        ];
        let cell_index = cell[2] as usize * medium.dim[1] as usize * medium.dim[0] as usize
            + cell[1] as usize * medium.dim[0] as usize + cell[0] as usize;
        if cell_index >= medium.sigma_a.len() {
            return Err("path sampling escaped the world voxel array".into());
        }
        let region_index = region_at(medium, cell);
        let (speed, density, alpha, scatter, g, absorption_exp, scattering_exp) =
            if let Some(index) = region_index {
                let region = &medium.material_regions[index];
                (
                    medium.c[cell_index].max(20.0) as f64,
                    region.effective_density_kg_m3.max(0.1) as f64,
                    medium.sigma_a[cell_index].max(0.0) as f64,
                    medium.sigma_s[cell_index].max(0.0) as f64,
                    medium.g[cell_index].clamp(-0.99, 0.99) as f64,
                    region.absorption_frequency_exponent as f64,
                    region.scattering_frequency_exponent as f64,
                )
            } else {
                (
                    air.sound_speed_mps.max(1.0) as f64,
                    air.density_kg_m3.max(0.001) as f64,
                    air.absorption_m_inv.max(0.0) as f64,
                    0.0,
                    0.0,
                    air.absorption_frequency_exponent as f64,
                    0.0,
                )
            };

        if region_index != previous_region {
            let interface = fluid_interface_power(
                previous_speed as f32,
                previous_density as f32,
                speed as f32,
                density as f32,
                1.0,
            );
            interface_transmission *= interface.transmitted as f64;
            previous_region = region_index;
        }

        let ds = distance / segments as f64;
        let term_index = terms.iter().position(|term| term.region_index == region_index);
        if let Some(index) = term_index {
            terms[index].absorption_integral += alpha * ds;
            terms[index].scattering_integral += scatter * (1.0 - g).max(0.0) * ds;
        } else {
            terms.push(PathLossTerm {
                region_index,
                absorption_integral: alpha * ds,
                scattering_integral: scatter * (1.0 - g).max(0.0) * ds,
                absorption_exponent: absorption_exp.clamp(-2.0, 4.0),
                scattering_exponent: scattering_exp.clamp(-2.0, 4.0),
            });
        }
        travel_seconds += ds / speed.max(1.0);
        previous_speed = speed;
        previous_density = density;
    }

    Ok(PathProfile {
        interface_transmission: interface_transmission.clamp(0.0, 1.0),
        travel_seconds: travel_seconds.max(0.0),
        terms,
    })
}

/// Process an audio clip through the configured direct source-to-listener path.
/// The listener is an absorbing endpoint: captured energy does not continue.
pub fn process_audio_for_listener(
    clip: &AudioClip,
    source_position_mm: [f32; 3],
    listener: ListenerModel,
    medium: &VoxelMedium,
    air: WorldAirModel,
) -> Result<ListenerCapture, String> {
    if clip.samples.is_empty() || clip.sample_rate_hz == 0 {
        return Err("audio input must contain samples and have a non-zero sample rate".into());
    }
    let path = trace_path_profile(source_position_mm, listener, medium, air)?;
    let delay_samples = (path.travel_seconds * clip.sample_rate_hz as f64).ceil() as usize;
    let required_len = clip.samples.len().checked_add(delay_samples).and_then(|n| n.checked_add(1))
        .ok_or_else(|| "audio processing length overflow".to_string())?;
    let fft_len = required_len.checked_next_power_of_two()
        .ok_or_else(|| "audio sample is too large for a power-of-two FFT buffer".to_string())?;
    if fft_len > (1usize << 23) {
        return Err(format!("audio FFT requires {fft_len} samples; current workbench limit is 8,388,608. Trim the recording and retry."));
    }

    let mut spectrum = vec![Complex::default(); fft_len];
    for (target, sample) in spectrum.iter_mut().zip(clip.samples.iter()) {
        *target = Complex::new(*sample as f64, 0.0);
    }
    fft(&mut spectrum, false)?;
    // Read each original positive-frequency bin before replacing it with the
    // filtered value. This avoids retaining a second full-size complex buffer.
    let bins = fft_len / 2 + 1;
    let mut frequencies = Vec::with_capacity(bins);
    let mut input_power = Vec::with_capacity(bins);
    let mut received_power = Vec::with_capacity(bins);
    let mut lost_power = Vec::with_capacity(bins);
    let mut transmission_db = Vec::with_capacity(bins);
    let sample_count = clip.samples.len().max(1) as f64;
    let mut total_input = 0.0f64;
    let mut total_received = 0.0f64;
    for bin in 0..bins {
        let frequency = bin as f64 * clip.sample_rate_hz as f64 / fft_len as f64;
        let attenuation = path.energy_transmission(frequency).sqrt();
        let phase = -2.0 * PI * frequency * path.travel_seconds;
        let transfer = Complex::new(phase.cos(), phase.sin()).scale(attenuation);
        let input_bin = spectrum[bin];
        let positive = input_bin.mul(transfer);
        spectrum[bin] = if bin == fft_len / 2 {
            // The Nyquist bin is self-conjugate in a real-valued signal.
            Complex::new(positive.re, 0.0)
        } else {
            positive
        };
        if bin > 0 && bin < fft_len / 2 {
            spectrum[fft_len - bin] = Complex::new(positive.re, -positive.im);
        }

        let symmetry_weight = if bin == 0 || bin == fft_len / 2 { 1.0 } else { 2.0 };
        let raw_input = symmetry_weight * input_bin.magnitude_squared()
            / (fft_len as f64 * sample_count);
        let raw_output = symmetry_weight * positive.magnitude_squared() / (fft_len as f64 * sample_count);
        let input = raw_input;
        let output = raw_output;
        total_input += input;
        total_received += output;
        frequencies.push(frequency as f32);
        input_power.push(input as f32);
        received_power.push(output as f32);
        lost_power.push((input - output).max(0.0) as f32);
        // Transmission is a property of the path transfer function, not of the
        // particular input recording. This also keeps silent bins meaningful.
        transmission_db.push((10.0 * path.energy_transmission(frequency).max(1e-12).log10()) as f32);
    }

    fft(&mut spectrum, true)?;
    let output_len = clip.samples.len().saturating_add(delay_samples);
    let mut output_samples = Vec::with_capacity(output_len);
    output_samples.extend(spectrum.iter().take(output_len).map(|value| value.re as f32));
    let total_lost = (total_input - total_received).max(0.0);

    Ok(ListenerCapture {
        samples: output_samples,
        sample_rate_hz: clip.sample_rate_hz,
        travel_time_seconds: path.travel_seconds as f32,
        total_input_power: total_input as f32,
        total_received_power: total_received as f32,
        total_lost_power: total_lost as f32,
        spectrum: ListenerSpectrum {
            frequency_hz: frequencies,
            input_power,
            received_power,
            lost_power,
            transmission_db,
        },
    })
}

pub fn export_listener_csv(path: impl AsRef<Path>, capture: &ListenerCapture) -> Result<(), String> {
    let path = path.as_ref();
    let mut csv = String::from("frequency_hz,input_power,received_power,lost_power,transmission_db\n");
    for index in 0..capture.spectrum.frequency_hz.len() {
        use std::fmt::Write as _;
        writeln!(
            &mut csv,
            "{:.6},{:.9},{:.9},{:.9},{:.6}",
            capture.spectrum.frequency_hz[index],
            capture.spectrum.input_power[index],
            capture.spectrum.received_power[index],
            capture.spectrum.lost_power[index],
            capture.spectrum.transmission_db[index],
        ).map_err(|error| format!("unable to format listener CSV: {error}"))?;
    }
    std::fs::write(path, csv)
        .map_err(|error| format!("unable to write listener CSV '{}': {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{fft, Complex};

    #[test]
    fn fft_round_trip_preserves_samples() {
        let input = [0.0, 1.0, 0.0, -1.0, 0.25, 0.0, -0.25, 0.0];
        let mut values = input.iter().map(|value| Complex::new(*value, 0.0)).collect::<Vec<_>>();
        fft(&mut values, false).unwrap();
        fft(&mut values, true).unwrap();
        for (actual, expected) in values.iter().zip(input.iter()) {
            assert!((actual.re - expected).abs() < 1e-9);
            assert!(actual.im.abs() < 1e-9);
        }
    }

    #[test]
    fn positive_frequency_exponent_reduces_high_frequency_transmission() {
        let path = super::PathProfile {
            interface_transmission: 1.0,
            travel_seconds: 0.001,
            terms: vec![super::PathLossTerm {
                region_index: Some(0),
                absorption_integral: 0.8,
                scattering_integral: 0.2,
                absorption_exponent: 0.8,
                scattering_exponent: 0.6,
            }],
        };
        assert!(path.energy_transmission(20_000.0) < path.energy_transmission(1_000.0));
    }
}

#[cfg(test)]
mod listener_integration_tests {
    use super::{process_audio_for_listener, ListenerModel, WorldAirModel};
    use val_core::audio::AudioClip;
    use val_felt::{MaterialRegion, VoxelMedium};

    fn one_dimensional_felt_path() -> VoxelMedium {
        let mut sigma_a = vec![0.0; 128];
        let mut sigma_s = vec![0.0; 128];
        let mut g = vec![0.0; 128];
        let mut c = vec![343.0; 128];
        for x in 64..112 {
            sigma_a[x] = 30.0;
            sigma_s[x] = 5.0;
            g[x] = 0.45;
            c[x] = 220.0;
        }
        VoxelMedium {
            dim: [128, 1, 1],
            spacing: 0.001,
            sigma_a,
            sigma_s,
            g,
            c,
            material_regions: vec![MaterialRegion {
                name: "test-felt".into(),
                id: 1,
                min: [64, 0, 0],
                max: [112, 1, 1],
                visual: [0.8, 0.3, 0.1, 0.02],
                visual_texture: [0.0, 0.0009, 0.7, 0.0],
                absorption_frequency_exponent: 0.8,
                scattering_frequency_exponent: 0.6,
                density_kg_m3: 180.0,
                effective_density_kg_m3: 1.8,
                porosity: 0.82,
                sigma_a_mean: 30.0,
                sigma_s_mean: 5.0,
                sound_speed_mean: 220.0,
            }],
        }
    }

    #[test]
    fn offline_listener_pass_applies_path_loss_and_travel_delay() {
        let clip = AudioClip {
            samples: (0..128)
                .map(|index| (std::f32::consts::TAU * 4.0 * index as f32 / 128.0).sin())
                .collect(),
            sample_rate_hz: 48_000,
            source_channels: 2,
        };
        let capture = process_audio_for_listener(
            &clip,
            [18.0, 0.5, 0.5],
            ListenerModel {
                position_mm: [120.0, 0.5, 0.5],
                normal: [-1.0, 0.0, 0.0],
                aperture_radius_mm: 2.0,
            },
            &one_dimensional_felt_path(),
            WorldAirModel {
                sound_speed_mps: 343.0,
                density_kg_m3: 1.204,
                absorption_m_inv: 0.0,
                absorption_frequency_exponent: 0.5,
            },
        )
        .unwrap();

        assert!(capture.travel_time_seconds > 0.0);
        assert!(capture.samples.len() > clip.samples.len());
        assert!(capture.total_received_power < capture.total_input_power);
        assert!(capture.total_lost_power > 0.0);
        assert_eq!(capture.spectrum.frequency_hz.len(), capture.spectrum.transmission_db.len());
    }
}
