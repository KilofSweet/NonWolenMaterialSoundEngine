//! WAV import/export helpers for the VAL listener pipeline.
//!
//! The workbench currently accepts uncompressed PCM and IEEE-float WAV files.
//! Samples are mixed down to mono; the imported sample rate is preserved.

use std::path::Path;

#[derive(Clone, Debug)]
pub struct AudioClip {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub source_channels: u16,
}

impl AudioClip {
    pub fn load_wav(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let mut reader = hound::WavReader::open(path)
            .map_err(|error| format!("unable to open audio file '{}': {error}", path.display()))?;
        let spec = reader.spec();
        if spec.channels == 0 || spec.sample_rate == 0 {
            return Err("audio file has an invalid channel count or sample rate".into());
        }
        let channel_count = spec.channels as usize;
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                    return Err(format!(
                        "unsupported PCM sample width: {} bits (expected 1..=32)",
                        spec.bits_per_sample
                    ));
                }
                let scale = ((1u64 << (spec.bits_per_sample - 1)) as f64) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| {
                        sample
                            .map(|value| (value as f32 / scale).clamp(-1.0, 1.0))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        if interleaved.is_empty() {
            return Err("audio file contains no samples".into());
        }
        if interleaved.len() % channel_count != 0 {
            return Err("audio file ends with an incomplete interleaved frame".into());
        }

        let mut mono = Vec::new();
        mono.try_reserve_exact(interleaved.len() / channel_count)
            .map_err(|error| format!("unable to allocate mono audio samples: {error}"))?;
        for frame in interleaved.chunks_exact(channel_count) {
            let sum = frame.iter().map(|sample| *sample as f64).sum::<f64>();
            mono.push((sum / channel_count as f64) as f32);
        }
        if mono.iter().any(|sample| !sample.is_finite()) {
            return Err("audio file contains NaN or infinite sample values".into());
        }

        Ok(Self {
            samples: mono,
            sample_rate_hz: spec.sample_rate,
            source_channels: spec.channels,
        })
    }

    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate_hz.max(1) as f64
    }
}

pub fn write_wav_mono(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
) -> Result<(), String> {
    if sample_rate_hz == 0 {
        return Err("cannot write WAV with a zero sample rate".into());
    }
    let path = path.as_ref();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|error| format!("unable to create WAV '{}': {error}", path.display()))?;
    for sample in samples {
        if !sample.is_finite() {
            return Err("refusing to export non-finite audio samples".into());
        }
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(value)
            .map_err(|error| format!("failed writing WAV '{}': {error}", path.display()))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("failed to finalize WAV '{}': {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::AudioClip;

    #[test]
    fn clip_duration_uses_sample_rate() {
        let clip = AudioClip {
            samples: vec![0.0; 48_000],
            sample_rate_hz: 48_000,
            source_channels: 2,
        };
        assert!((clip.duration_seconds() - 1.0).abs() < f64::EPSILON);
    }


    #[test]
    fn load_wav_downmixes_stereo_pcm_to_mono() {
        let path = std::env::temp_dir().join(format!(
            "val-audio-test-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        writer.write_sample(i16::MAX).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(i16::MAX).unwrap();
        writer.finalize().unwrap();

        let clip = AudioClip::load_wav(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(clip.sample_rate_hz, 48_000);
        assert_eq!(clip.source_channels, 2);
        assert_eq!(clip.samples.len(), 2);
        assert!((clip.samples[0] - 0.5).abs() < 0.001);
        assert!((clip.samples[1] - 0.5).abs() < 0.001);
    }
}
