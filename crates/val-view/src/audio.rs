//! MP3/WAV-independent audio source inspection.
//! MP3 frames are decoded to mono samples; the UI uses a decimated waveform and
//! spectral peak summary. This is input preparation, not yet broadband GPU transport.
use std::{fs::File, io::BufReader, path::Path};

#[derive(Clone, Debug, Default)]
pub struct AudioSource {
    pub path: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: f32,
    pub mono_samples: Vec<f32>,
    pub waveform: Vec<f32>,
    pub peak_frequency_hz: f32,
    pub rms: f32,
    pub spectrum_bands: Vec<(f32, f32)>,
}

pub fn load(path: &Path) -> Result<AudioSource, String> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mp3" => load_mp3(path),
        "wav" => load_wav(path),
        _ => Err("Unsupported audio format. Choose MP3 or PCM WAV.".into()),
    }
}

fn finish(path: &Path, sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<AudioSource, String> {
    if sample_rate == 0 || samples.is_empty() {
        return Err("Audio file contains no decodable samples.".into());
    }
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
    let bins = 1024usize.min(samples.len()).max(1);
    let mut waveform = Vec::with_capacity(512);
    for i in 0..512 {
        let start = i * samples.len() / 512;
        let end = ((i + 1) * samples.len() / 512).max(start + 1).min(samples.len());
        let peak = samples[start..end].iter().fold(0.0f32, |a, b| a.max(b.abs()));
        waveform.push(peak);
    }
    // A lightweight 1024-point DFT peak estimate for an informative source
    // summary. It is not used as a substitute for a broadband transport solve.
    let n = bins;
    let mut peak_bin = 0usize;
    let mut peak_power = 0.0f64;
    let mut powers = vec![0.0f64; n / 2];
    let window_count = (samples.len() / n).clamp(1, 8);
    for k in 1..(n / 2).min(512) {
        let mut accumulated_power = 0.0f64;
        for window in 0..window_count {
            let start = if window_count == 1 { 0 } else {
                window * (samples.len() - n) / (window_count - 1)
            };
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for j in 0..n {
                let sample = samples[start + j] as f64;
                let hann = if n > 1 {
                    0.5 - 0.5 * (std::f64::consts::TAU * j as f64 / (n - 1) as f64).cos()
                } else { 1.0 };
                let phase = std::f64::consts::TAU * k as f64 * j as f64 / n as f64;
                re += sample * hann * phase.cos();
                im -= sample * hann * phase.sin();
            }
            accumulated_power += re * re + im * im;
        }
        let power = accumulated_power / window_count as f64;
        powers[k] = power;
        if power > peak_power { peak_power = power; peak_bin = k; }
    }
    let min_hz = 80.0f32;
    let max_hz = (sample_rate as f32 * 0.45).min(16_000.0).max(min_hz * 1.01);
    let mut spectrum_bands = Vec::with_capacity(8);
    let mut band_powers = Vec::with_capacity(8);
    for band in 0..8 {
        let lo_hz = min_hz * (max_hz / min_hz).powf(band as f32 / 8.0);
        let hi_hz = min_hz * (max_hz / min_hz).powf((band + 1) as f32 / 8.0);
        let lo = (lo_hz * n as f32 / sample_rate as f32).floor() as usize;
        let hi = (hi_hz * n as f32 / sample_rate as f32).ceil() as usize;
        let power = powers.get(lo.max(1)..hi.min(powers.len())).unwrap_or(&[])
            .iter().sum::<f64>();
        spectrum_bands.push(((lo_hz * hi_hz).sqrt(), 0.0));
        band_powers.push(power);
    }
    let max_band = band_powers.iter().copied().fold(0.0f64, f64::max).max(1.0e-30);
    for (index, band) in spectrum_bands.iter_mut().enumerate() {
        band.1 = (band_powers[index] / max_band).sqrt() as f32;
    }
    Ok(AudioSource {
        path: path.display().to_string(),
        sample_rate,
        channels,
        duration_seconds: samples.len() as f32 / sample_rate as f32,
        mono_samples: samples,
        waveform,
        peak_frequency_hz: peak_bin as f32 * sample_rate as f32 / n as f32,
        rms,
        spectrum_bands,
    })
}

fn load_mp3(path: &Path) -> Result<AudioSource, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open audio file: {e}"))?;
    let mut decoder = minimp3::Decoder::new(BufReader::new(file));
    let mut samples = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                sample_rate = frame.sample_rate as u32;
                channels = frame.channels as u16;
                if frame.channels == 1 {
                    samples.extend(frame.data.iter().map(|s| *s as f32 / 32768.0));
                } else {
                    samples.extend(frame.data.chunks_exact(frame.channels)
                        .map(|ch| ch.iter().map(|s| *s as f32 / 32768.0).sum::<f32>() / frame.channels as f32));
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(minimp3::Error::SkippedData) => continue,
            Err(e) => return Err(format!("MP3 decode failed: {e:?}")),
        }
    }
    finish(path, sample_rate, channels, samples)
}

fn load_wav(path: &Path) -> Result<AudioSource, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Cannot read WAV: {e}"))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid RIFF/WAVE file.".into());
    }
    let mut pos = 12usize;
    let mut format = 0u16;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let start = pos + 8;
        let end = start.saturating_add(size).min(bytes.len());
        if id == b"fmt " && end >= start + 16 {
            format = u16::from_le_bytes(bytes[start..start+2].try_into().unwrap());
            channels = u16::from_le_bytes(bytes[start+2..start+4].try_into().unwrap());
            rate = u32::from_le_bytes(bytes[start+4..start+8].try_into().unwrap());
            bits = u16::from_le_bytes(bytes[start+14..start+16].try_into().unwrap());
        } else if id == b"data" { data = Some(&bytes[start..end]); }
        pos = start.saturating_add(size + (size & 1));
    }
    let data = data.ok_or("WAV has no data chunk")?;
    if channels == 0 || rate == 0 { return Err("Invalid WAV format metadata.".into()); }
    let bytes_per_sample = (bits / 8) as usize;
    if bytes_per_sample == 0 { return Err("Unsupported WAV bit depth.".into()); }
    let mut mono = Vec::new();
    for frame in data.chunks_exact(bytes_per_sample * channels as usize) {
        let mut sum = 0.0f32;
        for ch in 0..channels as usize {
            let b = &frame[ch*bytes_per_sample..(ch+1)*bytes_per_sample];
            let v = match (format, bits) {
                (1, 8) => (b[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([b[0],b[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let x = (b[0] as i32) | ((b[1] as i32)<<8) | ((b[2] as i32)<<16);
                    ((x << 8) >> 8) as f32 / 8_388_608.0
                },
                (1, 32) => i32::from_le_bytes(b.try_into().unwrap()) as f32 / 2_147_483_648.0,
                (3, 32) => f32::from_le_bytes(b.try_into().unwrap()),
                _ => return Err(format!("Unsupported WAV encoding: format {format}, {bits} bit.")),
            };
            sum += v;
        }
        mono.push(sum / channels as f32);
    }
    finish(path, rate, channels, mono)
}
