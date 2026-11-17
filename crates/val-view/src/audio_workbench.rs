use val_core::audio::AudioClip;
use crate::spectral::ListenerCapture;

#[derive(Default)]
pub struct AudioWorkbench {
    pub input_path: String,
    pub output_wav_path: String,
    pub output_csv_path: String,
    pub output_packet_csv_path: String,
    pub clip: Option<AudioClip>,
    pub capture: Option<ListenerCapture>,
    pub status: String,
}

impl AudioWorkbench {
    pub fn ensure_default_paths(&mut self) {
        if self.input_path.is_empty() {
            self.input_path = "audio/input.wav".into();
        }
        if self.output_wav_path.is_empty() {
            self.output_wav_path = "listener_output.wav".into();
        }
        if self.output_csv_path.is_empty() {
            self.output_csv_path = "listener_spectrum.csv".into();
        }
        if self.output_packet_csv_path.is_empty() {
            self.output_packet_csv_path = "packet_listener_capture.csv".into();
        }
    }
}
