use val_core::config::{EmitterConfig, SimulationConfig};

/// Shared playback state driven by keyboard shortcuts and the egui runtime panel.
#[derive(Clone, Debug)]
pub struct PlaybackState {
    pub time_us: f32,
    pub speed: f32,
    pub paused: bool,
    pub frequency_hz: f32,
}

impl PlaybackState {
    pub fn new(emitter: &EmitterConfig) -> Self {
        Self {
            time_us: 0.0,
            speed: 1.0,
            paused: false,
            frequency_hz: emitter.frequency_hz,
        }
    }

    pub fn advance(&mut self, real_dt_seconds: f32, simulation: &SimulationConfig) {
        if !self.paused {
            let history_end_us = simulation.energy_time_bin_us
                * simulation.energy_time_bins as f32;
            self.time_us = (self.time_us
                + real_dt_seconds.max(0.0)
                    * simulation.timeline_rate_us_per_second
                    * self.speed)
                .min(history_end_us);
            if self.time_us >= history_end_us {
                self.paused = true;
            }
        }
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn reset(&mut self) {
        self.time_us = 0.0;
    }

    pub fn scrub(&mut self, delta_us: f32, simulation: &SimulationConfig) {
        let history_end_us = simulation.energy_time_bin_us
            * simulation.energy_time_bins as f32;
        self.time_us = (self.time_us + delta_us).clamp(0.0, history_end_us);
    }

    pub fn adjust_speed(&mut self, factor: f32) {
        self.speed = (self.speed * factor).clamp(0.05, 16.0);
    }

    pub fn adjust_frequency(&mut self, factor: f32) {
        self.frequency_hz = (self.frequency_hz * factor).clamp(1.0, 200_000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::PlaybackState;
    use val_core::config::{EmitterConfig, SimulationConfig};

    #[test]
    fn playback_stops_at_the_end_of_recorded_history() {
        let simulation = SimulationConfig::default();
        let emitter = EmitterConfig::default();
        let mut state = PlaybackState::new(&emitter);
        state.speed = 16.0;

        state.advance(10.0, &simulation);

        let history_end = simulation.energy_time_bin_us * simulation.energy_time_bins as f32;
        assert_eq!(state.time_us, history_end);
        assert!(state.paused);
    }

    #[test]
    fn maximum_speed_advances_less_than_one_bin_per_sixtieth_second_frame() {
        let simulation = SimulationConfig::default();
        let emitter = EmitterConfig::default();
        let mut state = PlaybackState::new(&emitter);
        state.speed = 16.0;

        state.advance(1.0 / 60.0, &simulation);

        // 60 us/s at 1x × 16x / 60 FPS = 16 us per frame, below a 20 us bin.
        assert!((state.time_us - 16.0).abs() < 0.01);
        assert!(state.time_us < simulation.energy_time_bin_us);
    }

    #[test]
    fn scrub_moves_by_the_configured_histogram_step_and_stays_in_range() {
        let simulation = SimulationConfig::default();
        let emitter = EmitterConfig::default();
        let mut state = PlaybackState::new(&emitter);
        state.scrub(simulation.energy_time_bin_us, &simulation);
        assert_eq!(state.time_us, simulation.energy_time_bin_us);

        state.scrub(-1000.0, &simulation);
        assert_eq!(state.time_us, 0.0);
    }
}
