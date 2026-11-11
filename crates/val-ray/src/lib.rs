pub mod trace; pub mod wave;
pub use trace::{trace_bundle, Ray, RaySample, TraceOpts};
pub use wave::ArrivalGrid;
/// α estimate from the budget (FR 4.1.4 / verification seed):
pub fn alpha_estimate(absorbed: f32, incident: f32) -> f32 { (absorbed/incident.max(1e-9)).clamp(0.0,1.0) }
