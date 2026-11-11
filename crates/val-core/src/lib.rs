pub mod acoustics;
pub mod config;

pub use glam::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;

pub type Rng = Pcg64;

pub fn new_rng(seed: u64) -> Rng {
    Pcg64::seed_from_u64(seed)
}
