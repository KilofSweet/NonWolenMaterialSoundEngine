// Shader modules are separate files so transport math can be replaced without
// editing the visual renderer or its fullscreen presentation pass.
pub const SOLVE: &str = include_str!("shaders/transport.wgsl");
pub const RENDER: &str = include_str!("shaders/volume.wgsl");
pub const BLIT: &str = include_str!("shaders/blit.wgsl");
pub const ENERGY_DISPLAY: &str = include_str!("shaders/energy_display.wgsl");
pub const DEBUG_RAYS: &str = include_str!("shaders/debug_rays.wgsl");
