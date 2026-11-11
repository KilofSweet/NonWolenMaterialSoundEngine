struct Dim {
    dims: vec4<u32>,
    spacing: vec4<f32>,
};
struct Play {
    t_us: f32,
    freq: f32,
    shell_thickness_m: f32,
    fade_us: f32,
    source: vec4<f32>,
    visual: vec4<f32>,
    emission: vec4<f32>,
    history: vec4<f32>, // x=bin width us, z=inverse fixed-point scale, w=max felt alpha
    visual2: vec4<f32>, // xyz=energy hot color (linear RGB), w=max air-wave alpha
    history_counts: vec4<u32>, // x=exact number of time bins
};

@group(0) @binding(0) var<storage, read> energy_hist: array<u32>;
@group(0) @binding(1) var<uniform> dim: Dim;
@group(0) @binding(2) var<uniform> play: Play;
@group(0) @binding(3) var energy_out: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = dim.dims.xyz;
    if (any(gid >= dims)) { return; }

    let cell_index = gid.x + gid.y * dims.x + gid.z * dims.x * dims.y;
    let bin_width = max(play.history.x, 0.001);
    let bin_count = play.history_counts.x;
    let now_us = max(play.t_us, 0.0);
    let current_bin = u32(floor(now_us / bin_width));
    let fade_us = max(play.fade_us, 1.0);
    // Three time constants retain 95% of the exponential visual tail while
    // limiting history reads. The current bin is included for short fades.
    let fade_bins = min(bin_count, max(1u, u32(ceil(3.0 * fade_us / bin_width)) + 1u));
    var energy = 0.0;

    for (var age_bin = 0u; age_bin < fade_bins; age_bin = age_bin + 1u) {
        if (age_bin <= current_bin) {
            let bin = current_bin - age_bin;
            if (bin < bin_count) {
                let raw = energy_hist[cell_index * bin_count + bin];
                let bin_start = f32(bin) * bin_width;
                let bin_end = bin_start + bin_width;
                // Deposits inside a temporal bin are treated as uniformly
                // distributed over that bin: its contribution ramps continuously
                // from 0 to 1 instead of popping on at a histogram-bin boundary.
                let fill_fraction = clamp((now_us - bin_start) / bin_width, 0.0, 1.0);
                let age_after_bin = max(now_us - bin_end, 0.0);
                let fade = exp(-age_after_bin / fade_us);
                energy += f32(raw) * play.history.z * fill_fraction * fade;
            }
        }
    }

    textureStore(energy_out, vec3<i32>(gid), vec4<f32>(energy, 0.0, 0.0, 1.0));
}
