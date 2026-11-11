use val_core::{Fiber2, Rng, Vec2, AIR, WOOL};
use val_felt::FeltBlock;

#[derive(Clone, Copy, Debug)]
pub struct RaySample {
    pub pos: Vec2,
    pub w: f32,
    pub t: f32,
    pub freq: f32,
} // t = arrival time (s)
#[derive(Clone, Debug)]
pub struct Ray {
    pub freq: f32,
    pub samples: Vec<RaySample>,
    pub died_absorbed: bool,
}

pub struct TraceOpts {
    pub max_bounces: u32,
    pub ds: f32,
    pub eps: f32,
    pub lump_solid: bool,
}
impl Default for TraceOpts {
    fn default() -> Self {
        Self {
            max_bounces: 24,
            ds: 0.25,
            eps: 1e-4,
            lump_solid: false,
        }
    }
}

/// 2D ray-circle intersection: returns nearest t>eps hitting a fiber boundary,
/// plus the outward normal there. None if the ray leaves the block first.
fn intersect_fiber(o: Vec2, d: Vec2, f: &Fiber2, eps: f32) -> Option<(f32, Vec2)> {
    let oc = o - f.c;
    let b = oc.dot(d);
    let c = oc.dot(oc) - f.r * f.r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t = -b - sq;
    if t < eps {
        t = -b + sq
    }
    if t < eps {
        return None;
    }
    let p = o + d * t;
    Some((t, (p - f.c).norm()))
}
fn exit_block(o: Vec2, d: Vec2, l: f32) -> f32 {
    // param to leave [0,L]^2
    let mut t = f32::INFINITY;
    for (axis, od) in [(o.x, d.x), (o.y, d.y)] {
        if od.abs() > 1e-9 {
            let t1 = (0.0 - axis) / od;
            let t2 = (l - axis) / od;
            for tt in [t1, t2] {
                if tt > 1e-6 && tt < t {
                    t = tt
                }
            }
        }
    }
    t
}

/// Impedance partition at a fluid-fluid boundary with Snell refraction.
/// Returns (R_power, T_power, refracted_dir_or_none, new_medium_speed).
fn branch(_o: Vec2, d: Vec2, n_out: Vec2, going_in: bool) -> (f32, f32, Option<Vec2>) {
    let (z1, z2) = if going_in {
        (AIR.rho * AIR.c_snd, WOOL.rho * WOOL.c_snd)
    } else {
        (WOOL.rho * WOOL.c_snd, AIR.rho * AIR.c_snd)
    };
    let n = if going_in { -n_out } else { n_out }; // normal into incidence medium
    let cosi = (-d).dot(n).clamp(0.0, 1.0); // cos of incidence
    let ratio = z1 / z2;
    let sin2t = (ratio * ratio * (1.0 - cosi * cosi)).min(1.0); // Snell (pressure, fluid)
    if sin2t > 1.0 {
        return (1.0, 0.0, None);
    } // total internal reflection
    let cost = (1.0 - sin2t).sqrt();
    let rp = (z2 * cosi - z1 * cost) / (z2 * cosi + z1 * cost);
    let r_pow = rp * rp;
    let t_pow = (z1 * cost).max(1e-9) / (z2 * cosi).max(1e-9) * (1.0 - r_pow); // power transmission
                                                                               // refracted direction
    let t_dir = ((ratio * cosi - cost) * n - ratio * d).norm();
    (r_pow, t_pow, Some(t_dir))
}

/// Fire `rays_per_freq` packets per frequency over the left edge (cosine-weighted),
/// trace each, splat arrival times into `wave`, accumulate energy budget.
pub fn trace_bundle(
    block: &FeltBlock,
    freqs: &[f32],
    rays_per_freq: u32,
    seed: u64,
    opts: &TraceOpts,
    wave: &mut crate::wave::ArrivalGrid,
) -> (Vec<Ray>, f32 /*absorbed*/, f32 /*incident*/) {
    let mut rng = Rng::seed_from_u64(seed ^ 0x5DEECE66);
    let L = block.bounds.0;
    let mut out = Vec::new();
    let mut absorbed = 0.0f32;
    let mut incident = 0.0f32;
    for &f in freqs {
        let c_air = AIR.c_snd;
        for _ in 0..rays_per_freq {
            let o = Vec2::new(0.0, rng.range(0.0, L));
            let d = rng.cos_dir(); // fan into the block
            let w0 = 1.0 / (rays_per_freq as f32 * freqs.len() as f32) as f32;
            incident += w0;
            out.push(trace_one(
                block,
                o,
                d,
                f,
                w0,
                c_air,
                opts,
                wave,
                &mut absorbed,
                &mut rng,
            ));
        }
    }
    (out, absorbed, incident)
}

fn trace_one(
    block: &FeltBlock,
    o0: Vec2,
    d0: Vec2,
    freq: f32,
    w0: f32,
    c_air: f32,
    opts: &TraceOpts,
    wave: &mut crate::wave::ArrivalGrid,
    absorbed: &mut f32,
    rng: &mut Rng,
) -> Ray {
    let L = block.bounds.0;
    let mut pos = o0;
    let mut dir = d0.norm();
    let mut w = w0;
    let mut t = 0.0f32;
    let mut in_solid = false;
    let mut samples = vec![RaySample { pos, w, t, freq }];
    for _bounce in 0..opts.max_bounces {
        // nearest fiber hit vs block exit
        let mut best: Option<(f32, Vec2, usize)> = None;
        for (i, fb) in block.fibers.iter().enumerate() {
            if let Some((tt, n)) = intersect_fiber(pos, dir, fb, opts.eps) {
                if best.map_or(true, |(bt, _, _)| tt < bt) {
                    best = Some((tt, n, i));
                }
            }
        }
        let t_exit = exit_block(pos, dir, L);
        let step_end = match best {
            Some((tt, _, _)) if tt < t_exit => tt,
            _ => t_exit.min(opts.ds.max(t_exit)),
        };
        // march with volumetric attenuation + accumulate arrival time
        let c = if in_solid { WOOL.c_snd } else { c_air };
        let alpha_med = if in_solid {
            200.0 /*Np/m placeholder*/
        } else {
            0.0
        };
        let mut remaining = step_end;
        while remaining > opts.eps {
            let ds = remaining.min(opts.ds);
            w *= (-alpha_med * ds * 1e-3).exp(); // ds in mm -> m
            t += ds * 1e-3 / c;
            pos = pos + dir * ds;
            remaining -= ds;
            samples.push(RaySample { pos, w, t, freq });
            wave.splat(pos, t, w); // <-- wavefront accumulation
        }
        if w < 1e-6 {
            break;
        } // Russian-roulette floor
        match best {
            Some((_tt, n, _i)) if _tt < t_exit => {
                let (rp, tp, tdir) = branch(pos, dir, n, !in_solid);
                let u = rng.f32();
                if u < rp {
                    dir = dir.reflect(n);
                }
                // reflect, same medium
                else if u < rp + tp {
                    if let Some(td) = tdir {
                        dir = td;
                        in_solid = !in_solid;
                    }
                }
                // transmit
                else {
                    *absorbed += w;
                    return Ray {
                        freq,
                        samples,
                        died_absorbed: true,
                    };
                } // absorb
            }
            _ => break, // exited block (counts as scattered/reflected out, not absorbed)
        }
    }
    Ray {
        freq,
        samples,
        died_absorbed: false,
    }
}
