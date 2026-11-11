use val_core::Vec2;
pub struct ArrivalGrid { pub w: usize, pub h: usize, pub t: Vec<f32>, pub wt: Vec<f32> } // t=Σ(w·τ)/Σw
impl ArrivalGrid {
    pub fn new(w: usize, h: usize) -> Self { Self { w, h, t: vec![0.0; w*h], wt: vec![0.0; w*h] } }
    pub fn splat(&mut self, p: Vec2, tau: f32, weight: f32) {
        let L = (self.w.max(self.h)) as f32; // assume square block mapped to grid
        let gx = (p.x / L * (self.w-1) as f32).clamp(0.0,(self.w-1) as f32);
        let gy = (p.y / L * (self.h-1) as f32).clamp(0.0,(self.h-1) as f32);
        let (x0,y0) = (gx.floor() as usize, gy.floor() as usize);
        let (fx,fy) = (gx-x0 as f32, gy-y0 as f32);
        for (dx,wx) in [(0,1.0-fx),(1,fx)] { for (dy,wy) in [(0,1.0-fy),(1,fy)] {
            let x = (x0+dx).min(self.w-1); let y = (y0+dy).min(self.h-1); let i = y*self.w+x;
            let ww = weight*wx*wy; self.wt[i]+=ww; self.t[i]+=ww*tau;
        }}
    }
    pub fn mean_time(&self, i: usize) -> f32 { if self.wt[i]>1e-9 { self.t[i]/self.wt[i] } else { f32::NAN } }
}
