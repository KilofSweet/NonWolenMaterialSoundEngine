# VAL GUI layout prototype

This revision changes only the egui presentation layer in `crates/val-view/src/ui.rs`.
It does not change the acoustic transport model, material generation, GPU shaders,
rendering algorithms, configuration schema, or simulation data.

## Workspace layout

1. Top toolbar: File / Edit / View / Simulation / Help and common actions.
2. Left Scene Explorer: emitter and material selection plus world summary.
3. Center Viewport: presentation tabs and the existing 3D viewport.
4. Right Properties: the existing configuration editors, moved from the left dock.
5. Bottom dock: playback timeline and basic diagnostics/status.

Panels are resizable in the same general style as Blender / CAD applications.

## Viewport tabs and current implementation status

- **Packet paths** enables the existing GPU debug overlay. These are sampled Monte-Carlo
  transport packet trajectories, not coherent acoustic wavefront trajectories.
- **Wave field** displays the existing GPU energy-volume visualization.
- **Inside material** is currently a UI placeholder. The current renderer does not yet
  isolate a selected material volume or render a dedicated interior view. Selecting this
  tab displays an explanatory notice; it does not alter the simulation.

The intent is to establish the interface layout without changing simulation behavior.

## Validation

The source archive was inspected and patched, but compilation/tests could not be run in
this environment because the Rust `cargo` executable is unavailable. Run `cargo fmt --all`
and `cargo check --workspace` locally after extracting the archive.


## Viewport alignment fix

The 3D volume is now rendered into the central egui viewport rectangle rather than
being stretched across the complete window behind the dock panels. The viewport
rectangle is converted from egui points to physical pixels using
`pixels_per_point`; the scene target and camera aspect are recreated only when
the central viewport dimensions change. The final blit is restricted to that
rectangle with matching viewport/scissor state. Scene-space editor overlays use
the central viewport's own aspect ratio.

This is a presentation/layout correction only. Acoustic transport, material
generation, simulation configuration, and shader algorithms are unchanged.


## UI themes

The editor now includes seven UI-only themes: Light, Dark, and five custom palettes
based on the supplied swatches. Select them from View → Theme or from the
Interface theme selector at the top of the Properties panel. The selection is
stored in `val-theme.txt` in the working directory, separately from `val.toml`.
Theme changes only update egui visuals; they do not change the renderer, shaders,
camera, acoustic materials, or simulation parameters.
