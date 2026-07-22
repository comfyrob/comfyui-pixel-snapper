# Pixel Snapper — native GLSL edition

The Pixel Snapper algorithm rebuilt on **core ComfyUI nodes only** — three
chained `GLSLShader` nodes plus primitives. No custom node install needed
(requires a ComfyUI new enough to ship the core GLSL Shader node, and a
working headless GL context — see the note below for Modal).

Load [`pixel_snapper_glsl.json`](pixel_snapper_glsl.json) as a workflow, pick
an image, hit Run.

## The three stages

| Stage | File | size_mode | What it does |
|---|---|---|---|
| 1. Quantize | [`quantize.glsl`](quantize.glsl) | `from_input` | GPU k-means (farthest-point seeded, 10 iterations over a 64x64 sample cache). `IMAGE0` = quantized image, `IMAGE1` = palette stripes. |
| 2. Detect Grid | [`detect.glsl`](detect.glsl) | `custom` 2048x8 | Gradient profiles -> strong-peak median spacing (a port of the Python estimator) -> grid step + phase, written as float RGBA into a tiny params texture. |
| 3. Snap | [`snap.glsl`](snap.glsl) | `from_input` | Majority-vote per grid cell, painted at full resolution. Majority (not average) preserves dithering. |

Wiring (all done in the bundled workflow): the source image feeds stage 1's
`image0` **and** `image1` (multi-pass ping-pong consumes `image0`, so the
stable copy rides on `image1` — same trick in stage 2). A `Color Count` INT
primitive feeds stage 1 and stage 3; a `Pixel Size (0 = auto)` INT primitive
feeds stage 2. Values pass between shader nodes as float32 textures, so
nothing is lost to 8-bit.

For the tiny native-resolution asset, set the `ImageScaleBy` node to
`1 / pixel_size` (e.g. 0.125 for 8px art pixels), nearest-exact.

## Differences vs. the Python node

- **Uniform grid**: step + phase instead of the elastic walker. Slightly
  worse on generations whose grid drifts across the image; identical
  behavior otherwise. (A future version could write per-cut positions into
  the params texture — the single-fragment estimator pass makes an exact
  walker port possible.)
- **No mask/alpha handling.**
- Limits: source up to 2048px per side, up to 32 colors, art pixels up to
  128px.
- On the synthetic messy-grid benchmark both implementations score ~88.5%
  cell accuracy (`test_glsl_pipeline.py` vs `tests/smoke_test.py`).

## Testing

```bash
uv run --no-project --with numpy,pillow,requests \
  python glsl/test_glsl_pipeline.py https://<your-comfyui-host>
```

Builds messy synthetic pixel art, runs the pipeline through the API,
and scores the result against the Python implementation's ground truth.

## GLSL gotchas learned the hard way

- **Large local arrays miscompile** in fragment shaders (a 700-float array
  made texelFetch silently return junk, varying run to run). The estimator
  is deliberately arrayless — medians via counting + profile re-walks.
- The node's upload/readback both flip vertically (symmetric), so absolute
  `texelFetch` row addressing survives across chained shader nodes.
- Multi-pass ping-pong replaces `u_image0` only; other image slots stay
  bound across passes.
- On Modal, headless GL needs the graphics driver libs + an ANGLE bypass —
  see "Layer 5.6" in the Modal deploy config.
