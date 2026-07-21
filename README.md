# ComfyUI Pixel Snapper

Snap messy AI-generated pixel art to a **perfect grid**.

AI image models can't produce true pixel art: pixel sizes drift, the grid wobbles,
and colors smear into hundreds of near-duplicates. This node detects the *implicit*
grid in a generated image, quantizes the colors, and collapses every grid cell to
exactly one pixel — giving you a real, scalable, native-resolution sprite.

A ComfyUI port of [Sprite Fusion Pixel Snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper)
by Hugo Duprez (MIT).

## Install

On your ComfyUI machine:

```bash
cd /comfyui/custom_nodes
git clone https://github.com/rslosh/comfyui-pixel-snapper.git
```

Restart ComfyUI. The node appears as **Pixel Snapper** under `image/pixel art`.
Only dependency is `numpy`, which ComfyUI already ships.

> Requires a ComfyUI version with the V3 node API (`comfy_api.latest`), i.e. any
> reasonably current build.

## The node

### Inputs

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | The pixel-art-style image to fix. |
| `color_count` | INT | Palette size for k-means quantization (default 16). |
| `pixel_size` | INT | Size of one art pixel in the source. **0 = auto-detect.** Set explicitly for batches/animations so every frame gets the same grid. |
| `palette` | STRING | Optional fixed palette: comma-separated 6-digit hex (`0d2b45,203c56,...`). Empty = use quantized colors. Paste straight from [Lospec](https://lospec.com/palette-list). |
| `mask` | MASK | Optional transparency (1 = transparent, same convention as LoadImage's MASK output). |

### Outputs

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Snapped result scaled back to the input size (crisp nearest-neighbor) — chains into any workflow. |
| `native image` | IMAGE | The real asset: **one pixel per grid cell** (e.g. a 1024px generation with 16px cells comes out 64x64). |
| `native mask` | MASK | Transparency at native resolution. |
| `pixel size` | FLOAT | The detected (or overridden) grid step, in source pixels. |

## How it works

1. **Quantize** — seeded k-means reduces the image to `color_count` colors.
2. **Detect** — horizontal/vertical gradient profiles reveal where the grid lines are;
   the median peak spacing gives the pixel size.
3. **Cut** — an elastic walker places cut lines, snapping each to the strongest nearby
   color edge, then cross-validates both axes.
4. **Resample** — each cell becomes one pixel by majority vote (which preserves
   dithering), optionally remapped to your fixed palette.

## Testing without ComfyUI

The algorithm lives in `pixel_snapper.py` with no ComfyUI or torch dependency:

```bash
python tests/smoke_test.py
```

## Credits

- Algorithm: [Hugo Duprez](https://www.hugoduprez.com/) — [spritefusion-pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper) (MIT)
- ComfyUI port: Rob Losch

## License

MIT — see [LICENSE](LICENSE).
