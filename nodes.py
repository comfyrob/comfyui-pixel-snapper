"""ComfyUI V3 node for Pixel Snapper."""

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .pixel_snapper import Config, PixelSnapperError, snap_pixels


class PixelSnapper(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelSnapper",
            display_name="Pixel Snapper",
            category="image/pixel art",
            description=(
                "Snaps messy AI-generated pixel art to a perfect grid. Detects the "
                "implicit pixel grid, quantizes colors, and collapses each cell to one "
                "pixel. Port of Sprite Fusion's pixel-snapper by Hugo Duprez."
            ),
            search_aliases=["pixel art", "snap", "grid", "quantize", "sprite"],
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Pixel-art-style image to snap to a perfect grid.",
                ),
                io.Int.Input(
                    "color_count",
                    default=16,
                    min=1,
                    max=256,
                    tooltip=(
                        "Maximum number of colors in the result. The node picks the "
                        "best ones from the image automatically. Classic pixel art "
                        "uses 4-32 colors."
                    ),
                ),
                io.Int.Input(
                    "pixel_size",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip=(
                        "How big one 'art pixel' (one chunky block) is in the input, "
                        "measured in real image pixels. Example: a 1024px-wide image "
                        "that looks like 64 blocks across has 16px blocks, so "
                        "pixel_size = 16. Leave at 0 to measure it automatically. Set "
                        "it manually if auto-detect gets it wrong, or when processing "
                        "a batch/animation so every frame uses the same grid."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional transparency mask (1 = transparent, like the MASK "
                        "output of LoadImage). Transparent areas are ignored during "
                        "analysis and stay transparent in the output."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "snapped_image",
                    display_name="image",
                    tooltip=(
                        "The cleaned-up image at the SAME size as the input. Use this "
                        "for previews and for feeding the rest of your workflow."
                    ),
                ),
                io.Image.Output(
                    "native_image",
                    display_name="native image",
                    tooltip=(
                        "The true pixel-art file: ONE image pixel per art pixel, so "
                        "it is tiny (e.g. 64x64). This is the asset you save for "
                        "games/sprites. It looks like a small stamp in previews - "
                        "that is expected."
                    ),
                ),
                io.Mask.Output(
                    "native_mask",
                    display_name="native mask",
                    tooltip="Transparency for the native image (1 = transparent).",
                ),
                io.Float.Output(
                    "detected_pixel_size",
                    display_name="pixel size",
                    tooltip=(
                        "The measured size of one art pixel in the input image "
                        "(or the value you set)."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, color_count, pixel_size, mask=None):
        batch, height, width, channels = image.shape

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[-2:] != (height, width):
                raise PixelSnapperError(
                    f"Mask size {tuple(mask.shape[-2:])} does not match image size "
                    f"{(height, width)}"
                )

        config = Config(
            k_colors=color_count,
            pixel_size_override=float(pixel_size) if pixel_size > 0 else None,
        )

        natives = []
        detected = 0.0
        for b in range(batch):
            frame = image[b].detach().cpu().clamp(0.0, 1.0)
            rgb = (frame[:, :, :3] * 255.0).round().to(torch.uint8).numpy()

            if mask is not None:
                m = mask[min(b, mask.shape[0] - 1)].detach().cpu().clamp(0.0, 1.0)
                alpha = ((1.0 - m) * 255.0).round().to(torch.uint8).numpy()
            elif channels == 4:
                alpha = (frame[:, :, 3] * 255.0).round().to(torch.uint8).numpy()
            else:
                alpha = np.full((height, width), 255, dtype=np.uint8)

            rgba = np.dstack([rgb, alpha])
            result = snap_pixels(rgba, config)
            natives.append(result.native_rgba)
            if b == 0:
                detected = result.pixel_size

        shapes = {n.shape for n in natives}
        if len(shapes) > 1:
            sizes = ", ".join(f"{s[1]}x{s[0]}" for s in (n.shape for n in natives))
            raise PixelSnapperError(
                f"Auto-detected grids differ across the batch (native sizes: {sizes}). "
                "Set pixel_size explicitly so every frame uses the same grid."
            )

        stacked = np.stack(natives).astype(np.float32) / 255.0  # [B, h, w, 4]
        native_image = torch.from_numpy(stacked[:, :, :, :3])
        native_mask = torch.from_numpy(1.0 - stacked[:, :, :, 3])

        # Nearest-neighbor upscale back to the input size for easy chaining
        full = F.interpolate(
            native_image.permute(0, 3, 1, 2),
            size=(height, width),
            mode="nearest-exact",
        ).permute(0, 2, 3, 1)

        return io.NodeOutput(full, native_image, native_mask, float(detected))


class PixelSnapperExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PixelSnapper]
