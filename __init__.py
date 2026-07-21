"""ComfyUI Pixel Snapper — snap messy AI-generated pixel art to a perfect grid.

Algorithm ported from Sprite Fusion's pixel-snapper by Hugo Duprez (MIT):
https://github.com/Hugo-Dz/spritefusion-pixel-snapper
"""

from .nodes import PixelSnapperExtension


async def comfy_entrypoint() -> PixelSnapperExtension:
    return PixelSnapperExtension()
