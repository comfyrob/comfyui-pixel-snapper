"""Smoke test for the Pixel Snapper algorithm. Needs only numpy — no ComfyUI, no torch.

Run from the repo root:  python tests/smoke_test.py

Builds synthetic "messy" pixel art (jittered grid + color noise, the way AI
generations misbehave), then checks that the algorithm recovers the true grid
and colors.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixel_snapper import (  # noqa: E402
    Config,
    PixelSnapperError,
    parse_palette_hex,
    snap_pixels,
)

GRID = 24          # true cells per axis
CELL = 11          # true pixel size
JITTER = 2         # max boundary wobble in px
NOISE = 6          # max per-channel color noise

PALETTE = np.array([
    [13, 43, 69],
    [32, 60, 86],
    [84, 78, 104],
    [141, 105, 122],
    [208, 129, 89],
    [255, 170, 94],
    [255, 212, 163],
    [255, 236, 214],
], dtype=np.uint8)


def jittered_cuts(rng, n_cells, cell, jitter):
    cuts = [0]
    for i in range(1, n_cells):
        pos = i * cell + int(rng.integers(-jitter, jitter + 1))
        cuts.append(max(pos, cuts[-1] + 1))
    cuts.append(n_cells * cell)
    return cuts


def make_messy_pixel_art(rng, transparent_corner=False):
    """A GRIDxGRID sprite rendered at ~CELL px per cell with wobbly boundaries and noise."""
    true_cells = rng.integers(0, len(PALETTE), size=(GRID, GRID))
    col_cuts = jittered_cuts(rng, GRID, CELL, JITTER)
    row_cuts = jittered_cuts(rng, GRID, CELL, JITTER)
    size = GRID * CELL

    img = np.zeros((size, size, 4), dtype=np.uint8)
    for gy in range(GRID):
        for gx in range(GRID):
            color = PALETTE[true_cells[gy, gx]]
            img[row_cuts[gy]:row_cuts[gy + 1], col_cuts[gx]:col_cuts[gx + 1], :3] = color
    noise = rng.integers(-NOISE, NOISE + 1, size=(size, size, 3))
    img[:, :, :3] = np.clip(img[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[:, :, 3] = 255

    if transparent_corner:
        img[: 4 * CELL, : 4 * CELL, 3] = 0
        true_cells[:4, :4] = -1  # sentinel: transparent

    return img, true_cells


def cell_accuracy(native, true_cells):
    """Fraction of cells whose snapped color is closest to the correct palette entry."""
    assert native.shape[:2] == true_cells.shape
    correct = 0
    total = 0
    for gy in range(true_cells.shape[0]):
        for gx in range(true_cells.shape[1]):
            truth = true_cells[gy, gx]
            if truth < 0:
                if native[gy, gx, 3] == 0:
                    correct += 1
                total += 1
                continue
            d = ((PALETTE.astype(np.int64) - native[gy, gx, :3].astype(np.int64)) ** 2).sum(axis=1)
            if d.argmin() == truth:
                correct += 1
            total += 1
    return correct / total


def main():
    rng = np.random.default_rng(7)
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # --- 1. auto-detect on messy art -------------------------------------
    img, true_cells = make_messy_pixel_art(rng)
    result = snap_pixels(img, Config(k_colors=8))
    h, w = result.native_rgba.shape[:2]
    check("auto-detect native size ~ 24x24", abs(h - GRID) <= 3 and abs(w - GRID) <= 3,
          f"got {w}x{h}, pixel_size={result.pixel_size:.2f}")
    check("auto-detect pixel size ~ 11", abs(result.pixel_size - CELL) <= 2.0,
          f"got {result.pixel_size:.2f}")
    if (h, w) == (GRID, GRID):
        acc = cell_accuracy(result.native_rgba, true_cells)
        check("cell color accuracy >= 85%", acc >= 0.85, f"accuracy={acc:.1%}")
    colors = np.unique(result.native_rgba.reshape(-1, 4)[:, :3], axis=0)
    check("quantized to <= 8 colors", len(colors) <= 8, f"got {len(colors)} colors")

    # --- 2. explicit pixel_size override ---------------------------------
    result2 = snap_pixels(img, Config(k_colors=8, pixel_size_override=float(CELL)))
    h2, w2 = result2.native_rgba.shape[:2]
    check("override native size ~ 24x24", abs(h2 - GRID) <= 2 and abs(w2 - GRID) <= 2,
          f"got {w2}x{h2}")
    check("override pixel size reported", result2.pixel_size == float(CELL))

    # --- 3. fixed palette remap ------------------------------------------
    hex_palette = "000000,ff0000,00ff00,0000ff"
    pal = parse_palette_hex(hex_palette)
    result3 = snap_pixels(img, Config(k_colors=8, palette=pal))
    out_colors = {tuple(c) for c in result3.native_rgba.reshape(-1, 4)[:, :3]}
    allowed = {tuple(c) for c in pal}
    check("palette remap only uses palette colors", out_colors <= allowed,
          f"extra colors: {out_colors - allowed}")

    # --- 4. transparency preserved ---------------------------------------
    img_t, true_cells_t = make_messy_pixel_art(rng, transparent_corner=True)
    result4 = snap_pixels(img_t, Config(k_colors=8, pixel_size_override=float(CELL)))
    h4, w4 = result4.native_rgba.shape[:2]
    # The strong gradient at the transparency boundary can pull the elastic
    # walker's cuts, so allow the same +-2 cell tolerance as the other tests.
    check("transparency test grid size ~ 24x24", abs(h4 - GRID) <= 2 and abs(w4 - GRID) <= 2,
          f"got {w4}x{h4}")
    if abs(h4 - GRID) <= 2 and abs(w4 - GRID) <= 2:
        # The transparent block spans 4x4 true cells; its interior 3x3 stays
        # inside the block even if the recovered grid shifts by a cell.
        corner_alpha = result4.native_rgba[:3, :3, 3]
        check("transparent corner stays transparent", (corner_alpha == 0).all(),
              f"alphas={np.unique(corner_alpha)}")

    # --- 5. validation errors --------------------------------------------
    for bad in ["", "zzz", "12345", "0d2b45,xyz123"]:
        try:
            parse_palette_hex(bad)
            check(f"palette '{bad}' rejected", False)
        except PixelSnapperError:
            check(f"palette '{bad}' rejected", True)

    try:
        snap_pixels(img, Config(pixel_size_override=100000.0))
        check("huge pixel_size rejected", False)
    except PixelSnapperError:
        check("huge pixel_size rejected", True)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
