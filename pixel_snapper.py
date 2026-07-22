"""Pixel Snapper core algorithm.

A NumPy port of Sprite Fusion's pixel-snapper (https://github.com/Hugo-Dz/spritefusion-pixel-snapper,
MIT, (c) Hugo Duprez). Fixes messy AI-generated pixel art by detecting the implicit
pixel grid and snapping every cell to exactly one pixel.

Pipeline: k-means color quantization -> gradient profiles -> grid step estimation ->
elastic walker cuts -> two-axis stabilization -> majority-vote resampling ->
optional fixed-palette remap.

This module depends only on NumPy so it can be tested outside ComfyUI.
All images here are RGBA uint8 arrays of shape [H, W, 4].
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MAX_PALETTE_COLORS = 256
MAX_IMAGE_DIM = 10000


class PixelSnapperError(Exception):
    pass


@dataclass
class Config:
    # Defaults mirror the Rust implementation exactly.
    k_colors: int = 16
    k_seed: int = 42
    pixel_size_override: Optional[float] = None
    palette: Optional[list] = None  # list of (r, g, b) tuples
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8


@dataclass
class SnapResult:
    native_rgba: np.ndarray  # [out_h, out_w, 4] uint8, one pixel per detected grid cell
    pixel_size: float        # detected (or overridden) grid step in source pixels
    col_cuts: list = field(default_factory=list)
    row_cuts: list = field(default_factory=list)


def parse_palette_hex(value: str) -> list:
    """Parse a comma-separated list of 6-digit hex colors into (r, g, b) tuples."""
    if not value or not value.strip():
        raise PixelSnapperError("Palette must contain at least one color")

    seen = set()
    palette = []
    for part in value.split(","):
        hex_str = part.strip().lstrip("#")
        if len(hex_str) != 6 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
            raise PixelSnapperError(
                f"invalid palette color '{part.strip()}', expected a 6-digit hex code"
            )
        color = (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        if color not in seen:
            seen.add(color)
            palette.append(color)

    if len(palette) > MAX_PALETTE_COLORS:
        raise PixelSnapperError(
            f"Palette must contain at most {MAX_PALETTE_COLORS} distinct colors"
        )
    return palette


def _assign_to_centroids(pixels: np.ndarray, centroids: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """Nearest-centroid index for each pixel. pixels [N,3] float32, centroids [K,3] float32."""
    n = pixels.shape[0]
    out = np.empty(n, dtype=np.int64)
    c_sq = (centroids.astype(np.float64) ** 2).sum(axis=1)
    for s in range(0, n, chunk):
        block = pixels[s:s + chunk].astype(np.float64)
        # ||p-c||^2 = ||p||^2 - 2 p.c + ||c||^2; the ||p||^2 term is constant per row
        scores = c_sq[None, :] - 2.0 * (block @ centroids.astype(np.float64).T)
        out[s:s + chunk] = scores.argmin(axis=1)
    return out


def quantize_image(img: np.ndarray, config: Config) -> np.ndarray:
    """Reduce the image to config.k_colors colors via seeded k-means (transparent pixels untouched)."""
    if config.k_colors <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    alpha = img[:, :, 3]
    opaque_mask = alpha > 0
    opaque_pixels = img[opaque_mask][:, :3].astype(np.float32)
    n_pixels = opaque_pixels.shape[0]
    if n_pixels == 0:
        return img.copy()

    rng = np.random.default_rng(config.k_seed)
    k = min(config.k_colors, n_pixels)

    # k-means++ initialization
    centroids = np.empty((k, 3), dtype=np.float32)
    centroids[0] = opaque_pixels[int(rng.integers(n_pixels))]
    distances = np.full(n_pixels, np.inf, dtype=np.float64)
    for ci in range(1, k):
        last = centroids[ci - 1].astype(np.float64)
        d = ((opaque_pixels.astype(np.float64) - last) ** 2).sum(axis=1)
        np.minimum(distances, d, out=distances)
        total = distances.sum()
        if total <= 0.0:
            idx = int(rng.integers(n_pixels))
        else:
            idx = int(rng.choice(n_pixels, p=distances / total))
        centroids[ci] = opaque_pixels[idx]

    # Lloyd iterations
    prev_centroids = centroids.copy()
    for iteration in range(config.max_kmeans_iterations):
        assign = _assign_to_centroids(opaque_pixels, centroids)
        counts = np.bincount(assign, minlength=k)
        new_centroids = centroids.copy()
        for ch in range(3):
            sums = np.bincount(assign, weights=opaque_pixels[:, ch].astype(np.float64), minlength=k)
            nonzero = counts > 0
            new_centroids[nonzero, ch] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
        centroids = new_centroids

        if iteration > 0:
            movement = ((centroids - prev_centroids) ** 2).sum(axis=1).max()
            if movement < 0.01:
                break
        prev_centroids = centroids.copy()

    # Map every opaque pixel to its nearest centroid (rounded to u8)
    assign = _assign_to_centroids(opaque_pixels, centroids)
    rounded = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
    out = img.copy()
    out_rgb = out[:, :, :3]
    out_rgb[opaque_mask] = rounded[assign]
    return out


def apply_palette(img: np.ndarray, palette: list) -> np.ndarray:
    """Remap every opaque pixel to the nearest color of a fixed palette."""
    if not palette:
        raise PixelSnapperError("Palette must contain at least one RGB color")

    pal = np.asarray(palette, dtype=np.int64)  # [P, 3]
    out = img.copy()
    flat = out.reshape(-1, 4)
    opaque = flat[:, 3] > 0
    if opaque.any():
        colors = flat[opaque][:, :3].astype(np.int64)
        uniq, inverse = np.unique(colors, axis=0, return_inverse=True)
        d = ((uniq[:, None, :] - pal[None, :, :]) ** 2).sum(axis=-1)
        nearest = pal[d.argmin(axis=1)]
        flat[opaque, :3] = nearest[inverse].astype(np.uint8)
    return flat.reshape(img.shape)


def compute_profiles(img: np.ndarray):
    """Per-column / per-row summed absolute gradients of the grayscale image.

    Grid lines show up as peaks: color changes between cells pile up along them.
    """
    h, w = img.shape[:2]
    if w < 3 or h < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    rgb = img[:, :, :3].astype(np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray[img[:, :, 3] == 0] = 0.0

    # central-difference kernel [-1, 0, 1], matching the Rust implementation
    col_proj = np.zeros(w, dtype=np.float64)
    col_proj[1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]).sum(axis=0)
    row_proj = np.zeros(h, dtype=np.float64)
    row_proj[1:-1] = np.abs(gray[2:, :] - gray[:-2, :]).sum(axis=1)
    return col_proj, row_proj


def estimate_step_size(profile: np.ndarray, config: Config) -> Optional[float]:
    """Median distance between strong gradient peaks = the implicit pixel size."""
    if profile.size == 0:
        return None
    max_val = float(profile.max())
    if max_val == 0.0:
        return None
    threshold = max_val * config.peak_threshold_multiplier

    inner = profile[1:-1]
    # >= on the left neighbor so a plateau of equal peaks still yields one
    # peak (its right edge). Perfectly sharp pixel edges produce twin equal
    # peaks that a strict > on both sides would throw away entirely.
    is_peak = (inner > threshold) & (inner >= profile[:-2]) & (inner > profile[2:])
    peaks = (np.nonzero(is_peak)[0] + 1).tolist()
    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > config.peak_distance_filter - 1:
            clean_peaks.append(p)
    if len(clean_peaks) < 2:
        return None

    diffs = sorted(float(b - a) for a, b in zip(clean_peaks, clean_peaks[1:]))
    return diffs[len(diffs) // 2]


def resolve_step_sizes(step_x_opt, step_y_opt, width, height, config: Config):
    """Combine the two axis estimates; borrow from the sibling axis or fall back if needed."""
    if config.pixel_size_override is not None:
        px = config.pixel_size_override
        return px, px

    if step_x_opt is not None and step_y_opt is not None:
        sx, sy = step_x_opt, step_y_opt
        ratio = sx / sy if sx > sy else sy / sx
        if ratio > config.max_step_ratio:
            smaller = min(sx, sy)
            return smaller, smaller
        avg = (sx + sy) / 2.0
        return avg, avg
    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback = max(min(width, height) / config.fallback_target_segments, 1.0)
    return fallback, fallback


def walk(profile: np.ndarray, step_size: float, limit: int, config: Config) -> list:
    """Elastic walker: step by ~step_size, snapping each cut to the strongest nearby gradient."""
    if profile.size == 0:
        raise PixelSnapperError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(profile.sum()) / profile.size

    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break

        start_search = max(int(max(target - search_window, 0.0)), int(current_pos + 1.0))
        end_search = min(int(target + search_window), limit)
        if end_search <= start_search:
            current_pos = target
            continue

        segment = profile[start_search:end_search]
        local = int(segment.argmax())
        max_val = float(segment[local])
        max_idx = start_search + local

        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target
    return cuts


def sanitize_cuts(cuts: list, limit: int) -> list:
    if limit == 0:
        return [0]
    clamped = {min(c, limit) for c in cuts}
    clamped.add(0)
    clamped.add(limit)
    return sorted(clamped)


def snap_uniform_cuts(profile: np.ndarray, limit: int, target_step: float,
                      config: Config, min_required: int) -> list:
    """Lay down near-uniform cuts, still snapping each one to nearby gradient peaks."""
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    if math.isfinite(target_step) and target_step > 0.0:
        desired_cells = int(limit / target_step + 0.5)
    else:
        desired_cells = 0
    desired_cells = min(max(desired_cells, max(min_required - 1, 0), 1), limit)

    cell_width = limit / desired_cells
    search_window = max(cell_width * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(profile.sum()) / profile.size if profile.size else 0.0
    strength_threshold = mean_val * config.walker_strength_threshold
    last_valid = profile.size - 1

    cuts = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break
        start = max(int(math.floor(target - search_window)), prev + 1, 0)
        end = min(int(math.ceil(target + search_window)), limit - 1)
        if end < start:
            start = prev + 1
            end = start

        hi = min(end, last_valid)
        best_idx = min(start, last_valid)
        best_val = -1.0
        if start <= hi:
            segment = profile[start:hi + 1]
            local = int(segment.argmax())
            best_val = float(segment[local])
            best_idx = start + local

        if best_val < strength_threshold:
            fallback_idx = int(target + 0.5)
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx
        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)
    return sanitize_cuts(cuts, limit)


def stabilize_cuts(profile: np.ndarray, cuts: list, limit: int,
                   sibling_cuts: list, sibling_limit: int, config: Config) -> list:
    """If an axis produced too few cuts or is badly skewed vs. its sibling, redo it near-uniformly."""
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = min(max(config.min_cuts_per_axis, 2), limit + 1)
    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = (
        sibling_limit > 0 and sibling_cells >= max(min_required - 1, 0) and sibling_cells > 0
    )
    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / axis_cells
        sibling_step = sibling_limit / sibling_cells
        step_ratio = axis_step / sibling_step
        steps_skewed = step_ratio > config.max_step_ratio or step_ratio < 1.0 / config.max_step_ratio

    if len(cuts) >= min_required and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / sibling_cells
    elif config.fallback_target_segments > 1:
        target_step = limit / config.fallback_target_segments
    elif axis_cells > 0:
        target_step = limit / axis_cells
    else:
        target_step = float(limit)
    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(profile_x, profile_y, raw_col_cuts, raw_row_cuts,
                        width, height, config: Config):
    col_cuts = stabilize_cuts(profile_x, list(raw_col_cuts), width, raw_row_cuts, height, config)
    row_cuts = stabilize_cuts(profile_y, list(raw_row_cuts), height, raw_col_cuts, width, config)

    col_cells = max(len(col_cuts) - 1, 1)
    row_cells = max(len(row_cuts) - 1, 1)
    col_step = width / col_cells
    row_step = height / row_cells
    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            col_cuts = snap_uniform_cuts(profile_x, width, target_step, config, config.min_cuts_per_axis)
        if row_step > target_step * 1.2:
            row_cuts = snap_uniform_cuts(profile_y, height, target_step, config, config.min_cuts_per_axis)

    return col_cuts, row_cuts


def merge_edge_slivers(cuts: list) -> list:
    """Fold degenerate edge cells into their neighbors.

    A perfectly sharp edge between two art pixels yields two equal gradient
    peaks side by side; the walker snaps to the first of the tie, landing
    every cut one pixel early. The accumulated shift leaves a ~1px sliver
    cell at the far edge (e.g. 129 cells for a true 128-cell grid). Any edge
    cell narrower than half the median cell width is merged away.
    """
    if len(cuts) <= 2:
        return cuts
    widths = [b - a for a, b in zip(cuts, cuts[1:])]
    median = sorted(widths)[len(widths) // 2]
    if widths[-1] < median / 2:
        cuts.pop(-2)
    if len(cuts) > 2 and widths[0] < median / 2:
        cuts.pop(1)
    return cuts


def resample(img: np.ndarray, cols: list, rows: list) -> np.ndarray:
    """Collapse each grid cell to one pixel via majority vote (ties -> lowest RGBA value).

    Majority vote (rather than averaging) is what preserves dithering patterns.
    Fully vectorized: pixels are keyed by (cell, color) and counted in one pass.
    """
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    h, w = img.shape[:2]
    ncols = len(cols) - 1
    nrows = len(rows) - 1

    color = (
        (img[:, :, 0].astype(np.uint64) << 24)
        | (img[:, :, 1].astype(np.uint64) << 16)
        | (img[:, :, 2].astype(np.uint64) << 8)
        | img[:, :, 3].astype(np.uint64)
    )
    col_arr = np.asarray(cols)
    row_arr = np.asarray(rows)
    col_of_x = np.searchsorted(col_arr, np.arange(w), side="right") - 1
    row_of_y = np.searchsorted(row_arr, np.arange(h), side="right") - 1
    np.clip(col_of_x, 0, ncols - 1, out=col_of_x)
    np.clip(row_of_y, 0, nrows - 1, out=row_of_y)

    cell = (row_of_y[:, None].astype(np.uint64) * ncols + col_of_x[None, :].astype(np.uint64))
    key = (cell << np.uint64(32)) | color
    uniq, counts = np.unique(key.ravel(), return_counts=True)

    uniq_cells = (uniq >> np.uint64(32)).astype(np.int64)
    uniq_colors = (uniq & np.uint64(0xFFFFFFFF))
    # Sort by (cell, count desc, color asc) and keep the first entry per cell
    order = np.lexsort((uniq_colors, -counts, uniq_cells))
    sorted_cells = uniq_cells[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = sorted_cells[1:] != sorted_cells[:-1]
    winner_cells = sorted_cells[first]
    winner_colors = uniq_colors[order][first]

    flat = np.zeros(nrows * ncols, dtype=np.uint64)
    flat[winner_cells] = winner_colors
    out = np.empty((nrows, ncols, 4), dtype=np.uint8)
    out[:, :, 0] = ((flat >> np.uint64(24)) & np.uint64(255)).reshape(nrows, ncols)
    out[:, :, 1] = ((flat >> np.uint64(16)) & np.uint64(255)).reshape(nrows, ncols)
    out[:, :, 2] = ((flat >> np.uint64(8)) & np.uint64(255)).reshape(nrows, ncols)
    out[:, :, 3] = (flat & np.uint64(255)).reshape(nrows, ncols)
    return out


def snap_pixels(rgba: np.ndarray, config: Optional[Config] = None) -> SnapResult:
    """Run the full pipeline on an RGBA uint8 image of shape [H, W, 4]."""
    if config is None:
        config = Config()

    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise PixelSnapperError("Expected an RGBA uint8 array of shape [H, W, 4]")

    height, width = rgba.shape[:2]
    if width == 0 or height == 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
        raise PixelSnapperError(f"Image dimensions too large (max {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM})")

    if config.pixel_size_override is not None:
        px = config.pixel_size_override
        if not math.isfinite(px) or px < 1.0 or px > min(width, height) / 2.0:
            raise PixelSnapperError(
                f"pixel_size {px:.1f} is out of valid range [1, {min(width, height) // 2}]"
            )

    analysis_img = quantize_image(rgba, config)
    profile_x, profile_y = compute_profiles(analysis_img)

    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(step_x_opt, step_y_opt, width, height, config)

    raw_col_cuts = walk(profile_x, step_x, width, config)
    raw_row_cuts = walk(profile_y, step_y, height, config)

    col_cuts, row_cuts = stabilize_both_axes(
        profile_x, profile_y, raw_col_cuts, raw_row_cuts, width, height, config
    )
    col_cuts = merge_edge_slivers(col_cuts)
    row_cuts = merge_edge_slivers(row_cuts)

    snapped = resample(analysis_img, col_cuts, row_cuts)
    if config.palette:
        snapped = apply_palette(snapped, config.palette)

    return SnapResult(
        native_rgba=snapped,
        pixel_size=step_x,
        col_cuts=col_cuts,
        row_cuts=row_cuts,
    )
