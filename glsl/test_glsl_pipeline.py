"""End-to-end test for the native-GLSL Pixel Snapper pipeline.

Runs against a live ComfyUI instance (URL as argv[1]):
  1. builds synthetic messy pixel art (same recipe as tests/smoke_test.py),
  2. computes ground truth with the Python algorithm,
  3. uploads the image and runs the 3-stage GLSL pipeline via the API,
  4. downloads the outputs and scores them against ground truth.

Run:  uv run --no-project --with numpy,pillow,requests python glsl/test_glsl_pipeline.py <url>
"""

import io
import json
import os
import sys
import time
import uuid

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_snapper import Config, snap_pixels  # noqa: E402

GRID = 24
CELL = 11
JITTER = 2
NOISE = 6
K_COLORS = 8

PALETTE = np.array([
    [13, 43, 69], [32, 60, 86], [84, 78, 104], [141, 105, 122],
    [208, 129, 89], [255, 170, 94], [255, 212, 163], [255, 236, 214],
], dtype=np.uint8)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_test_out")


def make_messy_pixel_art(rng):
    def jittered_cuts(n_cells, cell, jitter):
        cuts = [0]
        for i in range(1, n_cells):
            pos = i * cell + int(rng.integers(-jitter, jitter + 1))
            cuts.append(max(pos, cuts[-1] + 1))
        cuts.append(n_cells * cell)
        return cuts

    true_cells = rng.integers(0, len(PALETTE), size=(GRID, GRID))
    col_cuts = jittered_cuts(GRID, CELL, JITTER)
    row_cuts = jittered_cuts(GRID, CELL, JITTER)
    size = GRID * CELL
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for gy in range(GRID):
        for gx in range(GRID):
            img[row_cuts[gy]:row_cuts[gy + 1], col_cuts[gx]:col_cuts[gx + 1]] = \
                PALETTE[true_cells[gy, gx]]
    noise = rng.integers(-NOISE, NOISE + 1, size=img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, true_cells, col_cuts, row_cuts


def glsl_source(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


def build_prompt(image_name, pixel_size=0):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "10": {"class_type": "GLSLShader", "inputs": {
            "fragment_shader": glsl_source("quantize.glsl"),
            "size_mode": "from_input",
            "images.image0": ["1", 0],
            "images.image1": ["1", 0],
            "ints.u_int0": K_COLORS,
        }},
        "20": {"class_type": "GLSLShader", "inputs": {
            "fragment_shader": glsl_source("detect.glsl"),
            "size_mode": "custom",
            "size_mode.width": 2048,
            "size_mode.height": 8,
            "images.image0": ["10", 0],
            "images.image1": ["10", 0],
            "ints.u_int0": pixel_size,
        }},
        "30": {"class_type": "GLSLShader", "inputs": {
            "fragment_shader": glsl_source("snap.glsl"),
            "size_mode": "from_input",
            "images.image0": ["10", 0],
            "images.image1": ["20", 0],
            "images.image2": ["10", 1],
            "ints.u_int0": K_COLORS,
        }},
        "21": {"class_type": "GLSLShader", "inputs": {
            # debug: row 0 (GL) = params-decode, row 1 = estimator debug,
            # rows 4..7 = the column profile scaled for visibility
            "fragment_shader": (
                "#version 300 es\nprecision highp float;\n"
                "uniform sampler2D u_image0;\nin vec2 v_texCoord;\n"
                "layout(location = 0) out vec4 fragColor0;\n"
                "void main() {\n"
                "  ivec2 f = ivec2(gl_FragCoord.xy);\n"
                "  if (f.y == 0) { vec4 p = texelFetch(u_image0, ivec2(0, 4), 0) * 2048.0; fragColor0 = vec4(p.xyz / 255.0, 1.0); }\n"
                "  else if (f.y == 1) { fragColor0 = vec4(texelFetch(u_image0, ivec2(f.x, 5), 0).rgb, 1.0); }\n"
                "  else if (f.y >= 4) { fragColor0 = vec4(vec3(texelFetch(u_image0, ivec2(f.x, 0), 0).r * 0.008), 1.0); }\n"
                "  else { fragColor0 = vec4(0.0, 0.0, 0.0, 1.0); }\n"
                "}"
            ),
            "size_mode": "custom",
            "size_mode.width": 2048,
            "size_mode.height": 8,
            "images.image0": ["20", 0],
        }},
        "94": {"class_type": "SaveImage", "inputs": {"images": ["21", 0], "filename_prefix": "pxglsl_params"}},
        "91": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "pxglsl_quant"}},
        "92": {"class_type": "SaveImage", "inputs": {"images": ["10", 1], "filename_prefix": "pxglsl_stripes"}},
        "93": {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "pxglsl_snap"}},
    }


def main():
    url = sys.argv[1].rstrip("/")
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)

    img, true_cells, col_cuts, row_cuts = make_messy_pixel_art(rng)
    Image.fromarray(img).save(os.path.join(OUT_DIR, "input.png"))

    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    truth = snap_pixels(rgba, Config(k_colors=K_COLORS))
    print(f"python ground truth: native {truth.native_rgba.shape[1]}x{truth.native_rgba.shape[0]}, "
          f"pixel_size {truth.pixel_size:.2f}")

    # upload
    name = f"pxglsl_{uuid.uuid4().hex[:8]}.png"
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    r = requests.post(f"{url}/api/upload/image",
                      files={"image": (name, buf.getvalue(), "image/png")},
                      data={"type": "input"}, timeout=120)
    r.raise_for_status()
    uploaded = r.json()["name"]
    print("uploaded:", uploaded)

    # run
    r = requests.post(f"{url}/api/prompt", json={"prompt": build_prompt(uploaded)}, timeout=120)
    r.raise_for_status()
    resp = r.json()
    if resp.get("node_errors"):
        print("NODE ERRORS:", json.dumps(resp["node_errors"])[:2000])
        sys.exit(1)
    pid = resp["prompt_id"]

    # poll
    for _ in range(90):
        time.sleep(2)
        h = requests.get(f"{url}/api/history/{pid}", timeout=60).json()
        if pid in h:
            status = h[pid].get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                break
    else:
        print("TIMEOUT waiting for prompt")
        sys.exit(1)

    if status.get("status_str") == "error":
        for m in status.get("messages", []):
            if "error" in m[0]:
                print("EXECUTION ERROR:", str(m[1].get("exception_message", m[1]))[:1500])
        sys.exit(1)

    outputs = h[pid]["outputs"]
    files = {}
    for node_id, out in outputs.items():
        for meta in out.get("images", []):
            r = requests.get(f"{url}/api/view", params={
                "filename": meta["filename"], "subfolder": meta["subfolder"], "type": meta["type"]},
                timeout=120)
            r.raise_for_status()
            arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            prefix = meta["filename"].rsplit("_", 2)[0]
            files[prefix] = arr
            Image.fromarray(arr).save(os.path.join(OUT_DIR, f"{prefix}.png"))

    print("downloaded:", sorted(files))

    params_img = files.get("pxglsl_params")
    if params_img is not None:
        # tensor row 0 = GL row 7 (readback flip): profile lives in tensor rows 0..3
        prof_row = params_img[0, :300, 0].astype(np.int64)
        peaks = [i for i in range(1, 299) if prof_row[i] > 30 and prof_row[i] >= prof_row[i-1] and prof_row[i] > prof_row[i+1]]
        print(f"profile dump: max={prof_row.max()}, nonzero={int((prof_row > 5).sum())}, "
              f"peaks>30: {peaks[:25]}")
        px = params_img[-1, 0].astype(np.int64)     # GL row 0 -> params decode
        print(f"decoded params: step_x~{px[0]}, phase_x~{px[1]}, step_y~{px[2]} "
              f"(expected step ~{CELL})")

    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    # 1. quantized output uses <= K distinct colors (8-bit rounding tolerant)
    quant = files.get("pxglsl_quant")
    n_colors = len(np.unique(quant.reshape(-1, 3), axis=0))
    check("quantized distinct colors <= K", n_colors <= K_COLORS, f"got {n_colors}")

    # 2. stripes contain ~K distinct colors
    stripes = files.get("pxglsl_stripes")
    n_stripe_colors = len(np.unique(stripes.reshape(-1, 3), axis=0))
    check("palette stripes ~K colors", K_COLORS - 2 <= n_stripe_colors <= K_COLORS + 2,
          f"got {n_stripe_colors}")

    # 3. snapped output: cells constant + correct vs truth at cell centers
    snap = files.get("pxglsl_snap")
    centers_ok = 0
    total = 0
    for gy in range(GRID):
        for gx in range(GRID):
            cx = (col_cuts[gx] + col_cuts[gx + 1]) // 2
            cy = (row_cuts[gy] + row_cuts[gy + 1]) // 2
            got = snap[cy, cx].astype(np.int64)
            want = PALETTE[true_cells[gy, gx]].astype(np.int64)
            d = ((PALETTE.astype(np.int64) - got) ** 2).sum(axis=1)
            if d.argmin() == true_cells[gy, gx]:
                centers_ok += 1
            total += 1
    acc = centers_ok / total
    check("snapped cell-center accuracy >= 85%", acc >= 0.85, f"accuracy {acc:.1%}")

    # 4. snapped output distinct colors <= K
    n_snap_colors = len(np.unique(snap.reshape(-1, 3), axis=0))
    check("snapped distinct colors <= K", n_snap_colors <= K_COLORS, f"got {n_snap_colors}")

    # 5. explicit pixel_size override path
    r = requests.post(f"{url}/api/prompt",
                      json={"prompt": build_prompt(uploaded, pixel_size=CELL)}, timeout=120)
    r.raise_for_status()
    pid2 = r.json()["prompt_id"]
    for _ in range(90):
        time.sleep(2)
        h2 = requests.get(f"{url}/api/history/{pid2}", timeout=60).json()
        if pid2 in h2 and (h2[pid2].get("status", {}).get("completed")
                           or h2[pid2]["status"].get("status_str") == "error"):
            break
    snap2 = None
    for node_id, out in h2[pid2].get("outputs", {}).items():
        for meta in out.get("images", []):
            if meta["filename"].startswith("pxglsl_snap"):
                r = requests.get(f"{url}/api/view", params={
                    "filename": meta["filename"], "subfolder": meta["subfolder"],
                    "type": meta["type"]}, timeout=120)
                snap2 = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    if snap2 is not None:
        ok2 = sum(
            ((PALETTE.astype(np.int64) - snap2[(row_cuts[gy] + row_cuts[gy + 1]) // 2,
                                               (col_cuts[gx] + col_cuts[gx + 1]) // 2]
              .astype(np.int64)) ** 2).sum(axis=1).argmin() == true_cells[gy, gx]
            for gy in range(GRID) for gx in range(GRID))
        acc2 = ok2 / (GRID * GRID)
        check("override pixel_size accuracy >= 85%", acc2 >= 0.85, f"accuracy {acc2:.1%}")
    else:
        check("override pixel_size run produced output", False)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("All GLSL pipeline checks passed.")


if __name__ == "__main__":
    main()
