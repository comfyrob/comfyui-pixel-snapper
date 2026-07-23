# GPU regression harnesses for the sprite pipeline shaders

Each script runs against a live ComfyUI instance via the API (the URL is
hardcoded to the rtxpro6000-b test slot; pass/adjust as needed):

```bash
cd /Users/rob/comfyvibe/projects/modal && modal deploy comfyui_webui.py::app_rtxpro6000_b
uv run --no-project --with numpy,pillow,requests python glsl/tests/<script>.py
modal app stop --yes comfyui-rtxpro6000-b   # ALWAYS stop the GPU after
```

| Script | Guards against |
|---|---|
| test_smartcrop.py | midline-crossing sword truncation (smartcrop, unused by V5+) |
| test_align.py | frame jitter on the real generated webps (feet/center spread) |
| test_anchor.py | body sway from bbox-centering (feet-band anchor) |
| test_zoom_chain.py | full zoom->crop->align chain, uniform scale |
| test_ground.py | grounded fills + arc preservation + lunge asymmetry (V8/V9 regressions) |
| test_lockoff.py | lock-off: raised sword across seam + apex altitude clipping |
| test_2k_chain.py | aspect/size agnosticism at 2560x1440 (2K 16:9) |
| test_alpha_chain.py | transparent pipeline: BiRefNet -> crisp -> RGBA webp, binary alpha, mask bleed |

Each builds synthetic sheets encoding a previously-shipped bug; PASS/FAIL
printed at the end. When a shader changes, run the scripts covering it.
