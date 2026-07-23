"""Grid-agnostic pipeline: 3x3 and 4x2 sheets through zoom -> snap -> native
split (GetImageSize + math + crop + SplitImageToTileList + RebatchImages)
-> align -> webp. Also exercises the ternary layout-index math expression
and uneven division (1024/3)."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image, ImageSequence

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
sh = {n: open(f"{REPO}/glsl/{n}.glsl").read() for n in ["contentzoom", "quantize", "detect", "snap", "align"]}

COLORS = [(180,30,30),(30,140,30),(30,30,180),(150,30,150),
          (200,120,30),(30,150,150),(0,0,0),(90,90,30),(30,60,90)]

def make_sheet(W, H, C, R, sword_cell=None):
    AP = 8
    rng = np.random.default_rng(21)
    sheet = np.full((H, W, 3), 255, np.uint8)
    cw, ch = W // C, H // R
    for r in range(R):
        for c in range(C):
            idx = r * C + c
            base = COLORS[idx]
            pair = (base, tuple(min(255, v + 70) for v in base))
            wpx = max(6, int(cw * 0.35) // AP)
            hpx = max(8, int(ch * 0.55) // AP)
            x0 = c * cw + (cw - wpx * AP) // 2 + int(rng.integers(-10, 11))
            y0 = (r + 1) * ch - hpx * AP - max(20, ch // 12)  # feet margin
            for gy in range(hpx):
                for gx in range(wpx):
                    col = pair[int(rng.integers(0, 2))]
                    sheet[y0+gy*AP:y0+(gy+1)*AP, x0+gx*AP:x0+(gx+1)*AP] = col
            if sword_cell == idx and r > 0:
                # raised sword crossing the horizontal boundary above
                sx = x0 + wpx * AP // 2
                sheet[r*ch - 60:y0 + 8, sx:sx+16] = base
    return sheet

def run(prompt):
    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=180)
    resp = r.json()
    if r.status_code != 200 or resp.get("node_errors"):
        print("SUBMIT:", json.dumps(resp)[:1400]); sys.exit(1)
    pid = resp["prompt_id"]
    for _ in range(120):
        time.sleep(2)
        h = requests.get(f"{URL}/api/history/{pid}", timeout=60).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "error": break
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if "error" in m[0]: print("ERR:", str(m[1].get("exception_message", m[1]))[:900])
        sys.exit(1)
    return h[pid]["outputs"]

def chain(tag, sheet, C, R, layout_index):
    buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"grid_{tag}.png", buf.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=180)
    r.raise_for_status(); up = r.json()["name"]
    p = {
      "1": {"class_type": "LoadImage", "inputs": {"image": up}},
      # layout-index -> cols/rows via ternary math (the dropdown-INDEX trick)
      "mc": {"class_type": "ComfyMathExpression", "inputs": {
             "expression": "2 if a == 0 else (3 if a == 1 else 4)", "values.a": layout_index}},
      "mr": {"class_type": "ComfyMathExpression", "inputs": {
             "expression": "2 if a == 0 else (3 if a == 1 else 2)", "values.a": layout_index}},
      "2": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["contentzoom"], "size_mode": "from_input",
            "images.image0": ["1", 0], "images.image1": ["1", 0],
            "ints.u_int0": ["mc", 1], "ints.u_int1": ["mr", 1],
            "bools.u_bool0": True, "floats.u_float0": 0.90}},
      "3": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["quantize"], "size_mode": "from_input",
            "images.image0": ["2", 0], "images.image1": ["2", 0], "ints.u_int0": 24}},
      "4": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["detect"], "size_mode": "custom",
            "size_mode.width": 4096, "size_mode.height": 8,
            "images.image0": ["3", 0], "images.image1": ["3", 0], "ints.u_int0": 0}},
      "5": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["snap"], "size_mode": "from_input",
            "images.image0": ["3", 0], "images.image1": ["4", 0], "images.image2": ["3", 1], "ints.u_int0": 24}},
      # native splitter: divisible-crop then tile-split then rebatch
      "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
      "7": {"class_type": "ComfyMathExpression", "inputs": {
            "expression": "max(1, int(a / b))", "values.a": ["6", 0], "values.b": ["mc", 1]}},
      "8": {"class_type": "ComfyMathExpression", "inputs": {
            "expression": "max(1, int(a / b))", "values.a": ["6", 1], "values.b": ["mr", 1]}},
      "9": {"class_type": "ComfyMathExpression", "inputs": {
            "expression": "a * b", "values.a": ["7", 1], "values.b": ["mc", 1]}},
      "10": {"class_type": "ComfyMathExpression", "inputs": {
            "expression": "a * b", "values.a": ["8", 1], "values.b": ["mr", 1]}},
      "11": {"class_type": "PrimitiveBoundingBox", "inputs": {
             "x": 0, "y": 0, "width": ["9", 1], "height": ["10", 1]}},
      "12": {"class_type": "ImageCropV2", "inputs": {"image": ["5", 0], "crop_region": ["11", 0]}},
      "13": {"class_type": "SplitImageToTileList", "inputs": {
             "image": ["12", 0], "tile_width": ["7", 1], "tile_height": ["8", 1], "overlap": 0}},
      "14": {"class_type": "RebatchImages", "inputs": {"images": ["13", 0], "batch_size": 16}},
      "15": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["align"], "size_mode": "from_input",
             "images.image0": ["14", 0], "images.image1": ["14", 0],
             "bools.u_bool0": True, "floats.u_float0": 0.06}},
      "90": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["15", 0], "filename_prefix": f"grid/{tag}",
             "fps": 8, "lossless": True, "quality": 90, "method": "default"}},
      "91": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": f"grid/{tag}f"}},
      "92": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": f"grid/{tag}zoom"}},
      "93": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": f"grid/{tag}snap"}},
    }
    outs = run(p)
    webp = None
    pngs = 0
    for node_id, out in outs.items():
        for meta in out.get("images", []):
            if meta.get("subfolder") != "grid": continue
            if meta["filename"].startswith(tag + "f"):
                pngs += 1
            elif meta["filename"].startswith(tag):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                webp = Image.open(iolib.BytesIO(r.content))
    print(f"  ({tag}: {pngs} aligned frame PNGs reached the save node)")
    return webp

ok = True
for tag, W, H, C, R, li, sword in [("g3x3", 1024, 1024, 3, 3, 1, 4), ("g4x2", 1376, 768, 4, 2, 2, 5)]:
    sheet = make_sheet(W, H, C, R, sword_cell=sword)
    webp = chain(tag, sheet, C, R, li)
    n = getattr(webp, "n_frames", 1)
    print(f"{tag}: webp {webp.size}, frames={n} (expect {C*R})")
    if n != C * R: ok = False; continue
    feet = []
    for i, f in enumerate(ImageSequence.Iterator(webp)):
        a = np.array(f.convert("RGB")).astype(np.int16)
        fg = (255 - a).max(axis=2) > 20
        ys, xs = np.nonzero(fg)
        feet.append(int(ys.max()))
        # frame i should be dominated by COLORS[i]: assign fg px to nearest base color
        fgpx = a[fg].astype(np.int64)
        pal = np.array(COLORS, dtype=np.int64)
        pal_all = np.vstack([pal, np.minimum(pal + 70, 255)])  # base + bright shades
        d = ((fgpx[:, None, :] - pal_all[None, :, :]) ** 2).sum(-1)
        owner_hist = np.bincount(d.argmin(1) % len(pal), minlength=len(pal))
        m = int(owner_hist[i]); others = int(np.delete(owner_hist, i).max())
        edge = int(fg[:3,:].sum() + fg[-3:,:].sum() + fg[:,:3].sum() + fg[:,-3:].sum())
        fillh = 100*(ys.max()-ys.min()+1)//a.shape[0]
        status = "ok" if m > 300 and m > others * 3 and edge < 30 else "BAD"
        if status == "BAD": ok = False
        print(f"  f{i}: own-color px {m}, max-other {others}, fillH {fillh}%, edge {edge} [{status}]")
    print(f"  feet spread: {max(feet)-min(feet)}")
    if max(feet) - min(feet) > 2: ok = False
print("RESULT:", "PASS" if ok else "FAIL")
