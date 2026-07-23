"""Aspect/size-agnostic validation: 2560x1440 sheet through zoom -> snap stages -> align."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
sh = {n: open(f"{REPO}/glsl/{n}.glsl").read() for n in ["contentzoom", "quantize", "detect", "snap", "align"]}

W, H = 2560, 1440
AP = 16  # art pixel size
sheet = np.full((H, W, 3), 255, np.uint8)
PAIRS = [((180, 30, 30), (240, 120, 120)), ((30, 140, 30), (120, 220, 120)),
         ((30, 30, 180), (120, 120, 240)), ((150, 30, 150), (230, 120, 230))]
rng = np.random.default_rng(3)

def draw_char(x0, y0, wpx, hpx, pair):
    for gy in range(hpx):
        for gx in range(wpx):
            c = pair[int(rng.integers(0, 2))]
            sheet[y0 + gy*AP:y0 + (gy+1)*AP, x0 + gx*AP:x0 + (gx+1)*AP] = c

# chars in cells (cells 1280x720), jittered positions, varied sizes
draw_char(430, 180, 22, 26, PAIRS[0])    # TL: 352x416
draw_char(1780, 140, 20, 28, PAIRS[1])   # TR: 320x448
draw_char(380, 900, 26, 28, PAIRS[2])    # BL: 416x448 (largest)
draw_char(1860, 940, 18, 24, PAIRS[3])   # BR: 288x384
buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("t2k.png", buf.getvalue(), "image/png")},
                  data={"type": "input", "overwrite": "true"}, timeout=180)
r.raise_for_status(); up = r.json()["name"]

def run(prompt):
    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=180)
    resp = r.json()
    if r.status_code != 200 or resp.get("node_errors"):
        print("SUBMIT:", json.dumps(resp)[:1200]); sys.exit(1)
    pid = resp["prompt_id"]
    for _ in range(90):
        time.sleep(2)
        h = requests.get(f"{URL}/api/history/{pid}", timeout=60).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "error": break
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if "error" in m[0]: print("ERR:", str(m[1].get("exception_message", m[1]))[:700])
        sys.exit(1)
    return h[pid]["outputs"]

def fetch(outputs, prefix):
    for node_id, out in outputs.items():
        for meta in out.get("images", []):
            if meta.get("subfolder") == "t2k" and meta["filename"].startswith(prefix):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                yield np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))

# stage 1: zoom -> quantize -> detect -> snap
p1 = {
  "1": {"class_type": "LoadImage", "inputs": {"image": up}},
  "2": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["contentzoom"], "size_mode": "from_input",
        "images.image0": ["1", 0], "images.image1": ["1", 0], "bools.u_bool0": True, "floats.u_float0": 0.90}},
  "3": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["quantize"], "size_mode": "from_input",
        "images.image0": ["2", 0], "images.image1": ["2", 0], "ints.u_int0": 12}},
  "4": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["detect"], "size_mode": "custom",
        "size_mode.width": 4096, "size_mode.height": 8,
        "images.image0": ["3", 0], "images.image1": ["3", 0], "ints.u_int0": 0}},
  "5": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["snap"], "size_mode": "from_input",
        "images.image0": ["3", 0], "images.image1": ["4", 0], "images.image2": ["3", 1], "ints.u_int0": 12}},
  "90": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "t2k/snapped"}},
}
snapped = list(fetch(run(p1), "snapped"))[0]
print("snapped sheet:", snapped.shape)

# split at exact halves (what the Crop Images 2x2 blueprint does), reading order
hh, hw = snapped.shape[0] // 2, snapped.shape[1] // 2
frames = [snapped[:hh, :hw], snapped[:hh, hw:], snapped[hh:, :hw], snapped[hh:, hw:]]
ups = []
for i, fr in enumerate(frames):
    b = iolib.BytesIO(); Image.fromarray(fr).save(b, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"t2kf{i}.png", b.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=120)
    r.raise_for_status(); ups.append(r.json()["name"])

p2 = {str(10+i): {"class_type": "LoadImage", "inputs": {"image": n}} for i, n in enumerate(ups)}
p2["20"] = {"class_type": "BatchImagesNode", "inputs": {
    "images.image0": ["10", 0], "images.image1": ["11", 0],
    "images.image2": ["12", 0], "images.image3": ["13", 0]}}
p2["30"] = {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["align"], "size_mode": "from_input",
    "images.image0": ["20", 0], "images.image1": ["20", 0], "bools.u_bool0": True, "floats.u_float0": 0.06}}
p2["91"] = {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "t2k/final"}}
finals = list(fetch(run(p2), "final"))
print("final frames:", len(finals), finals[0].shape)

ok = True
feet, cxs = [], []
for i, a in enumerate(finals):
    a16 = a.astype(np.int16)
    present = set()
    for ci, pair in enumerate(PAIRS):
        for c in pair:
            if int((np.abs(a16 - np.array(c)).max(axis=2) < 40).sum()) > 500: present.add(ci)
    fgm = (255 - a16).max(axis=2) > 20
    ys, xs = np.nonzero(fgm)
    feet.append(int(ys.max())); cxs.append(int((xs.min()+xs.max())//2))
    hfill = (ys.max()-ys.min()+1)/a.shape[0]
    print(f"frame {i}: chars {sorted(present)}, fill {hfill:.0%}h, feet {ys.max()}, cx {cxs[-1]}")
    if sorted(present) != [i]: ok = False
print("feet spread:", max(feet)-min(feet), "| cx spread:", max(cxs)-min(cxs))
print("RESULT:", "PASS" if ok and max(feet)-min(feet) <= 2 and max(cxs)-min(cxs) <= 2 else "CHECK")
