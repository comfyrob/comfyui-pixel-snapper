"""Feet-anchor test: asymmetric swords must not sway the body."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
sh = {n: open(f"{REPO}/glsl/{n}.glsl").read() for n in ["contentzoom", "quantize", "detect", "snap", "align"]}

W, H = 1376, 768   # their actual NB2 16:9 sheet size
AP = 8
rng = np.random.default_rng(5)
sheet = np.full((H, W, 3), 255, np.uint8)
PAIRS = [((180,30,30),(240,120,120)), ((30,140,30),(120,220,120)),
         ((30,30,180),(120,120,240)), ((150,30,150),(230,120,230))]

def draw(x0, y0, wpx, hpx, pair):
    for gy in range(hpx):
        for gx in range(wpx):
            sheet[y0+gy*AP:y0+(gy+1)*AP, x0+gx*AP:x0+(gx+1)*AP] = pair[int(rng.integers(0,2))]

# bodies (feet at bottom), cells 688x384
draw(280, 90, 14, 30, PAIRS[0])                 # f1 body 112x240
draw(970, 90, 14, 30, PAIRS[1]); draw(1082, 130, 22, 3, PAIRS[1])   # f2: sword RIGHT 176px
draw(300, 480, 14, 30, PAIRS[2]); draw(124, 520, 22, 3, PAIRS[2])   # f3: sword LEFT 176px
draw(990, 480, 14, 30, PAIRS[3])                # f4 body
buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("anch.png", buf.getvalue(), "image/png")},
                  data={"type": "input", "overwrite": "true"}, timeout=180)
r.raise_for_status(); up = r.json()["name"]

def run(prompt):
    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=180)
    resp = r.json()
    if r.status_code != 200 or resp.get("node_errors"):
        print("SUBMIT:", json.dumps(resp)[:1000]); sys.exit(1)
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
            if meta.get("subfolder") == "anch" and meta["filename"].startswith(prefix):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                yield np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))

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
  "90": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "anch/snapped"}},
}
snapped = list(fetch(run(p1), "snapped"))[0]
hh, hw = snapped.shape[0]//2, snapped.shape[1]//2
frames = [snapped[:hh, :hw], snapped[:hh, hw:], snapped[hh:, :hw], snapped[hh:, hw:]]
ups = []
for i, fr in enumerate(frames):
    b = iolib.BytesIO(); Image.fromarray(fr).save(b, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"anchf{i}.png", b.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=120)
    r.raise_for_status(); ups.append(r.json()["name"])
p2 = {str(10+i): {"class_type": "LoadImage", "inputs": {"image": n}} for i, n in enumerate(ups)}
p2["20"] = {"class_type": "BatchImagesNode", "inputs": {
    "images.image0": ["10", 0], "images.image1": ["11", 0],
    "images.image2": ["12", 0], "images.image3": ["13", 0]}}
p2["30"] = {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["align"], "size_mode": "from_input",
    "images.image0": ["20", 0], "images.image1": ["20", 0], "bools.u_bool0": True, "floats.u_float0": 0.06}}
p2["91"] = {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "anch/final"}}
finals = list(fetch(run(p2), "final"))

feet_cx, feet_y, clip = [], [], []
for i, a in enumerate(finals):
    a16 = a.astype(np.int16)
    fg = (255 - a16).max(axis=2) > 20
    ys, xs = np.nonzero(fg)
    top, bot = ys.min(), ys.max()
    band = ys >= bot - int(0.18*(bot-top))
    feet_cx.append(int(xs[band].mean())); feet_y.append(int(bot))
    borders = int(fg[:, :3].sum() + fg[:, -3:].sum() + fg[:3, :].sum() + fg[-3:, :].sum())
    clip.append(borders)
    print(f"frame {i}: feet_cx={feet_cx[-1]} feet_y={feet_y[-1]} bbox_w={xs.max()-xs.min()+1} border_px={borders}")
print("feet_cx spread:", max(feet_cx)-min(feet_cx), "| feet_y spread:", max(feet_y)-min(feet_y),
      "| clipping:", sum(clip))
print("RESULT:", "PASS" if max(feet_cx)-min(feet_cx) <= 3 and max(feet_y)-min(feet_y) <= 2 and sum(clip) < 50 else "CHECK")
