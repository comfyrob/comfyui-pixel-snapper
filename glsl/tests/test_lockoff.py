"""Lock-off test: raised sword across the horizontal seam + floating apex char."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
sh = {n: open(f"{REPO}/glsl/{n}.glsl").read() for n in ["contentzoom", "quantize", "detect", "snap", "align"]}

W, H = 1376, 768
AP = 8
rng = np.random.default_rng(9)
sheet = np.full((H, W, 3), 255, np.uint8)
PAIRS = [((180,30,30),(240,120,120)), ((30,140,30),(120,220,120)),
         ((30,30,180),(120,120,240)), ((150,30,150),(230,120,230))]
def draw(x0, y0, wpx, hpx, pair):
    for gy in range(hpx):
        for gx in range(wpx):
            sheet[y0+gy*AP:y0+(gy+1)*AP, x0+gx*AP:x0+(gx+1)*AP] = pair[int(rng.integers(0,2))]

draw(240, 64, 14, 24, PAIRS[0])                    # TL: y 64..256 (high in cell)
draw(960, 96, 14, 24, PAIRS[1])                    # TR: floating apex y 96..288
draw(280, 512, 14, 28, PAIRS[2]); draw(320, 304, 3, 26, PAIRS[2])  # BL + sword UP y304..512 crossing y=384
draw(1000, 520, 14, 26, PAIRS[3])                  # BR grounded
buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("lockoff.png", buf.getvalue(), "image/png")},
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
            if meta.get("subfolder") == "lockoff" and meta["filename"].startswith(prefix):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                yield np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))

p1 = {
  "1": {"class_type": "LoadImage", "inputs": {"image": up}},
  "2": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["contentzoom"], "size_mode": "from_input",
        "images.image0": ["1", 0], "images.image1": ["1", 0], "bools.u_bool0": False, "floats.u_float0": 0.90}},
  "3": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["quantize"], "size_mode": "from_input",
        "images.image0": ["2", 0], "images.image1": ["2", 0], "ints.u_int0": 12}},
  "4": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["detect"], "size_mode": "custom",
        "size_mode.width": 4096, "size_mode.height": 8,
        "images.image0": ["3", 0], "images.image1": ["3", 0], "ints.u_int0": 0}},
  "5": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["snap"], "size_mode": "from_input",
        "images.image0": ["3", 0], "images.image1": ["4", 0], "images.image2": ["3", 1], "ints.u_int0": 12}},
  "90": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "lockoff/snapped"}},
}
snapped = list(fetch(run(p1), "snapped"))[0]
hh, hw = snapped.shape[0]//2, snapped.shape[1]//2
frames = [snapped[:hh, :hw], snapped[:hh, hw:], snapped[hh:, :hw], snapped[hh:, hw:]]
ups = []
for i, fr in enumerate(frames):
    b = iolib.BytesIO(); Image.fromarray(fr).save(b, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"lockofff{i}.png", b.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=120)
    r.raise_for_status(); ups.append(r.json()["name"])
p2 = {str(10+i): {"class_type": "LoadImage", "inputs": {"image": n}} for i, n in enumerate(ups)}
p2["20"] = {"class_type": "BatchImagesNode", "inputs": {
    "images.image0": ["10", 0], "images.image1": ["11", 0],
    "images.image2": ["12", 0], "images.image3": ["13", 0]}}
p2["30"] = {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["align"], "size_mode": "from_input",
    "images.image0": ["20", 0], "images.image1": ["20", 0], "bools.u_bool0": False, "floats.u_float0": 0.06}}
p2["91"] = {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "lockoff/final"}}
finals = list(fetch(run(p2), "final"))

ok = True
hts = []
for i, a in enumerate(finals):
    a16 = a.astype(np.int16)
    present = set()
    for ci, pair in enumerate(PAIRS):
        for c in pair:
            if int((np.abs(a16 - np.array(c)).max(axis=2) < 40).sum()) > 400: present.add(ci)
    fg = (255 - a16).max(axis=2) > 20
    ys, xs = np.nonzero(fg)
    edges = int(fg[:3,:].sum() + fg[-3:,:].sum() + fg[:,:3].sum() + fg[:,-3:].sum())
    hts.append(ys.max() - ys.min() + 1)
    print(f"frame {i}: chars {sorted(present)}, content y[{ys.min()},{ys.max()}] of {a.shape[0]}, edge_px={edges}")
    if sorted(present) != [i] or edges > 30: ok = False
# frame 2 (BL) must include the full sword: its content ~ (512-304+224)/224 = ~1.9x frame 3's
ratio = hts[2] / hts[3]
print(f"frame2/frame3 content-height ratio: {ratio:.2f} (sword intact if ~1.7-2.1; ~1.1 = amputated)")
if not (1.5 < ratio < 2.4): ok = False
print("RESULT:", "PASS" if ok else "FAIL")
