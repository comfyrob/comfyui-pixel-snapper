"""Transparent pipeline test: snap -> BiRefNet -> crisp -> frames -> align -> RGBA webp."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image, ImageSequence

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
sh = {n: open(f"{REPO}/glsl/{n}.glsl").read() for n in ["contentzoom", "quantize", "detect", "snap", "align", "alphacrisp"]}

# knight-ish synthetic on white, 1024x1024
W, H, AP = 1024, 1024, 8
rng = np.random.default_rng(13)
sheet = np.full((H, W, 3), 255, np.uint8)
PAIRS = [((120,60,20),(180,120,60)), ((60,80,140),(120,150,210)),
         ((40,100,40),(90,170,90)), ((130,40,110),(200,110,180))]
def draw(x0, y0, wpx, hpx, pair):
    for gy in range(hpx):
        for gx in range(wpx):
            sheet[y0+gy*AP:y0+(gy+1)*AP, x0+gx*AP:x0+(gx+1)*AP] = pair[int(rng.integers(0,2))]
draw(180, 100, 18, 34, PAIRS[0]); draw(700, 108, 17, 33, PAIRS[1])
draw(190, 620, 18, 33, PAIRS[2]); draw(690, 612, 17, 34, PAIRS[3])
buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("alpha_t.png", buf.getvalue(), "image/png")},
                  data={"type": "input", "overwrite": "true"}, timeout=180)
r.raise_for_status(); up = r.json()["name"]

def run(prompt):
    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=180)
    resp = r.json()
    if r.status_code != 200 or resp.get("node_errors"):
        print("SUBMIT:", json.dumps(resp)[:1200]); sys.exit(1)
    pid = resp["prompt_id"]
    for _ in range(120):
        time.sleep(2)
        h = requests.get(f"{URL}/api/history/{pid}", timeout=60).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "error": break
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if "error" in m[0]: print("ERR:", str(m[1].get("exception_message", m[1]))[:800])
        sys.exit(1)
    return h[pid]["outputs"]

def fetch_raw(outputs, prefix):
    for node_id, out in outputs.items():
        for meta in out.get("images", []):
            if meta.get("subfolder") == "alphat" and meta["filename"].startswith(prefix):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                yield meta["filename"], r.content

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
  "6": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
  "7": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["6", 0], "image": ["5", 0]}},
  "8": {"class_type": "InvertMask", "inputs": {"mask": ["7", 0]}},
  "9": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["5", 0], "alpha": ["8", 0]}},
  "10": {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["alphacrisp"], "size_mode": "from_input",
         "images.image0": ["9", 0], "images.image1": ["9", 0], "floats.u_float0": 0.5}},
  "90": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "alphat/sheet"}},
}
name, data = next(fetch_raw(run(p1), "sheet"))
sheet_rgba = Image.open(iolib.BytesIO(data))
print("alpha sheet:", sheet_rgba.mode, sheet_rgba.size)
arr = np.array(sheet_rgba.convert("RGBA"))
alphas = np.unique(arr[:, :, 3])
print("alpha values in sheet:", alphas[:10], "(binary =", set(alphas.tolist()) <= {0, 255}, ")")

# split RGBA halves, re-upload, rejoin alpha after LoadImage splits it
hh, hw = arr.shape[0]//2, arr.shape[1]//2
for qn, sl in [("TL",(slice(0,hh),slice(0,hw))),("TR",(slice(0,hh),slice(hw,None))),
               ("BL",(slice(hh,None),slice(0,hw))),("BR",(slice(hh,None),slice(hw,None)))]:
    qa = arr[sl][:,:,3] > 128
    rows = np.nonzero(qa.sum(axis=1) >= 3)[0]
    print(f"  sheet {qn}: alpha rows [{rows.min()},{rows.max()}] h={rows.max()-rows.min()+1}, "
          f"opaque={int(qa.sum())}, top10rows_px={int(qa[:10].sum())}")
frames = [arr[:hh,:hw], arr[:hh,hw:], arr[hh:,:hw], arr[hh:,hw:]]
ups = []
for i, fr in enumerate(frames):
    b = iolib.BytesIO(); Image.fromarray(fr).save(b, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"alpha_f{i}.png", b.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=120)
    r.raise_for_status(); ups.append(r.json()["name"])
p2 = {}
for i, n in enumerate(ups):
    p2[str(10+i)] = {"class_type": "LoadImage", "inputs": {"image": n}}
    p2[str(20+i)] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": [str(10+i), 0], "alpha": [str(10+i), 1]}}
p2["30"] = {"class_type": "BatchImagesNode", "inputs": {
    "images.image0": ["20", 0], "images.image1": ["21", 0],
    "images.image2": ["22", 0], "images.image3": ["23", 0]}}
p2["40"] = {"class_type": "GLSLShader", "inputs": {"fragment_shader": sh["align"], "size_mode": "from_input",
    "images.image0": ["30", 0], "images.image1": ["30", 0], "bools.u_bool0": True, "floats.u_float0": 0.06}}
p2["91"] = {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["40", 0], "filename_prefix": "alphat/anim",
    "fps": 8, "lossless": True, "quality": 90, "method": "default"}}
outs = run(p2)
name, data = next(fetch_raw(outs, "anim"))
webp = Image.open(iolib.BytesIO(data))
print("webp:", webp.mode, webp.size, "frames:", getattr(webp, "n_frames", 1))
ok = True
feet = []
for i, f in enumerate(ImageSequence.Iterator(webp)):
    fa = np.array(f.convert("RGBA"))
    a = fa[:, :, 3]
    corner_t = (a[:10, :10] == 0).all()
    binary = set(np.unique(a).tolist()) <= {0, 255}
    fg = a > 128
    ys, xs = np.nonzero(fg)
    feet.append(int(ys.max()))
    print(f"  frame {i}: corners transparent={corner_t}, alpha binary={binary}, "
          f"opaque px={int(fg.sum())}, feet={ys.max()}")
    if not (corner_t and binary): ok = False
print("feet spread:", max(feet)-min(feet))
print("RESULT:", "PASS" if ok and max(feet)-min(feet) <= 1 else "FAIL")
