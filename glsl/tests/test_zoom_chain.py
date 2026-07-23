"""Full-chain test: contentzoom -> smartcrop -> batch -> align on GPU."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
zoom_sh = open(f"{REPO}/glsl/contentzoom.glsl").read()
crop_sh = open(f"{REPO}/glsl/smartcrop.glsl").read()
align_sh = open(f"{REPO}/glsl/align.glsl").read()

# raw-style sheet: varied char sizes/positions, one midline-crossing sword
sheet = np.full((1024, 1024, 3), 255, np.uint8)
C = [(200, 40, 40), (40, 160, 40), (40, 40, 200), (160, 40, 160)]
sheet[140:420, 200:320] = C[0]      # top-left, 280 tall (small)
sheet[100:440, 690:830] = C[1]      # top-right, 340 tall
sheet[580:940, 170:310] = C[2]      # bottom-left, 360 tall (largest)
sheet[660:950, 790:910] = C[3]      # bottom-right, 290 tall
sheet[700:725, 310:640] = C[2]      # sword crossing x=512
buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("zoomchain.png", buf.getvalue(), "image/png")},
                  data={"type": "input", "overwrite": "true"}, timeout=120)
r.raise_for_status(); up = r.json()["name"]

prompt = {
  "1": {"class_type": "LoadImage", "inputs": {"image": up}},
  "2": {"class_type": "GLSLShader", "inputs": {
      "fragment_shader": zoom_sh, "size_mode": "from_input",
      "images.image0": ["1", 0], "images.image1": ["1", 0],
      "bools.u_bool0": True, "floats.u_float0": 0.90}},
  "3": {"class_type": "GLSLShader", "inputs": {
      "fragment_shader": crop_sh, "size_mode": "custom",
      "size_mode.width": 512, "size_mode.height": 512,
      "images.image0": ["2", 0], "images.image1": ["2", 0], "floats.u_float0": 0.08}},
  "4": {"class_type": "BatchImagesNode", "inputs": {
      "images.image0": ["3", 0], "images.image1": ["3", 1],
      "images.image2": ["3", 2], "images.image3": ["3", 3]}},
  "5": {"class_type": "GLSLShader", "inputs": {
      "fragment_shader": align_sh, "size_mode": "from_input",
      "images.image0": ["4", 0], "images.image1": ["4", 0],
      "bools.u_bool0": True, "floats.u_float0": 0.06}},
  "90": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "zoomchain/zoomed"}},
  "91": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "zoomchain/final"}},
}
r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=120)
resp = r.json()
if r.status_code != 200 or resp.get("node_errors"):
    print("SUBMIT:", json.dumps(resp)[:1200]); sys.exit(1)
pid = resp["prompt_id"]
for _ in range(60):
    time.sleep(2)
    h = requests.get(f"{URL}/api/history/{pid}", timeout=60).json()
    if pid in h:
        st = h[pid].get("status", {})
        if st.get("completed") or st.get("status_str") == "error": break
if st.get("status_str") == "error":
    for m in st.get("messages", []):
        if "error" in m[0]: print("ERR:", str(m[1].get("exception_message", m[1]))[:700])
    sys.exit(1)

finals, zoomed = {}, None
for node_id, out in h[pid]["outputs"].items():
    for meta in out.get("images", []):
        if meta.get("subfolder") != "zoomchain": continue
        r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                         "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=120)
        img = np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))
        if meta["filename"].startswith("final"):
            finals[len(finals)] = img
        elif meta["filename"].startswith("zoomed"):
            zoomed = img

print("zoomed sheet:", zoomed.shape if zoomed is not None else None,
      "| final frames:", len(finals), "at", finals[0].shape if finals else None)
ok = True
feet = []
for i in range(len(finals)):
    a = finals[i].astype(np.int16)
    present = {}
    for ci, c in enumerate(C):
        n = int((np.abs(a - np.array(c)).max(axis=2) < 30).sum())
        if n > 100: present[ci] = n
    fgm = (255 - a).max(axis=2) > 20
    ys, xs = np.nonzero(fgm)
    hfill = (ys.max() - ys.min() + 1) / a.shape[0]
    feet.append(int(ys.max()))
    print(f"frame {i}: colors {list(present.keys())}, height fill {hfill:.0%}, "
          f"feet-y {ys.max()}, cx {(xs.min()+xs.max())//2}")
    if list(present.keys()) != [i]: ok = False
print("feet spread:", max(feet) - min(feet))
print("RESULT:", "PASS" if ok and max(feet) - min(feet) <= 1 else "CHECK")
