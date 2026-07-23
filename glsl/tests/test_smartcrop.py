import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
shader = open("/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper/glsl/smartcrop.glsl").read()

# synthetic sheet: 4 colored characters, frame 3 (bottom-left) has a sword
# crossing the vertical midline into frame 4's cell
sheet = np.full((1024, 1024, 3), 255, np.uint8)
C1, C2, C3, C4 = (200, 40, 40), (40, 160, 40), (40, 40, 200), (160, 40, 160)
sheet[120:440, 190:330] = C1        # top-left body
sheet[100:420, 700:840] = C2        # top-right body
sheet[600:940, 180:320] = C3        # bottom-left body
sheet[640:960, 780:920] = C4        # bottom-right body
sheet[690:715, 320:640] = C3        # frame 3's sword: crosses x=512, tip at 640
truth = {"c1": C1, "c2": C2, "c3": C3, "c4": C4}

buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
r = requests.post(f"{URL}/api/upload/image", files={"image": ("smartcrop_test.png", buf.getvalue(), "image/png")},
                  data={"type": "input", "overwrite": "true"}, timeout=120)
r.raise_for_status(); up = r.json()["name"]

prompt = {
  "1": {"class_type": "LoadImage", "inputs": {"image": up}},
  "2": {"class_type": "GLSLShader", "inputs": {
      "fragment_shader": shader, "size_mode": "custom",
      "images.image0": ["1", 0], "images.image1": ["1", 0], "floats.u_float0": 0.18,
      "size_mode.width": 704, "size_mode.height": 704}},
}
for i in range(4):
    prompt[str(10 + i)] = {"class_type": "SaveImage", "inputs": {"images": ["2", i], "filename_prefix": f"smartcrop/out{i}"}}

r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=120)
resp = r.json()
if r.status_code != 200 or resp.get("node_errors"):
    print("SUBMIT:", json.dumps(resp)[:900]); sys.exit(1)
pid = resp["prompt_id"]
for _ in range(60):
    time.sleep(2)
    h = requests.get(f"{URL}/api/history/{pid}", timeout=60).json()
    if pid in h:
        st = h[pid].get("status", {})
        if st.get("completed") or st.get("status_str") == "error": break
if st.get("status_str") == "error":
    for m in st.get("messages", []):
        if "error" in m[0]: print("ERR:", str(m[1].get("exception_message", m[1]))[:600])
    sys.exit(1)

outs = {}
for node_id, out in h[pid]["outputs"].items():
    for meta in out.get("images", []):
        r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                         "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=120)
        idx = int(meta["filename"][3]) if meta.get("subfolder") == "smartcrop" and meta["filename"].startswith("out") else None
        if idx is not None:
            outs[idx] = np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))

def count_colors(img):
    res = {}
    for name, c in truth.items():
        res[name] = int((np.abs(img.astype(np.int16) - np.array(c)).max(axis=2) < 30).sum())
    return res

total_c3 = int((np.abs(sheet.astype(np.int16) - np.array(C3)).max(axis=2) < 30).sum())
names = ["top-left", "top-right", "bottom-left", "bottom-right"]
ok = True
for i in range(4):
    cc = count_colors(outs[i])
    present = {k: v for k, v in cc.items() if v > 50}
    print(f"IMAGE{i} ({names[i]}): {present}")
    expected = ["c1", "c2", "c3", "c4"][i]
    if list(present.keys()) != [expected]: ok = False
c3_in_out2 = count_colors(outs[2])["c3"]
print(f"frame 3 pixels: sheet={total_c3}, IMAGE2={c3_in_out2} "
      f"({'COMPLETE - sword intact' if c3_in_out2 == total_c3 else 'TRUNCATED'})")
naive_lost = int((np.abs(sheet[:, 512:].astype(np.int16) - np.array(C3)).max(axis=2) < 30).sum())
print(f"(fixed-midline crop would have amputated {naive_lost} of those pixels)")
for i in range(4):
    a = outs[i].astype(np.int16)
    fgm = (255 - a).max(axis=2) > 20
    ys, xs = np.nonzero(fgm)
    print(f"IMAGE{i}: canvas {a.shape[1]}x{a.shape[0]}, char y[{ys.min()},{ys.max()}] x[{xs.min()},{xs.max()}] fills {100*(ys.max()-ys.min()+1)/a.shape[0]:.0f}%h")
print("RESULT:", "PASS" if ok and c3_in_out2 == total_c3 else "FAIL")
