"""GPU test of align.glsl on the real jittery animation frames."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image, ImageSequence

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
shader = open(f"{REPO}/glsl/align.glsl").read()

def measure(frames):
    feet, cx = [], []
    for a in frames:
        fg = (255 - a.astype(np.int16)).max(axis=2) > 20
        ys, xs = np.nonzero(fg)
        feet.append(int(ys.max())); cx.append(int((xs.min() + xs.max()) // 2))
    return feet, cx

for name, lock in [("sprites_anim_00003_.webp", True), ("sprites_anim_00004_.webp", True)]:
    im = Image.open(f"{REPO}/{name}")
    frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    up = []
    for i, fr in enumerate(frames):
        buf = iolib.BytesIO(); Image.fromarray(fr).save(buf, format="PNG")
        r = requests.post(f"{URL}/api/upload/image",
                          files={"image": (f"align_{name[:18]}_{i}.png", buf.getvalue(), "image/png")},
                          data={"type": "input", "overwrite": "true"}, timeout=120)
        r.raise_for_status(); up.append(r.json()["name"])

    prompt = {}
    for i, n in enumerate(up):
        prompt[str(10 + i)] = {"class_type": "LoadImage", "inputs": {"image": n}}
    prompt["20"] = {"class_type": "BatchImagesNode", "inputs": {
        "images.image0": ["10", 0], "images.image1": ["11", 0],
        "images.image2": ["12", 0], "images.image3": ["13", 0]}}
    prompt["30"] = {"class_type": "GLSLShader", "inputs": {
        "fragment_shader": shader, "size_mode": "from_input",
        "images.image0": ["20", 0], "images.image1": ["20", 0],
        "bools.u_bool0": lock, "floats.u_float0": 0.06}}
    prompt["40"] = {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": f"align_test/{name[:20]}"}}

    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=120)
    resp = r.json()
    if r.status_code != 200:
        print("SUBMIT FAIL:", json.dumps(resp)[:1200]); sys.exit(1)
    if resp.get("node_errors"):
        print("NODE ERRORS:", json.dumps(resp["node_errors"])[:800]); sys.exit(1)
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

    aligned = []
    for node_id, out in h[pid]["outputs"].items():
        for meta in out.get("images", []):
            if not meta["filename"].startswith(name[:20].split('/')[-1]) and "align_test" not in meta.get("subfolder", ""): continue
            r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                             "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=120)
            aligned.append(np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB")))
    aligned = aligned[:4]

    f0, c0 = measure(frames)
    f1, c1 = measure(aligned)
    print(f"{name} (lock_feet={lock}):")
    print(f"  BEFORE feet {f0} (spread {max(f0)-min(f0)}), cx {c0} (spread {max(c0)-min(c0)})")
    print(f"  AFTER  feet {f1} (spread {max(f1)-min(f1)}), cx {c1} (spread {max(c1)-min(c1)})")
