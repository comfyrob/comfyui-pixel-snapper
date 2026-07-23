"""Shared-ground fix: arcs preserved AND grounded sheets keep their zoom."""
import io as iolib, json, sys, time
import numpy as np
import requests
from PIL import Image

URL = "https://rob-73810--comfyui-rtxpro6000-b-comfyui-rtxpro6000-b.modal.run"
REPO = "/Users/rob/comfyvibe/projects/custom-nodes/pixel-snapper"
zoom_sh = open(f"{REPO}/glsl/contentzoom.glsl").read()
PAIRS = [((180,30,30),(240,120,120)), ((30,140,30),(120,220,120)),
         ((30,30,180),(120,120,240)), ((150,30,150),(230,120,230))]

def make_sheet(kind):
    W, H, AP = 1376, 768, 8
    rng = np.random.default_rng(11)
    sheet = np.full((H, W, 3), 255, np.uint8)
    def draw(x0, y0, wpx, hpx, pair):
        for gy in range(hpx):
            for gx in range(wpx):
                sheet[y0+gy*AP:y0+(gy+1)*AP, x0+gx*AP:x0+(gx+1)*AP] = pair[int(rng.integers(0,2))]
    if kind == "grounded":  # realistic margins under feet (the regression case)
        draw(250, 130, 15, 24, PAIRS[0])   # feet at y=322, cell bottom 384 -> 62px margin
        draw(950, 138, 16, 23, PAIRS[1])   # 62px margin
        draw(260, 520, 15, 25, PAIRS[2])   # feet 720 -> 48px margin
        draw(960, 528, 14, 24, PAIRS[3])   # 48px margin
    elif kind == "lunge":  # asymmetric: long sword right of feet-left body
        draw(230, 130, 13, 24, PAIRS[0])
        draw(920, 130, 13, 24, PAIRS[1]); draw(1024, 200, 30, 3, PAIRS[1])  # sword 240px right
        draw(240, 520, 13, 25, PAIRS[2]); draw(344, 590, 32, 3, PAIRS[2])   # sword 256px right
        draw(930, 522, 13, 24, PAIRS[3])
    else:  # arc: apex char + raised sword across seam
        draw(240, 64, 14, 24, PAIRS[0])
        draw(960, 96, 14, 24, PAIRS[1])
        draw(280, 512, 14, 28, PAIRS[2]); draw(320, 304, 3, 26, PAIRS[2])
        draw(1000, 520, 14, 26, PAIRS[3])
    return sheet

def run_zoom(sheet, tag):
    buf = iolib.BytesIO(); Image.fromarray(sheet).save(buf, format="PNG")
    r = requests.post(f"{URL}/api/upload/image", files={"image": (f"g_{tag}.png", buf.getvalue(), "image/png")},
                      data={"type": "input", "overwrite": "true"}, timeout=180)
    r.raise_for_status(); up = r.json()["name"]
    prompt = {
      "1": {"class_type": "LoadImage", "inputs": {"image": up}},
      "2": {"class_type": "GLSLShader", "inputs": {"fragment_shader": zoom_sh, "size_mode": "from_input",
            "images.image0": ["1", 0], "images.image1": ["1", 0], "bools.u_bool0": False, "floats.u_float0": 0.90}},
      "90": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": f"ground/{tag}"}},
    }
    r = requests.post(f"{URL}/api/prompt", json={"prompt": prompt}, timeout=180)
    resp = r.json()
    if r.status_code != 200 or resp.get("node_errors"):
        print("SUBMIT:", json.dumps(resp)[:800]); sys.exit(1)
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
    for node_id, out in h[pid]["outputs"].items():
        for meta in out.get("images", []):
            if meta.get("subfolder") == "ground" and meta["filename"].startswith(tag):
                r = requests.get(f"{URL}/api/view", params={"filename": meta["filename"],
                                 "subfolder": meta["subfolder"], "type": meta["type"]}, timeout=180)
                return np.array(Image.open(iolib.BytesIO(r.content)).convert("RGB"))

ok = True
# --- regression case: grounded with margins, lock OFF -> zoom must return
z = run_zoom(make_sheet("grounded"), "grounded")
H, W = z.shape[:2]; hh, hw = H//2, W//2
fg = (255 - z.astype(np.int16)).max(axis=2) > 20
print("grounded sheet (lock off):")
for qn, q in [("TL", fg[:hh,:hw]), ("TR", fg[:hh,hw:]), ("BL", fg[hh:,:hw]), ("BR", fg[hh:,hw:])]:
    ys, xs = np.nonzero(q)
    fillh = 100*(ys.max()-ys.min()+1)//q.shape[0]
    feet_margin = q.shape[0]-1-ys.max()
    edge = int(q[:3,:].sum())
    print(f"  {qn}: fillH {fillh}%, feet margin {feet_margin}px, top-edge px {edge}")
    if fillh < 70 or edge > 20: ok = False

# --- arc case: must still keep sword + altitude, no clipping
z2 = run_zoom(make_sheet("arc"), "arc")
fg2 = (255 - z2.astype(np.int16)).max(axis=2) > 20
print("arc sheet (lock off):")
hts = {}
for qn, q in [("TL", fg2[:hh,:hw]), ("TR", fg2[:hh,hw:]), ("BL", fg2[hh:,:hw]), ("BR", fg2[hh:,hw:])]:
    ys, xs = np.nonzero(q)
    hts[qn] = ys.max()-ys.min()+1
    edge = int(q[:3,:].sum())
    print(f"  {qn}: contentH {hts[qn]}, top-edge px {edge}, feet_row {ys.max()}")
    if edge > 20: ok = False
ratio = hts["BL"]/hts["BR"]
print(f"BL/BR height ratio {ratio:.2f} (sword intact if ~1.7-2.2)")
if not (1.5 < ratio < 2.4): ok = False
# --- lunge case: bbox-bound scale keeps fills high; clamped position, no clip
z3 = run_zoom(make_sheet("lunge"), "lunge")
fg3 = (255 - z3.astype(np.int16)).max(axis=2) > 20
print("lunge sheet (lock off):")
for qn, q in [("TL", fg3[:hh,:hw]), ("TR", fg3[:hh,hw:]), ("BL", fg3[hh:,:hw]), ("BR", fg3[hh:,hw:])]:
    ys, xs = np.nonzero(q)
    fillw = 100*(xs.max()-xs.min()+1)//q.shape[1]
    edge = int(q[:,:3].sum() + q[:,-3:].sum() + q[:3,:].sum() + q[-3:,:].sum())
    print(f"  {qn}: fillW {fillw}%, fillH {100*(ys.max()-ys.min()+1)//q.shape[0]}%, edge px {edge}")
    if edge > 20: ok = False
# the sword frames should reach high WIDTH fill (bbox-bound, not anchor-bound)
q = fg3[hh:,:hw]; ys, xs = np.nonzero(q)
if (xs.max()-xs.min()+1)/q.shape[1] < 0.75: ok = False
print("RESULT:", "PASS" if ok else "FAIL")
