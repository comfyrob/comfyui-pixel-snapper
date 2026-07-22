"""Generate pixel_snapper_glsl.json — a loadable ComfyUI workflow wiring the
three GLSL stages with core nodes only.

Run:  python3 glsl/emit_workflow.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


def glsl_inputs(n_images, int_label, links):
    """Input slot list for a GLSLShader node (autogrow: connected + 1 spare)."""
    slots = []
    for i in range(n_images + 1):
        slots.append({
            "name": f"images.image{i}",
            "type": "IMAGE",
            "link": links.get(f"image{i}"),
            **({"shape": 7} if i >= 1 else {}),
        })
    slots.append({"name": "floats.u_float0", "shape": 7, "type": "FLOAT", "link": None})
    slots.append({
        "label": int_label,
        "name": "ints.u_int0",
        "shape": 7,
        "type": "INT",
        "link": links.get("u_int0"),
    })
    slots.append({"name": "ints.u_int1", "shape": 7, "type": "INT", "link": None})
    slots.append({"name": "bools.u_bool0", "shape": 7, "type": "BOOLEAN", "link": None})
    slots.append({"name": "curves.u_curve0", "shape": 7, "type": "CURVE", "link": None})
    return slots


def glsl_outputs(links):
    return [
        {"name": f"IMAGE{i}", "type": "IMAGE", "links": links.get(i)} for i in range(4)
    ]


def main():
    nodes = [
        {
            "id": 1, "type": "LoadImage", "pos": [40, 200], "size": [300, 340],
            "flags": {}, "order": 0, "mode": 0, "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [1, 2]},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": ["example.png", "image"],
        },
        {
            "id": 2, "type": "PrimitiveInt", "pos": [40, 620], "size": [300, 90],
            "flags": {}, "order": 1, "mode": 0,
            "inputs": [{"label": "Color Count", "name": "value", "type": "INT",
                        "widget": {"name": "value"}, "link": None}],
            "outputs": [{"name": "INT", "type": "INT", "links": [3, 4]}],
            "title": "Color Count",
            "properties": {"Node name for S&R": "PrimitiveInt"},
            "widgets_values": [16, "fixed"],
        },
        {
            "id": 3, "type": "PrimitiveInt", "pos": [40, 770], "size": [300, 90],
            "flags": {}, "order": 2, "mode": 0,
            "inputs": [{"label": "Pixel Size (0 = auto)", "name": "value", "type": "INT",
                        "widget": {"name": "value"}, "link": None}],
            "outputs": [{"name": "INT", "type": "INT", "links": [5]}],
            "title": "Pixel Size (0 = auto)",
            "properties": {"Node name for S&R": "PrimitiveInt"},
            "widgets_values": [0, "fixed"],
        },
        {
            "id": 10, "type": "GLSLShader", "pos": [420, 120], "size": [380, 620],
            "flags": {}, "order": 3, "mode": 0,
            "inputs": glsl_inputs(2, "color_count", {"image0": 1, "image1": 2, "u_int0": 3}),
            "outputs": glsl_outputs({0: [10, 11, 12], 1: [13]}),
            "title": "Pixel Snapper 1/3 — Quantize",
            "properties": {"Node name for S&R": "GLSLShader"},
            "widgets_values": [src("quantize.glsl"), "from_input"],
        },
        {
            "id": 20, "type": "GLSLShader", "pos": [860, 120], "size": [380, 620],
            "flags": {}, "order": 4, "mode": 0,
            "inputs": glsl_inputs(2, "pixel_size (0=auto)", {"image0": 10, "image1": 11, "u_int0": 5})
            + [
                {"name": "size_mode.width", "type": "INT",
                 "widget": {"name": "size_mode.width"}, "link": None},
                {"name": "size_mode.height", "type": "INT",
                 "widget": {"name": "size_mode.height"}, "link": None},
            ],
            "outputs": glsl_outputs({0: [20]}),
            "title": "Pixel Snapper 2/3 — Detect Grid",
            "properties": {"Node name for S&R": "GLSLShader"},
            "widgets_values": [src("detect.glsl"), "custom", 2048, 8],
        },
        {
            "id": 30, "type": "GLSLShader", "pos": [1300, 120], "size": [380, 620],
            "flags": {}, "order": 5, "mode": 0,
            "inputs": glsl_inputs(3, "color_count", {"image0": 12, "image1": 20,
                                                     "image2": 13, "u_int0": 4}),
            "outputs": glsl_outputs({0: [30, 31]}),
            "title": "Pixel Snapper 3/3 — Snap",
            "properties": {"Node name for S&R": "GLSLShader"},
            "widgets_values": [src("snap.glsl"), "from_input"],
        },
        {
            "id": 40, "type": "PreviewImage", "pos": [1740, 120], "size": [320, 300],
            "flags": {}, "order": 6, "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 30}],
            "outputs": [],
            "title": "Snapped (full size)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 50, "type": "ImageScaleBy", "pos": [1740, 480], "size": [320, 100],
            "flags": {}, "order": 7, "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": 31}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [32]}],
            "title": "Native scale (set 1/pixel_size)",
            "properties": {"Node name for S&R": "ImageScaleBy"},
            "widgets_values": ["nearest-exact", 1.0],
        },
        {
            "id": 51, "type": "PreviewImage", "pos": [1740, 640], "size": [320, 300],
            "flags": {}, "order": 8, "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 32}],
            "outputs": [],
            "title": "Native (downscaled)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
    ]

    # [link_id, from_node, from_slot, to_node, to_slot, type]
    def slot_index(node_id, input_name):
        node = next(n for n in nodes if n["id"] == node_id)
        for i, s in enumerate(node["inputs"]):
            if s["name"] == input_name:
                return i
        raise KeyError(input_name)

    links = [
        [1, 1, 0, 10, slot_index(10, "images.image0"), "IMAGE"],
        [2, 1, 0, 10, slot_index(10, "images.image1"), "IMAGE"],
        [3, 2, 0, 10, slot_index(10, "ints.u_int0"), "INT"],
        [4, 2, 0, 30, slot_index(30, "ints.u_int0"), "INT"],
        [5, 3, 0, 20, slot_index(20, "ints.u_int0"), "INT"],
        [10, 10, 0, 20, slot_index(20, "images.image0"), "IMAGE"],
        [11, 10, 0, 20, slot_index(20, "images.image1"), "IMAGE"],
        [12, 10, 0, 30, slot_index(30, "images.image0"), "IMAGE"],
        [13, 10, 1, 30, slot_index(30, "images.image2"), "IMAGE"],
        [20, 20, 0, 30, slot_index(30, "images.image1"), "IMAGE"],
        [30, 30, 0, 40, 0, "IMAGE"],
        [31, 30, 0, 50, 0, "IMAGE"],
        [32, 50, 0, 51, 0, "IMAGE"],
    ]

    workflow = {
        "id": "pixel-snapper-glsl",
        "revision": 0,
        "last_node_id": 51,
        "last_link_id": 32,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }

    out = os.path.join(HERE, "pixel_snapper_glsl.json")
    with open(out, "w") as f:
        json.dump(workflow, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
