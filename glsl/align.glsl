#version 300 es
// Pixel Snapper companion — Align Frame (anti-jitter for sprite animations).
//
// Image-edit models never place a character at the exact same spot in every
// sprite-sheet cell, so cropped frames jitter. This shader re-anchors each
// frame: the character's bounding box is centered horizontally and its
// bottom (the feet) is pinned to a shared baseline — the same trick the
// sprite-sheet-creator app does at render time, baked into the asset.
// Translation only, in whole pixels: lossless for pixel art.
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = frame or frame batch (each batch item aligns independently)
//   image1 = the SAME input again (stable copy; pass 0 consumes image0)
//   u_bool0  = lock_feet: pin bbox bottom to the baseline. Enable for
//              walk/attack/idle; DISABLE for jump so the arc survives.
//   u_float0 = baseline margin as a fraction of frame height from the
//              bottom edge (0 -> default 0.06).
//
// Foreground detection: alpha channel if the frame has transparency
// (e.g. after Remove Background), otherwise distance-from-white keying.
// Revealed edges fill with transparent black (alpha mode) or white.
#pragma passes 2

precision highp float;

const int MAXDIM = 4096;
const float WHITE_KEY_DIST = 0.06;

uniform sampler2D u_image0; // pass 0: frame; pass 1: params ping-pong
uniform sampler2D u_image1; // frame (stable)
uniform vec2 u_resolution;
uniform bool u_bool0;       // lock_feet
uniform float u_float0;     // baseline margin fraction (0 -> 0.06)
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

bool isFgWhite(vec4 c) {
    vec3 d = abs(c.rgb - vec3(1.0));
    return max(d.r, max(d.g, d.b)) > WHITE_KEY_DIST;
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    int W = sz.x;
    int H = sz.y;

    // PASS 0: scan the frame once; write the foreground bbox to pixel (0,0)
    // and the detection mode to pixel (1,0).
    if (u_pass == 0) {
        if (frag.y == 0 && frag.x < 2) {
            int aMinX = MAXDIM; int aMaxX = -1; int aMinY = MAXDIM; int aMaxY = -1;
            int wMinX = MAXDIM; int wMaxX = -1; int wMinY = MAXDIM; int wMaxY = -1;
            int transparentCount = 0;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H) break;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    vec4 c = texelFetch(u_image1, ivec2(x, y), 0);
                    if (c.a < 0.5) {
                        transparentCount++;
                    } else {
                        aMinX = min(aMinX, x); aMaxX = max(aMaxX, x);
                        aMinY = min(aMinY, y); aMaxY = max(aMaxY, y);
                        if (isFgWhite(c)) {
                            wMinX = min(wMinX, x); wMaxX = max(wMaxX, x);
                            wMinY = min(wMinY, y); wMaxY = max(wMaxY, y);
                        }
                    }
                }
            }
            bool alphaMode = transparentCount > 0;
            int minX = alphaMode ? aMinX : wMinX;
            int maxX = alphaMode ? aMaxX : wMaxX;
            int minY = alphaMode ? aMinY : wMinY;
            int maxY = alphaMode ? aMaxY : wMaxY;
            if (frag.x == 0) {
                if (maxX < 0) {
                    fragColor0 = vec4(0.0); // no foreground -> pass 1 no-ops
                } else {
                    fragColor0 = vec4(float(minX) / 4096.0, float(minY) / 4096.0,
                                      float(maxX) / 4096.0, float(maxY) / 4096.0);
                }
            } else {
                // feet anchor: x-centroid of the character's lowest rows
                // (bbox centers sway with swords/effects; feet stay planted)
                float anchorX = (float(minX) + float(maxX)) * 0.5;
                if (maxX >= 0) {
                    int bandTop = minY + max(int(0.18 * float(maxY - minY + 1)), 6);
                    float sumX = 0.0;
                    float cnt = 0.0;
                    for (int y = 0; y < MAXDIM; y++) {
                        if (y > bandTop || y >= H) break;
                        if (y < minY) continue;
                        for (int x = 0; x < MAXDIM; x++) {
                            if (x >= W) break;
                            vec4 c = texelFetch(u_image1, ivec2(x, y), 0);
                            bool fg = alphaMode ? (c.a >= 0.5) : (c.a >= 0.5 && isFgWhite(c));
                            if (fg) { sumX += float(x); cnt += 1.0; }
                        }
                    }
                    if (cnt > 0.0) anchorX = sumX / cnt;
                }
                fragColor0 = vec4(alphaMode ? 1.0 : 0.0, maxX < 0 ? 0.0 : 1.0,
                                  anchorX / 4096.0, 1.0);
            }
        } else {
            fragColor0 = vec4(0.0);
        }
        return;
    }

    // PASS 1: translate. Texture y=0 is the image BOTTOM (upload flip), so
    // the feet are the bbox's minY and the baseline sits margin*H above 0.
    vec4 bbox = texelFetch(u_image0, ivec2(0, 0), 0) * 4096.0;
    vec4 mode = texelFetch(u_image0, ivec2(1, 0), 0);
    bool alphaMode = mode.x > 0.5;
    bool valid = mode.y > 0.5;

    if (!valid) {
        fragColor0 = texelFetch(u_image1, frag, 0);
        return;
    }

    float margin = u_float0 <= 0.0 ? 0.06 : u_float0;
    float anchorX = mode.z * 4096.0;
    int dx = int(floor(anchorX - float(W) * 0.5 + 0.5));
    int dy = u_bool0 ? int(bbox.y + 0.5) - int(margin * float(H) + 0.5) : 0;

    ivec2 src = frag + ivec2(dx, dy);
    if (src.x < 0 || src.x >= W || src.y < 0 || src.y >= H) {
        fragColor0 = alphaMode ? vec4(0.0) : vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        fragColor0 = texelFetch(u_image1, src, 0);
    }
}
