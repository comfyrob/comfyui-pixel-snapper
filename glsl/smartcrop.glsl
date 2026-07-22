#version 300 es
// Pixel Snapper companion — Smart Crop 2x2 (content-aware frame extraction).
//
// A generated sprite sheet arranges 4 frames in a 2x2 grid, but the model
// often lets a pose spill across the nominal midlines (a lunging sword
// crossing into the neighbor's cell). Fixed half-and-half cropping then
// truncates those pixels. This shader cuts along ADAPTIVE SEAMS instead:
// within a band around each midline it picks the row/column crossing the
// fewest foreground pixels (ideally a clean background gap), so every
// character keeps all of its pixels.
//
// Node setup: core GLSLShader, size_mode = custom, sized to one cell plus
// the seam band — e.g. 704x704 for a 1024 sheet (512 * (1 + 2*band)).
// from_input also works; frames then sit on a sheet-sized canvas.
//   image0 = sprite sheet (consumed by pass 0's ping-pong)
//   image1 = the SAME sheet again (stable copy)
//   u_float0 = seam search band as a fraction of the sheet dimension
//              (0 -> default 0.18, i.e. +-18% around each midline)
// Outputs (reading order):
//   IMAGE0 = top-left frame      IMAGE1 = top-right frame
//   IMAGE2 = bottom-left frame   IMAGE3 = bottom-right frame
//
// Every frame's region is translated to a COMMON origin (its cell's
// bottom-left corner -> canvas bottom-left), so cell offsets vanish while
// in-cell motion (a jump arc) survives. Chain the Align Frames shader
// downstream to center each character and pin the feet baseline.
// Foreground detection: alpha if the sheet has transparency, else
// distance-from-white. Blanked areas fill to match (transparent / white).
#pragma passes 2

precision highp float;

const int MAXDIM = 2048;
const float WHITE_KEY_DIST = 0.06;

uniform sampler2D u_image0; // pass 0: sheet; pass 1: params ping-pong
uniform sampler2D u_image1; // sheet (stable)
uniform vec2 u_resolution;
uniform float u_float0;     // seam search band fraction (0 -> 0.18)
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;

bool isFg(vec4 c, bool alphaMode) {
    if (alphaMode) return c.a > 0.5;
    vec3 d = abs(c.rgb - vec3(1.0));
    return max(d.r, max(d.g, d.b)) > WHITE_KEY_DIST;
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    int W = sz.x;
    int H = sz.y;
    float band = u_float0 <= 0.0 ? 0.18 : u_float0;

    // PASS 0: find the seams; write params to pixel (0,0) and mode to (1,0).
    if (u_pass == 0) {
        if (frag.y == 0 && frag.x < 2) {
            // does the sheet carry real transparency?
            bool alphaMode = false;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H || alphaMode) break;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    if (texelFetch(u_image1, ivec2(x, y), 0).a < 0.5) { alphaMode = true; break; }
                }
            }

            // horizontal seam: row (texture space) with fewest fg crossings
            // in a band around H/2, ties resolved toward the midline
            int yLo = int(float(H) * (0.5 - band));
            int yHi = int(float(H) * (0.5 + band));
            int bestY = H / 2;
            float bestYCost = 1e9;
            for (int y = 0; y < MAXDIM; y++) {
                if (y < yLo) continue;
                if (y > yHi || y >= H) break;
                float cost = 0.0;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    if (isFg(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
                }
                cost += abs(float(y) - float(H) * 0.5) * 0.01; // tie-break: prefer midline
                if (cost < bestYCost) { bestYCost = cost; bestY = y; }
            }

            // vertical seams, computed separately for each row band
            // (texture y=0 is the image BOTTOM, so y > bestY = image top row)
            int xLo = int(float(W) * (0.5 - band));
            int xHi = int(float(W) * (0.5 + band));
            int bestXTop = W / 2;
            int bestXBot = W / 2;
            for (int hIdx = 0; hIdx < 2; hIdx++) {
                float bestCost = 1e9;
                int bestX = W / 2;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x < xLo) continue;
                    if (x > xHi || x >= W) break;
                    float cost = 0.0;
                    for (int y = 0; y < MAXDIM; y++) {
                        if (y >= H) break;
                        bool inHalf = (hIdx == 0) ? (y > bestY) : (y <= bestY);
                        if (inHalf && isFg(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
                    }
                    cost += abs(float(x) - float(W) * 0.5) * 0.01;
                    if (cost < bestCost) { bestCost = cost; bestX = x; }
                }
                if (hIdx == 0) bestXTop = bestX; else bestXBot = bestX;
            }

            if (frag.x == 0) {
                fragColor0 = vec4(float(bestY) / 2048.0, float(bestXTop) / 2048.0,
                                  float(bestXBot) / 2048.0, 1.0);
            } else {
                fragColor0 = vec4(alphaMode ? 1.0 : 0.0, 0.0, 0.0, 1.0);
            }
        } else {
            fragColor0 = vec4(0.0);
        }
        fragColor1 = vec4(0.0); fragColor2 = vec4(0.0); fragColor3 = vec4(0.0);
        return;
    }

    // PASS 1: each output samples the sheet at its own cell origin, so all
    // four frames land at the canvas bottom-left in a shared coordinate
    // frame. A sampled pixel only shows if that frame actually owns it.
    vec4 params = texelFetch(u_image0, ivec2(0, 0), 0) * 2048.0;
    bool alphaMode = texelFetch(u_image0, ivec2(1, 0), 0).x > 0.5;
    int yCut = int(params.x + 0.5);
    int xCutTop = int(params.y + 0.5);
    int xCutBot = int(params.z + 0.5);
    vec4 blank = alphaMode ? vec4(0.0) : vec4(1.0, 1.0, 1.0, 1.0);

    // cell origins in texture space (texture y=0 is the image bottom)
    ivec2 origins[4];
    origins[0] = ivec2(0, yCut + 1);       // image top-left
    origins[1] = ivec2(xCutTop, yCut + 1); // image top-right
    origins[2] = ivec2(0, 0);              // image bottom-left
    origins[3] = ivec2(xCutBot, 0);        // image bottom-right

    vec4 outs[4];
    for (int i = 0; i < 4; i++) {
        ivec2 src = frag + origins[i];
        vec4 c = blank;
        if (src.x >= 0 && src.x < W && src.y >= 0 && src.y < H) {
            bool topRow = src.y > yCut;
            bool leftCol = src.x < (topRow ? xCutTop : xCutBot);
            int owner = topRow ? (leftCol ? 0 : 1) : (leftCol ? 2 : 3);
            if (owner == i) c = texelFetch(u_image1, src, 0);
        }
        outs[i] = c;
    }
    fragColor0 = outs[0];
    fragColor1 = outs[1];
    fragColor2 = outs[2];
    fragColor3 = outs[3];
}
