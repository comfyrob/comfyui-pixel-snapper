#version 300 es
// Pixel Snapper companion — Content Zoom (full-bleed normalizer).
//
// Runs on the RAW generated sprite sheet, BEFORE pixel snapping (scaling
// fake pixel art is harmless there — the art-pixel grid scales with it and
// the snapper re-detects it). One shader sees all four frames at once, so
// cross-frame decisions are possible: it computes a single UNIFORM scale
// factor that makes the largest character fill `fill` of its cell, then
// re-renders the sheet with every cell's content scaled and re-centered
// onto clean nominal quadrants. Relative sizes between frames are
// preserved (one factor for all), spill across midlines is eliminated,
// and downstream Smart Crop can run with a tight band and cell-sized
// canvas for near-full-bleed frames.
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = raw sprite sheet (consumed by pass 0's ping-pong)
//   image1 = the SAME sheet again (stable copy)
//   u_bool0  = lock_feet: also pin each character's feet to a shared
//              per-cell baseline. Enable for walk/attack/idle; DISABLE
//              for jump (content scales about the cell bottom instead,
//              so the arc survives, proportionally scaled).
//   u_float0 = target fill fraction of the cell (0 -> default 0.90)
//
// Foreground: alpha if the sheet has transparency, else white-keying.
#pragma passes 2

precision highp float;

const int MAXDIM = 4096;
const float WHITE_KEY_DIST = 0.06;
const float SEAM_BAND = 0.18;
const float BASELINE = 0.08;   // feet margin as fraction of cell height

uniform sampler2D u_image0; // pass 0: sheet; pass 1: params ping-pong
uniform sampler2D u_image1; // sheet (stable)
uniform vec2 u_resolution;
uniform bool u_bool0;       // lock_feet
uniform float u_float0;     // fill target (0 -> 0.90)
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

bool isFgColor(vec4 c, bool alphaMode) {
    if (alphaMode) return c.a > 0.5;
    vec3 d = abs(c.rgb - vec3(1.0));
    return max(d.r, max(d.g, d.b)) > WHITE_KEY_DIST;
}

// ---- seam search: horizontal seams computed PER COLUMN HALF so a sword
// raised across the midline in one column only lifts that side's seam ----
void findSeams(int W, int H, bool alphaMode, out int yCutL, out int yCutR, out int xCutTop, out int xCutBot) {
    int yLo = int(float(H) * (0.5 - SEAM_BAND));
    int yHi = int(float(H) * (0.5 + SEAM_BAND));
    yCutL = H / 2;
    yCutR = H / 2;
    for (int side = 0; side < 2; side++) {
        int x0 = side == 0 ? 0 : W / 2;
        int x1 = side == 0 ? W / 2 : W;
        float bestCostY = 1e9;
        int bestY = H / 2;
        for (int y = 0; y < MAXDIM; y++) {
            if (y < yLo) continue;
            if (y > yHi || y >= H) break;
            float cost = 0.0;
            for (int x = 0; x < MAXDIM; x++) {
                if (x < x0) continue;
                if (x >= x1 || x >= W) break;
                if (isFgColor(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
            }
            cost += abs(float(y) - float(H) * 0.5) * 0.01;
            if (cost < bestCostY) { bestCostY = cost; bestY = y; }
        }
        if (side == 0) yCutL = bestY; else yCutR = bestY;
    }
    int xLo = int(float(W) * (0.5 - SEAM_BAND));
    int xHi = int(float(W) * (0.5 + SEAM_BAND));
    xCutTop = W / 2;
    xCutBot = W / 2;
    for (int hIdx = 0; hIdx < 2; hIdx++) {
        float bestCost = 1e9;
        int bestX = W / 2;
        for (int x = 0; x < MAXDIM; x++) {
            if (x < xLo) continue;
            if (x > xHi || x >= W) break;
            float cost = 0.0;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H) break;
                int yc = x < W / 2 ? yCutL : yCutR;
                bool inHalf = (hIdx == 0) ? (y > yc) : (y <= yc);
                if (inHalf && isFgColor(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
            }
            cost += abs(float(x) - float(W) * 0.5) * 0.01;
            if (cost < bestCost) { bestCost = cost; bestX = x; }
        }
        if (hIdx == 0) xCutTop = bestX; else xCutBot = bestX;
    }
}

int ownerOf(ivec2 p, int W, int yCutL, int yCutR, int xCutTop, int xCutBot) {
    int yc = p.x < W / 2 ? yCutL : yCutR;     // per-side horizontal seam
    bool topRow = p.y > yc;                   // texture y up = image top
    bool leftCol = p.x < (topRow ? xCutTop : xCutBot);
    return topRow ? (leftCol ? 0 : 1) : (leftCol ? 2 : 3);
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    int W = sz.x;
    int H = sz.y;

    // PASS 0: params. (0,0)=seams, (1,0)=mode, (2..5,0)=per-cell bboxes,
    // (6..9,0)=per-cell feet anchors (x-centroid of the lowest ~18% of the
    // character — extremities like swords swing the bbox, feet stay planted).
    if (u_pass == 0) {
        if (frag.y == 0 && frag.x < 10) {
            bool alphaMode = false;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H || alphaMode) break;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    if (texelFetch(u_image1, ivec2(x, y), 0).a < 0.5) { alphaMode = true; break; }
                }
            }
            int yCutL; int yCutR; int xCutTop; int xCutBot;
            findSeams(W, H, alphaMode, yCutL, yCutR, xCutTop, xCutBot);

            if (frag.x == 0) {
                fragColor0 = vec4(float(yCutL) / 4096.0, float(yCutR) / 4096.0,
                                  float(xCutTop) / 4096.0, float(xCutBot) / 4096.0);
            } else if (frag.x == 1) {
                fragColor0 = vec4(alphaMode ? 1.0 : 0.0, 0.0, 0.0, 1.0);
            } else {
                int cell = (frag.x - 2) % 4;
                bool wantAnchor = frag.x >= 6;
                int minX = MAXDIM; int maxX = -1; int minY = MAXDIM; int maxY = -1;
                for (int y = 0; y < MAXDIM; y++) {
                    if (y >= H) break;
                    for (int x = 0; x < MAXDIM; x++) {
                        if (x >= W) break;
                        ivec2 p = ivec2(x, y);
                        if (ownerOf(p, W, yCutL, yCutR, xCutTop, xCutBot) != cell) continue;
                        if (isFgColor(texelFetch(u_image1, p, 0), alphaMode)) {
                            minX = min(minX, x); maxX = max(maxX, x);
                            minY = min(minY, y); maxY = max(maxY, y);
                        }
                    }
                }
                if (maxX < 0) {
                    fragColor0 = vec4(0.0); // empty cell
                } else if (!wantAnchor) {
                    fragColor0 = vec4(float(minX) / 4096.0, float(minY) / 4096.0,
                                      float(maxX) / 4096.0, float(maxY) / 4096.0);
                } else {
                    // feet band: lowest rows of the character (texture y up)
                    int bandTop = minY + max(int(0.18 * float(maxY - minY + 1)), 6);
                    float sumX = 0.0;
                    float cnt = 0.0;
                    for (int y = 0; y < MAXDIM; y++) {
                        if (y > bandTop || y >= H) break;
                        if (y < minY) continue;
                        for (int x = 0; x < MAXDIM; x++) {
                            if (x >= W) break;
                            ivec2 p = ivec2(x, y);
                            if (ownerOf(p, W, yCutL, yCutR, xCutTop, xCutBot) != cell) continue;
                            if (isFgColor(texelFetch(u_image1, p, 0), alphaMode)) {
                                sumX += float(x); cnt += 1.0;
                            }
                        }
                    }
                    float anchorX = cnt > 0.0 ? sumX / cnt : (float(minX) + float(maxX)) * 0.5;
                    fragColor0 = vec4(anchorX / 4096.0, cnt > 0.0 ? 1.0 : 0.0, 0.0, 1.0);
                }
            }
        } else {
            fragColor0 = vec4(0.0);
        }
        return;
    }

    // PASS 1: re-render onto clean nominal quadrants with one uniform scale.
    vec4 seams = texelFetch(u_image0, ivec2(0, 0), 0) * 4096.0;
    bool alphaMode = texelFetch(u_image0, ivec2(1, 0), 0).x > 0.5;
    int yCutL = int(seams.x + 0.5);
    int yCutR = int(seams.y + 0.5);
    int xCutTop = int(seams.z + 0.5);
    int xCutBot = int(seams.w + 0.5);
    float fill = u_float0 <= 0.0 ? 0.90 : u_float0;
    float halfW = float(W) * 0.5;
    float halfH = float(H) * 0.5;
    vec4 blank = alphaMode ? vec4(0.0) : vec4(1.0, 1.0, 1.0, 1.0);

    vec4 bboxes[4];
    float anchors[4];
    bool valid[4];
    float s = 1e9;
    bool anyValid = false;
    for (int i = 0; i < 4; i++) {
        bboxes[i] = texelFetch(u_image0, ivec2(2 + i, 0), 0) * 4096.0;
        anchors[i] = texelFetch(u_image0, ivec2(6 + i, 0), 0).x * 4096.0;
        valid[i] = bboxes[i].z >= bboxes[i].x && (bboxes[i].z + bboxes[i].w) > 0.0;
        if (valid[i]) {
            // content is positioned by its feet anchor, so the fit must
            // respect the asymmetric extents about that anchor (a long
            // sword on one side must not clip the cell edge)
            float extL = max(anchors[i] - bboxes[i].x + 1.0, 1.0);
            float extR = max(bboxes[i].z - anchors[i] + 1.0, 1.0);
            // vertical: with lock_feet the content re-bases to the baseline
            // (bound its height); without, it keeps its in-cell altitude
            // (bound its TOP EXTENT above the cell base, or arcs clip)
            float cellBase = (i < 2) ? float(((i == 0 ? yCutL : yCutR)) + 1) : 0.0;
            float vExt = u_bool0 ? (bboxes[i].w - bboxes[i].y + 1.0)
                                 : max(bboxes[i].w - cellBase + 1.0, 1.0);
            s = min(s, fill * min(min((halfW * 0.5) / extL, (halfW * 0.5) / extR),
                                  halfH / vExt));
            anyValid = true;
        }
    }
    if (!anyValid) { fragColor0 = texelFetch(u_image1, frag, 0); return; }
    s = clamp(s, 0.25, 6.0);

    // destination cell = nominal quadrant of this output pixel
    int col = frag.x >= int(halfW) ? 1 : 0;
    bool topRow = frag.y >= int(halfH);       // texture top half = image top row
    int i = topRow ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    if (!valid[i]) { fragColor0 = blank; return; }

    vec2 cellDst = vec2(float(col) * halfW, topRow ? halfH : 0.0);
    float srcCellBottom = topRow ? float((col == 0 ? yCutL : yCutR) + 1) : 0.0;

    float anchorSrcX = anchors[i];
    float anchorDstX = cellDst.x + halfW * 0.5;
    float srcX = anchorSrcX + (float(frag.x) - anchorDstX) / s;

    float srcY;
    if (u_bool0) {
        float feetDstY = cellDst.y + BASELINE * halfH;
        srcY = bboxes[i].y + (float(frag.y) - feetDstY) / s;
    } else {
        srcY = srcCellBottom + (float(frag.y) - cellDst.y) / s;
    }

    ivec2 src = ivec2(int(srcX + 0.5), int(srcY + 0.5));
    if (src.x < 0 || src.x >= W || src.y < 0 || src.y >= H
        || ownerOf(src, W, yCutL, yCutR, xCutTop, xCutBot) != i) {
        fragColor0 = blank;
    } else {
        fragColor0 = texelFetch(u_image1, src, 0);
    }
}
