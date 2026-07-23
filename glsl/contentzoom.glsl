#version 300 es
// Pixel Snapper companion — Content Zoom (full-bleed normalizer, grid-aware).
//
// Runs on the RAW generated sprite sheet, BEFORE pixel snapping. One shader
// sees every frame at once, so cross-frame decisions are possible: it
// computes a single UNIFORM scale factor sizing the most-constrained
// character to `fill` of its cell, then re-renders the sheet with each
// cell's content scaled and re-anchored onto clean nominal cells.
//
// Works for any grid up to 4x4 (2x2, 3x3, 4x2, ...). Seams between cells
// are ADAPTIVE: each boundary segment is searched per nominal column/row
// band for the line crossing the fewest foreground pixels, so poses that
// spill across nominal midlines keep all their pixels.
//
// Design ledger (each rule exists because a real artifact forced it):
//   scale    = bbox-bounded + shared-ground vertical (asymmetry and
//              under-feet margins must never shrink the sheet)
//   position = feet-band anchor at cell center, CLAMPED in-cell
//              (anti-sway, but alignment yields to visibility)
//   ownership= per-band adaptive seams (raised/lunging weapons stay whole)
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = raw sprite sheet (consumed by pass 0's ping-pong)
//   image1 = the SAME sheet again (stable copy)
//   u_int0   = grid columns (0 -> 2)
//   u_int1   = grid rows    (0 -> 2)
//   u_bool0  = lock_feet: pin every frame's feet to the cell baseline.
//              Enable for grounded animations; DISABLE for jump — frames
//              then rebase to the SHARED GROUND (lowest feet across
//              frames) so arcs survive, proportionally scaled.
//   u_float0 = target fill fraction of the cell (0 -> default 0.90)
//
// Foreground: alpha if the sheet has transparency, else white-keying.
#pragma passes 2

precision highp float;

const int MAXDIM = 4096;
const int MAX_COLS = 4;
const int MAX_ROWS = 4;
const int MAX_CELLS = 16;
const float WHITE_KEY_DIST = 0.06;
const float SEAM_BAND = 0.36;  // search band as a fraction of the cell dim
const float BASELINE = 0.08;   // feet margin as fraction of cell height

uniform sampler2D u_image0; // pass 0: sheet; pass 1: params ping-pong
uniform sampler2D u_image1; // sheet (stable)
uniform vec2 u_resolution;
uniform int u_int0;         // columns
uniform int u_int1;         // rows
uniform bool u_bool0;       // lock_feet
uniform float u_float0;     // fill target (0 -> 0.90)
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

int gridCols() { return clamp(u_int0 <= 0 ? 2 : u_int0, 1, MAX_COLS); }
int gridRows() { return clamp(u_int1 <= 0 ? 2 : u_int1, 1, MAX_ROWS); }

bool isFgColor(vec4 c, bool alphaMode) {
    if (alphaMode) return c.a > 0.5;
    vec3 d = abs(c.rgb - vec3(1.0));
    return max(d.r, max(d.g, d.b)) > WHITE_KEY_DIST;
}

// Horizontal seam for boundary k (1..R-1) within nominal column c's band.
int findHSeam(int k, int c, int W, int H, float cellW, float cellH, bool alphaMode) {
    float y0 = float(k) * cellH;
    int yLo = int(y0 - SEAM_BAND * cellH);
    int yHi = int(y0 + SEAM_BAND * cellH);
    int x0 = int(float(c) * cellW);
    int x1 = int(float(c + 1) * cellW);
    float bestCost = 1e9;
    int bestY = int(y0);
    for (int y = 0; y < MAXDIM; y++) {
        if (y < yLo) continue;
        if (y > yHi || y >= H) break;
        float cost = 0.0;
        for (int x = 0; x < MAXDIM; x++) {
            if (x < x0) continue;
            if (x >= x1 || x >= W) break;
            if (isFgColor(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
        }
        cost += abs(float(y) - y0) * 0.01;
        if (cost < bestCost) { bestCost = cost; bestY = y; }
    }
    return bestY;
}

// Vertical seam for boundary j (1..C-1) within nominal row r's band.
int findVSeam(int j, int r, int W, int H, float cellW, float cellH, bool alphaMode) {
    float x0 = float(j) * cellW;
    int xLo = int(x0 - SEAM_BAND * cellW);
    int xHi = int(x0 + SEAM_BAND * cellW);
    int y0 = int(float(r) * cellH);
    int y1 = int(float(r + 1) * cellH);
    float bestCost = 1e9;
    int bestX = int(x0);
    for (int x = 0; x < MAXDIM; x++) {
        if (x < xLo) continue;
        if (x > xHi || x >= W) break;
        float cost = 0.0;
        for (int y = 0; y < MAXDIM; y++) {
            if (y < y0) continue;
            if (y >= y1 || y >= H) break;
            if (isFgColor(texelFetch(u_image1, ivec2(x, y), 0), alphaMode)) cost += 1.0;
        }
        cost += abs(float(x) - x0) * 0.01;
        if (cost < bestCost) { bestCost = cost; bestX = x; }
    }
    return bestX;
}

void computeSeams(int W, int H, float cellW, float cellH, bool alphaMode,
                  out int hS[12], out int vS[12]) {
    int C = gridCols();
    int R = gridRows();
    for (int i = 0; i < 12; i++) { hS[i] = 0; vS[i] = 0; }
    for (int k = 1; k < MAX_ROWS; k++) {
        if (k >= R) break;
        for (int c = 0; c < MAX_COLS; c++) {
            if (c >= C) break;
            hS[(k - 1) * MAX_COLS + c] = findHSeam(k, c, W, H, cellW, cellH, alphaMode);
        }
    }
    for (int j = 1; j < MAX_COLS; j++) {
        if (j >= C) break;
        for (int r = 0; r < MAX_ROWS; r++) {
            if (r >= R) break;
            vS[(j - 1) * MAX_ROWS + r] = findVSeam(j, r, W, H, cellW, cellH, alphaMode);
        }
    }
}

// Cell index (texture space: row 0 = bottom band) that owns pixel p.
int ownerOf(ivec2 p, float cellW, float cellH, int hS[12], int vS[12]) {
    int C = gridCols();
    int R = gridRows();
    int cNom = clamp(int(float(p.x) / cellW), 0, C - 1);
    int r = 0;
    for (int k = 1; k < MAX_ROWS; k++) {
        if (k >= R) break;
        if (p.y > hS[(k - 1) * MAX_COLS + cNom]) r = k;
    }
    int c = 0;
    for (int j = 1; j < MAX_COLS; j++) {
        if (j >= C) break;
        if (p.x >= vS[(j - 1) * MAX_ROWS + r]) c = j;
    }
    return r * C + c;
}

float cellBaseOf(int cell, int hS[12]) {
    int C = gridCols();
    int r = cell / C;
    int c = cell - r * C;
    return r == 0 ? 0.0 : float(hS[(r - 1) * MAX_COLS + c] + 1);
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    int W = sz.x;
    int H = sz.y;
    int C = gridCols();
    int R = gridRows();
    float cellW = float(W) / float(C);
    float cellH = float(H) / float(R);
    int nCells = C * R;

    // PASS 0: params.
    //   row 0: (0,0)=mode/dims, (1..12,0)=hSeams, (13..24,0)=vSeams
    //   row 1: per-cell bboxes   row 2: per-cell feet anchors
    if (u_pass == 0) {
        bool metaRow = frag.y == 0 && frag.x < 25;
        bool bboxRow = frag.y == 1 && frag.x < nCells;
        bool anchRow = frag.y == 2 && frag.x < nCells;
        if (metaRow || bboxRow || anchRow) {
            bool alphaMode = false;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H || alphaMode) break;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    if (texelFetch(u_image1, ivec2(x, y), 0).a < 0.5) { alphaMode = true; break; }
                }
            }
            int hS[12]; int vS[12];
            computeSeams(W, H, cellW, cellH, alphaMode, hS, vS);

            if (metaRow) {
                if (frag.x == 0) {
                    fragColor0 = vec4(alphaMode ? 1.0 : 0.0, float(C) / 16.0, float(R) / 16.0, 1.0);
                } else if (frag.x <= 12) {
                    fragColor0 = vec4(float(hS[frag.x - 1]) / 4096.0, 0.0, 0.0, 1.0);
                } else {
                    fragColor0 = vec4(float(vS[frag.x - 13]) / 4096.0, 0.0, 0.0, 1.0);
                }
                return;
            }

            int cell = frag.x;
            int minX = MAXDIM; int maxX = -1; int minY = MAXDIM; int maxY = -1;
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H) break;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    ivec2 p = ivec2(x, y);
                    if (ownerOf(p, cellW, cellH, hS, vS) != cell) continue;
                    if (isFgColor(texelFetch(u_image1, p, 0), alphaMode)) {
                        minX = min(minX, x); maxX = max(maxX, x);
                        minY = min(minY, y); maxY = max(maxY, y);
                    }
                }
            }
            if (maxX < 0) { fragColor0 = vec4(0.0); return; }

            if (bboxRow) {
                fragColor0 = vec4(float(minX) / 4096.0, float(minY) / 4096.0,
                                  float(maxX) / 4096.0, float(maxY) / 4096.0);
                return;
            }
            // feet anchor: x-centroid of the character's lowest rows
            int bandTop = minY + max(int(0.18 * float(maxY - minY + 1)), 6);
            float sumX = 0.0;
            float cnt = 0.0;
            for (int y = 0; y < MAXDIM; y++) {
                if (y > bandTop || y >= H) break;
                if (y < minY) continue;
                for (int x = 0; x < MAXDIM; x++) {
                    if (x >= W) break;
                    ivec2 p = ivec2(x, y);
                    if (ownerOf(p, cellW, cellH, hS, vS) != cell) continue;
                    if (isFgColor(texelFetch(u_image1, p, 0), alphaMode)) {
                        sumX += float(x); cnt += 1.0;
                    }
                }
            }
            float anchorX = cnt > 0.0 ? sumX / cnt : (float(minX) + float(maxX)) * 0.5;
            fragColor0 = vec4(anchorX / 4096.0, 1.0, 0.0, 1.0);
            return;
        }
        fragColor0 = vec4(0.0);
        return;
    }

    // PASS 1: re-render onto clean nominal cells with one uniform scale.
    vec4 meta = texelFetch(u_image0, ivec2(0, 0), 0);
    bool alphaMode = meta.x > 0.5;
    float fill = u_float0 <= 0.0 ? 0.90 : u_float0;
    vec4 blank = alphaMode ? vec4(0.0) : vec4(1.0, 1.0, 1.0, 1.0);

    int hS[12]; int vS[12];
    for (int i = 0; i < 12; i++) {
        hS[i] = int(texelFetch(u_image0, ivec2(1 + i, 0), 0).r * 4096.0 + 0.5);
        vS[i] = int(texelFetch(u_image0, ivec2(13 + i, 0), 0).r * 4096.0 + 0.5);
    }

    vec4 bboxes[MAX_CELLS];
    float anchors[MAX_CELLS];
    bool valid[MAX_CELLS];
    bool anyValid = false;
    float ground = 1e9;
    for (int i = 0; i < MAX_CELLS; i++) {
        if (i >= nCells) { valid[i] = false; continue; }
        bboxes[i] = texelFetch(u_image0, ivec2(i, 1), 0) * 4096.0;
        vec4 av = texelFetch(u_image0, ivec2(i, 2), 0);
        anchors[i] = av.x * 4096.0;
        valid[i] = av.y > 0.5 && bboxes[i].z >= bboxes[i].x;
        if (valid[i]) {
            anyValid = true;
            ground = min(ground, bboxes[i].y - cellBaseOf(i, hS));
        }
    }
    if (!anyValid) { fragColor0 = texelFetch(u_image1, frag, 0); return; }

    float s = 1e9;
    for (int i = 0; i < MAX_CELLS; i++) {
        if (i >= nCells || !valid[i]) continue;
        float bw = bboxes[i].z - bboxes[i].x + 1.0;
        float vExt = u_bool0 ? (bboxes[i].w - bboxes[i].y + 1.0)
                             : max(bboxes[i].w - cellBaseOf(i, hS) - ground + 1.0, 1.0);
        s = min(s, fill * min(cellW / bw, cellH / vExt));
    }
    s = clamp(s, 0.25, 6.0);

    int col = clamp(int(float(frag.x) / cellW), 0, C - 1);
    int tRow = clamp(int(float(frag.y) / cellH), 0, R - 1);
    int i = tRow * C + col;
    if (!valid[i]) { fragColor0 = blank; return; }

    vec2 cellDst = vec2(float(col) * cellW, float(tRow) * cellH);
    float srcBase = cellBaseOf(i, hS);

    float pad = 0.5 * (1.0 - fill) * cellW;
    float dstLo = cellDst.x + pad + s * (anchors[i] - bboxes[i].x);
    float dstHi = cellDst.x + cellW - pad - s * (bboxes[i].z - anchors[i]);
    float anchorDstX = dstLo <= dstHi
        ? clamp(cellDst.x + cellW * 0.5, dstLo, dstHi)
        : (dstLo + dstHi) * 0.5;
    float srcX = anchors[i] + (float(frag.x) - anchorDstX) / s;

    float feetDstY = cellDst.y + BASELINE * cellH;
    float srcY = u_bool0
        ? bboxes[i].y + (float(frag.y) - feetDstY) / s
        : srcBase + ground + (float(frag.y) - feetDstY) / s;

    ivec2 src = ivec2(int(srcX + 0.5), int(srcY + 0.5));
    if (src.x < 0 || src.x >= W || src.y < 0 || src.y >= H
        || ownerOf(src, cellW, cellH, hS, vS) != i) {
        fragColor0 = blank;
    } else {
        fragColor0 = texelFetch(u_image1, src, 0);
    }
}
