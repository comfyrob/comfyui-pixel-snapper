#version 300 es
// Pixel Snapper (native GLSL) — stage 2/3: grid detection.
//
// Node setup: core GLSLShader, size_mode = custom, width 2048, height 8.
//   image0 = quantized image from stage 1 (becomes ping-pong after pass 0)
//   image1 = the SAME quantized image again (stable copy)
//   u_int0 = pixel_size override (0 = auto-detect)
// Output:
//   IMAGE0 = params texture; every pixel holds
//            (step_x, phase_x, step_y, phase_y) / 2048 as float RGBA.
//
// Method — a GLSL port of the Python estimator (jitter-robust, unlike
// autocorrelation which locks onto harmonics):
//   gradient profiles per axis -> strong local peaks (plateau-tolerant,
//   min 4px apart) -> median peak spacing = art-pixel size ->
//   phase = offset whose grid lines collect the most gradient energy.
// The whole estimator runs sequentially in one fragment of pass 1.
//
// Ping-pong layout: row 0 = column profile, row 1 = row profile,
// row 4 px 0 = resolved params. Supports sources up to 2048 px per side.
#pragma passes 3

precision highp float;

const int MAXDIM = 2048;
const int MAX_DIFFS = 700;
const float PEAK_THRESHOLD = 0.2;  // fraction of profile max
const int PEAK_MIN_DIST = 4;       // matches peak_distance_filter
const float MAX_STEP_RATIO = 1.8;

uniform sampler2D u_image0; // pass 0: quantized source; pass 1+: ping-pong
uniform sampler2D u_image1; // quantized source (stable)
uniform vec2 u_resolution;
uniform int u_int0;         // pixel_size override, 0 = auto
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

float grayAt(ivec2 p) {
    vec3 c = texelFetch(u_image1, p, 0).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float prof(int x, int axis) { return texelFetch(u_image0, ivec2(x, axis), 0).r; }

bool isPeak(int i, int axis, float threshold) {
    float v = prof(i, axis);
    return v > threshold && v >= prof(i - 1, axis) && v > prof(i + 1, axis);
}

// Number of kept-peak spacings <= v (v < 0: count all spacings).
// Kept peaks follow the Python rule: first strong peak always kept, later
// ones only when > PEAK_MIN_DIST-1 past the last kept one.
int countSpacings(int axis, int n, float threshold, int v) {
    int count = 0;
    int last = -1;
    for (int i = 1; i < MAXDIM; i++) {
        if (i >= n - 1) break;
        if (isPeak(i, axis, threshold)) {
            if (last < 0) {
                last = i;
            } else if (i - last > PEAK_MIN_DIST - 1) {
                if (v < 0 || i - last <= v) count++;
                last = i;
            }
        }
    }
    return count;
}

// Median peak spacing of one axis profile; 0.0 if no reliable estimate.
// Arrayless on purpose: large local arrays miscompiled in fragment shaders.
float estimateStep(int axis, int n) {
    float maxVal = 0.0;
    for (int i = 0; i < MAXDIM; i++) {
        if (i >= n) break;
        maxVal = max(maxVal, prof(i, axis));
    }
    if (maxVal <= 0.0) return 0.0;
    float threshold = maxVal * PEAK_THRESHOLD;

    int m = countSpacings(axis, n, threshold, -1);
    if (m < 1) return 0.0;

    // upper median = (m/2 + 1)-th smallest spacing = smallest value v with
    // count(spacings <= v) >= m/2 + 1  (spacings are integers)
    int target = m / 2 + 1;
    for (int v = PEAK_MIN_DIST; v <= MAX_DIFFS; v++) {
        if (countSpacings(axis, n, threshold, v) >= target) return float(v);
    }
    return 0.0;
}

// Offset in [0, step) whose grid lines collect the most gradient energy.
float bestPhase(float st, int axis, int n) {
    int sti = max(int(st + 0.5), 1);
    float bestScore = -1.0;
    int bestOff = 0;
    for (int off = 0; off < 512; off++) {
        if (off >= sti) break;
        float s = 0.0;
        for (int j = 0; j < MAXDIM; j++) {
            int pos = int(float(off) + float(j) * st + 0.5);
            if (pos >= n) break;
            s += prof(pos, axis);
        }
        if (s > bestScore) { bestScore = s; bestOff = off; }
    }
    return float(bestOff);
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    int W = sz.x;
    int H = sz.y;

    // PASS 0: summed absolute central-difference gradients.
    if (u_pass == 0) {
        float v = 0.0;
        if (frag.y == 0 && frag.x >= 1 && frag.x < min(W - 1, MAXDIM)) {
            for (int y = 0; y < MAXDIM; y++) {
                if (y >= H) break;
                v += abs(grayAt(ivec2(frag.x + 1, y)) - grayAt(ivec2(frag.x - 1, y)));
            }
        } else if (frag.y == 1 && frag.x >= 1 && frag.x < min(H - 1, MAXDIM)) {
            for (int x = 0; x < MAXDIM; x++) {
                if (x >= W) break;
                v += abs(grayAt(ivec2(x, frag.x + 1)) - grayAt(ivec2(x, frag.x - 1)));
            }
        }
        fragColor0 = vec4(v, 0.0, 0.0, 1.0);
        return;
    }

    // PASS 1: full estimator in one fragment -> row 4 pixel 0.
    if (u_pass == 1) {
        if (frag.y == 4 && frag.x == 0) {
            float sx = estimateStep(0, W);
            float sy = estimateStep(1, H);

            if (u_int0 > 0) {
                sx = float(u_int0); sy = float(u_int0);
            } else if (sx <= 0.0 && sy <= 0.0) {
                sx = max(float(min(W, H)) / 64.0, 1.0); sy = sx;
            } else if (sx <= 0.0) { sx = sy; }
            else if (sy <= 0.0) { sy = sx; }
            else {
                float ratio = max(sx / sy, sy / sx);
                if (ratio > MAX_STEP_RATIO) { sx = min(sx, sy); sy = sx; }
                else { sx = (sx + sy) * 0.5; sy = sx; }
            }

            float phX = bestPhase(sx, 0, W);
            float phY = bestPhase(sy, 1, H);
            fragColor0 = vec4(sx / 2048.0, phX / 2048.0, sy / 2048.0, phY / 2048.0);
        } else {
            fragColor0 = texelFetch(u_image0, frag, 0);
        }
        return;
    }

    // FINAL PASS: plain copy — profiles stay in rows 0/1, params in row 4.
    // Consumers texelFetch absolute rows (upload/readback flips are
    // symmetric, so GL row addressing survives node hops).
    fragColor0 = texelFetch(u_image0, frag, 0);
}
