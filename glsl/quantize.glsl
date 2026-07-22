#version 300 es
// Pixel Snapper (native GLSL) — stage 1/3: k-means color quantization.
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = source image (becomes the ping-pong buffer after pass 0)
//   image1 = the SAME source image again (stable copy for the final pass)
//   u_int0 = color_count K (2..32, 0 -> 16)
// Outputs:
//   IMAGE0 = quantized image (every pixel snapped to its nearest centroid)
//   IMAGE1 = palette stripes (K vertical bands; stage 3 samples these)
//
// Working state lives in the ping-pong buffer: row 0 cols 0..K-1 hold the
// centroids, rows 1..GRID hold a GRID x GRID downsample of the source used
// for the k-means iterations.
#pragma passes 13

precision highp float;

const int MAX_K = 32;
const int GRID = 64;        // cached source samples per axis
const int NUM_PASSES = 13;  // keep equal to `#pragma passes` above

uniform sampler2D u_image0; // pass 0: source; pass 1+: ping-pong buffer
uniform sampler2D u_image1; // source (stable across passes)
uniform vec2 u_resolution;
uniform int u_int0;         // color_count K
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;

int kColors() { return clamp(u_int0 <= 0 ? 16 : u_int0, 2, MAX_K); }

vec3 readCentroid(int i) { return texelFetch(u_image0, ivec2(i, 0), 0).rgb; }
vec3 readCache(int sx, int sy) { return texelFetch(u_image0, ivec2(sx, sy + 1), 0).rgb; }

int nearestCentroid(vec3 p, int k) {
    float best = 1e9;
    int bi = 0;
    for (int j = 0; j < MAX_K; j++) {
        if (j >= k) break;
        vec3 d = p - readCentroid(j);
        float dist = dot(d, d);
        if (dist < best) { best = dist; bi = j; }
    }
    return bi;
}

void main() {
    int k = kColors();
    ivec2 frag = ivec2(gl_FragCoord.xy);

    // PASS 0: cache a GRID x GRID downsample of the source (rows 1..GRID).
    if (u_pass == 0) {
        if (frag.y >= 1 && frag.y <= GRID && frag.x < GRID) {
            vec2 uv = (vec2(float(frag.x), float(frag.y - 1)) + 0.5) / float(GRID);
            fragColor0 = vec4(texture(u_image0, uv).rgb, 1.0);
        } else {
            fragColor0 = vec4(0.0, 0.0, 0.0, 1.0);
        }
        fragColor1 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // PASS 1: farthest-point seeding. Fragment (j, 0) replays the greedy
    // maximin sequence over the cached samples and writes centroid j.
    // Deterministic, and unlike random scatter it will not start two
    // centroids inside the same dominant color.
    if (u_pass == 1) {
        if (frag.y == 0 && frag.x < k) {
            vec3 chosen[MAX_K];
            chosen[0] = readCache(0, 0);
            for (int s = 1; s < MAX_K; s++) {
                if (s > frag.x) break;
                float bestD = -1.0;
                vec3 bestP = chosen[0];
                for (int sy = 0; sy < GRID; sy++) {
                    for (int sx = 0; sx < GRID; sx++) {
                        vec3 p = readCache(sx, sy);
                        float dmin = 1e9;
                        for (int t = 0; t < MAX_K; t++) {
                            if (t >= s) break;
                            vec3 d = p - chosen[t];
                            dmin = min(dmin, dot(d, d));
                        }
                        if (dmin > bestD) { bestD = dmin; bestP = p; }
                    }
                }
                chosen[s] = bestP;
            }
            fragColor0 = vec4(chosen[frag.x], 1.0);
        } else {
            fragColor0 = texelFetch(u_image0, frag, 0);
        }
        fragColor1 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // PASSES 2..N-2: one k-means iteration over the cached samples.
    if (u_pass < NUM_PASSES - 1) {
        if (frag.y == 0 && frag.x < k) {
            int c = frag.x;
            vec3 sum = vec3(0.0);
            float cnt = 0.0;
            for (int sy = 0; sy < GRID; sy++) {
                for (int sx = 0; sx < GRID; sx++) {
                    vec3 p = readCache(sx, sy);
                    if (nearestCentroid(p, k) == c) { sum += p; cnt += 1.0; }
                }
            }
            fragColor0 = vec4(cnt > 0.0 ? sum / cnt : readCentroid(c), 1.0);
        } else {
            fragColor0 = texelFetch(u_image0, frag, 0);
        }
        fragColor1 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // FINAL PASS: map every source pixel to its nearest centroid, and
    // emit the palette as K vertical stripes on IMAGE1.
    vec3 src = texture(u_image1, v_texCoord).rgb;
    fragColor0 = vec4(readCentroid(nearestCentroid(src, k)), 1.0);
    int band = clamp(int(v_texCoord.x * float(k)), 0, k - 1);
    fragColor1 = vec4(readCentroid(band), 1.0);
}
