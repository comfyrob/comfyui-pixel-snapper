#version 300 es
// Pixel Snapper (native GLSL) — stage 3/3: snap to grid (majority vote).
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = quantized image from stage 1 (becomes ping-pong after pass 0)
//   image1 = params texture from stage 2
//   image2 = palette stripes from stage 1 (IMAGE1 output)
//   u_int0 = color_count K (same value as stage 1)
// Output:
//   IMAGE0 = snapped image at full resolution: every grid cell painted
//            with its majority color (majority vote preserves dithering).
//
// Pass 0 computes each cell's winning palette color into the top-left
// corner of the buffer (one pixel per cell); pass 1 paints every output
// pixel by looking up its cell. For a true native-resolution asset,
// downscale this output by 1/pixel_size with nearest-neighbor.
#pragma passes 2

precision highp float;

const int MAX_K = 32;
const int MAX_CELL = 128; // max art-pixel size in source px

uniform sampler2D u_image0; // pass 0: quantized source; pass 1: cell map
uniform sampler2D u_image1; // params texture (step/phase, /2048)
uniform sampler2D u_image2; // palette stripes
uniform vec2 u_resolution;
uniform int u_int0;         // color_count K
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

int kColors() { return clamp(u_int0 <= 0 ? 16 : u_int0, 2, MAX_K); }

vec3 paletteColor(int i, int k) {
    return texture(u_image2, vec2((float(i) + 0.5) / float(k), 0.5)).rgb;
}

int nearestPalette(vec3 p, int k) {
    float best = 1e9;
    int bi = 0;
    for (int j = 0; j < MAX_K; j++) {
        if (j >= k) break;
        vec3 d = p - paletteColor(j, k);
        float dist = dot(d, d);
        if (dist < best) { best = dist; bi = j; }
    }
    return bi;
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    int k = kColors();
    vec4 params = texelFetch(u_image1, ivec2(0, 4), 0) * 2048.0;
    float stepX = max(params.x, 1.0);
    float phX = params.y;
    float stepY = max(params.z, 1.0);
    float phY = params.w;
    int W = int(u_resolution.x);
    int H = int(u_resolution.y);

    // Pixel x belongs to cell floor((x - phase) / step); cell ids are
    // shifted so the first in-image cell is 0.
    int cx0 = int(floor((0.0 - phX) / stepX));
    int cy0 = int(floor((0.0 - phY) / stepY));

    if (u_pass == 0) {
        // Fragment (cx, cy) = majority palette color of that cell.
        int cellX = frag.x + cx0;
        int cellY = frag.y + cy0;
        float xs = phX + float(cellX) * stepX;
        float xe = xs + stepX;
        float ys = phY + float(cellY) * stepY;
        float ye = ys + stepY;
        int x0 = max(int(ceil(xs - 0.0001)), 0);
        int x1 = min(int(ceil(xe - 0.0001)), W);
        int y0 = max(int(ceil(ys - 0.0001)), 0);
        int y1 = min(int(ceil(ye - 0.0001)), H);

        if (x0 >= x1 || y0 >= y1 || x0 >= W || y0 >= H) {
            fragColor0 = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        int counts[MAX_K];
        for (int j = 0; j < MAX_K; j++) counts[j] = 0;

        for (int dy = 0; dy < MAX_CELL; dy++) {
            int y = y0 + dy;
            if (y >= y1) break;
            for (int dx = 0; dx < MAX_CELL; dx++) {
                int x = x0 + dx;
                if (x >= x1) break;
                vec3 p = texelFetch(u_image0, ivec2(x, y), 0).rgb;
                counts[nearestPalette(p, k)]++;
            }
        }

        int bestCount = -1;
        int bestId = 0;
        for (int j = 0; j < MAX_K; j++) {
            if (j >= k) break;
            if (counts[j] > bestCount) { bestCount = counts[j]; bestId = j; }
        }
        fragColor0 = vec4(paletteColor(bestId, k), 1.0);
        return;
    }

    // PASS 1: paint every pixel with its cell's winning color.
    int cellX = int(floor((float(frag.x) - phX) / stepX)) - cx0;
    int cellY = int(floor((float(frag.y) - phY) / stepY)) - cy0;
    fragColor0 = vec4(texelFetch(u_image0, ivec2(cellX, cellY), 0).rgb, 1.0);
}
