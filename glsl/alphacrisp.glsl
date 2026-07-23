#version 300 es
// Pixel Snapper companion — Alpha Crisp.
//
// Hardens a soft background-removal alpha into a pixel-art-ready binary
// cutout. Segmentation masks can bleed across empty background (e.g.
// bridging two vertically adjacent characters), so pass 0 INTERSECTS the
// mask with actual color content: opaque requires mask foreground AND a
// non-background pixel. Pass 1 is a 5x5 majority cleanup that kills
// leftover specks and refills small enclosed highlights (pixels within
// ~2px of colored content). Limitation: large interior regions the exact
// color of the background stay transparent.
//
// Node setup: core GLSLShader, size_mode = from_input.
//   image0 = RGBA image (e.g. BiRefNet output; consumed by ping-pong)
//   image1 = the SAME image again (stable copy)
//   u_float0 = alpha threshold (0 -> default 0.5)
#pragma passes 2

precision highp float;

const float WHITE_KEY_DIST = 0.06;

uniform sampler2D u_image0; // pass 0: source; pass 1: intersected ping-pong
uniform sampler2D u_image1; // source (stable)
uniform vec2 u_resolution;
uniform float u_float0;
uniform int u_pass;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

bool notBackground(vec3 rgb) {
    vec3 d = abs(rgb - vec3(1.0));
    return max(d.r, max(d.g, d.b)) > WHITE_KEY_DIST;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_image1, 0);
    float t = u_float0 <= 0.0 ? 0.5 : u_float0;
    vec4 src = texelFetch(u_image1, p, 0);

    // PASS 0: mask AND color content
    if (u_pass == 0) {
        bool opaque = src.a > t && notBackground(src.rgb);
        fragColor0 = vec4(src.rgb, opaque ? 1.0 : 0.0);
        return;
    }

    // PASS 1: majority cleanup on the intersected alpha
    int n = 0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            ivec2 q = p + ivec2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;
            if (texelFetch(u_image0, q, 0).a > 0.5) n++;
        }
    }
    bool prevOpaque = texelFetch(u_image0, p, 0).a > 0.5;
    bool opaque = (prevOpaque && n >= 6) || n >= 20;
    fragColor0 = vec4(src.rgb, opaque ? 1.0 : 0.0);
}
