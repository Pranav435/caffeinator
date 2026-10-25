extension SkyWindow {
    /// Volumetric clouds, ray-marched at half resolution and averaged over a few frames: a low layer whose kind,
    /// base and depth come from the weather (fair-weather cumulus, a stratocumulus deck, rain-bearing nimbostratus or a
    /// thunderstorm's towers), a thin mid-level sheet of altocumulus, and fibrous cirrus high above. Each sample is
    /// lit by sunlight or moonlight reddened by the air it crossed (and cut off in the Earth's shadow at dusk), with
    /// several orders of scattering, and seen through the haze in front of it.
    static let clouds = """
    // MARK: Noise, made once

    inline float3 cellHash(float3 c, float period) { return hash33(fmod(c + period * 64, period) * 1.618 + 0.5); }

    // Tileable gradient noise, about 0...1.
    inline float perlin3(float3 p, float period) {
        float3 i = floor(p), f = fract(p);
        float3 w = f * f * f * (f * (f * 6 - 15) + 10);
        float n = 0;
        for (int c = 0; c < 8; c++) {
            float3 o = float3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            float3 g = normalize(cellHash(i + o, period) * 2 - 1);
            float3 a = mix(1 - w, w, o);
            n += dot(g, f - o) * a.x * a.y * a.z;
        }
        return n * 0.85 + 0.5;
    }

    // Tileable Worley noise, inverted: 1 at the cell centres.
    inline float worley3(float3 p, float period) {
        float3 i = floor(p), f = fract(p);
        float d = 1;
        for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
            float3 o = float3(x, y, z);
            d = min(d, length(o + cellHash(i + o, period) - f));
        }
        return 1 - d;
    }

    inline float worleyFbm(float3 p, float f) {
        return worley3(p * f, f) * 0.625 + worley3(p * f * 2, f * 2) * 0.25 + worley3(p * f * 4, f * 4) * 0.125;
    }

    inline float remap(float v, float a, float b, float c, float d) { return c + (v - a) / (b - a) * (d - c); }

    // 128³ of cloud noise: Perlin-Worley for the billows, then Worley at three finer scales for erosion.
    kernel void cloudNoise(texture3d<float, access::write> out [[texture(0)]], uint3 g [[thread_position_in_grid]]) {
        float3 p = (float3(g) + 0.5) / 128.0;
        float perlin = 0, a = 1, total = 0;
        for (int o = 0; o < 4; o++) {
            float f = 4 * pow(2.0, float(o));
            perlin += perlin3(p * f, f) * a;
            total += a;
            a *= 0.5;
        }
        perlin /= total;
        float w4 = worleyFbm(p, 4), w8 = worleyFbm(p, 8), w16 = worleyFbm(p, 16);
        float pw = saturate(remap(perlin, w4 - 1, 1, 0, 1));
        out.write(float4(pw, w4, w8, w16), g);
    }

    // 256² of weather: where cloud may form, a second field for variety, and where it's thickest.
    kernel void cloudWeather(texture2d<float, access::write> out [[texture(0)]], uint2 g [[thread_position_in_grid]]) {
        float2 uv = (float2(g) + 0.5) / 256.0;
        float3 p = float3(uv, 0.37);
        float n = 0, a = 1, total = 0;
        for (int o = 0; o < 5; o++) {
            float f = 3 * pow(2.0, float(o));
            n += perlin3(p * f, f) * a;
            total += a;
            a *= 0.55;
        }
        n /= total;
        float cells = worleyFbm(p, 5);
        float cover = saturate((n - 0.5) * 2.6 + 0.5) * 0.7 + cells * 0.3;
        float other = perlin3(float3(uv, 0.71) * 6, 6) * 0.6 + perlin3(float3(uv, 0.13) * 12, 12) * 0.4;
        float thick = worleyFbm(float3(uv, 0.53), 3);
        out.write(float4(cover, other, thick, 1), g);
    }

    // MARK: Density

    constexpr sampler wrap3(filter::linear, address::repeat);

    // How much cloud the low layer holds at a point (planet coordinates, km), 0...1. `fine` adds the eroding detail.
    inline float lowCloud(float3 q, float h, constant Sky &k, texture3d<float> N, texture2d<float> W, bool fine) {
        float base = k.cumulus.x, thick = k.cumulus.y;
        float hf = (h - base) / thick;
        if (hf <= 0 || hf >= 1 || k.cover.x < 0.01) return 0;
        float2 xz = q.xz + k.wind.xy;
        float4 w = W.sample(wrap3, xz / 36.0);
        float cover = k.cover.x;
        float kind = k.cumulus.z;
        // The area that can hold cloud tracks the reported cover; full cover closes into a deck.
        float local = saturate(remap(w.r, 0.62 - cover * 0.7, 0.75 - cover * 0.5, 0, 1));
        local = mix(local, 1.0, smoothstep(0.82, 1.0, cover));
        // Height profiles: cumulus towers are rounded on top and flat underneath; layer clouds are slabs.
        float profile = kind < 0.5 ? smoothstep(0.0, 0.08, hf) * smoothstep(1.0, 0.25 + 0.4 * w.g, hf)
                      : kind < 1.5 ? smoothstep(0.0, 0.15, hf) * smoothstep(1.0, 0.55, hf)
                      : kind < 2.5 ? smoothstep(0.0, 0.06, hf) * smoothstep(1.0, 0.4, hf)
                      : smoothstep(0.0, 0.05, hf) * smoothstep(1.0, 0.8, hf);
        float scale = kind < 0.5 || kind > 2.5 ? 2.4 : 4.5;
        float3 pn = float3(xz.x, h * (kind < 0.5 ? 1.3 : 1.8), xz.y) / scale;
        float4 n = N.sample(wrap3, pn);
        float wf = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
        float shape = saturate(remap(n.r, wf - 1, 1, 0, 1));
        // Cumulus: higher up, only the strongest billows survive, so the tops come out as rounded heads.
        float need = kind < 0.5 || kind > 2.5 ? mix(0.0, 0.55, hf) * (1 - 0.6 * k.cumulus.w) : 0.0;
        float d = saturate(remap(shape * profile, 1 - local * mix(0.7, 0.95, cover) + need, 1, 0, 1)) * local;
        if (fine && d > 0) {
            float4 e = N.sample(wrap3, float3(xz.x, h * 2.2, xz.y) / 0.55 + 0.37);
            float detail = e.g * 0.6 + e.b * 0.3 + e.a * 0.1;
            float erode = mix(detail, 1 - detail, saturate(hf * 4));
            d = saturate(remap(d, erode * (kind < 0.5 ? 0.38 : 0.22), 1, 0, 1));
        }
        return d;
    }

    // Altocumulus: a thin sheet of small rounded puffs; altostratus when it closes up.
    inline float midCloud(float3 q, float h, constant Sky &k, texture3d<float> N, texture2d<float> W, bool fine) {
        float base = k.extra.z, thick = 0.6;
        float hf = (h - base) / thick;
        if (hf <= 0 || hf >= 1 || k.cover.y < 0.01) return 0;
        float2 xz = q.xz + k.wind.xy * 1.6;
        float4 w = W.sample(wrap3, xz / 50.0 + 0.31);
        float cover = k.cover.y;
        float local = saturate(remap(w.g, 0.7 - cover * 0.75, 0.82 - cover * 0.5, 0, 1));
        local = mix(local, 1.0, smoothstep(0.85, 1.0, cover));
        float profile = smoothstep(0.0, 0.3, hf) * smoothstep(1.0, 0.6, hf);
        float4 n = N.sample(wrap3, float3(xz.x, h * 2, xz.y) / 1.7 + 0.5);
        float cellsHere = n.b * 0.7 + n.a * 0.3;
        float d = saturate(remap(cellsHere * profile, 1 - local * 0.85, 1, 0, 1)) * local;
        return d;
    }

    // MARK: Lighting

    inline float hg(float c, float g) { return (1 - g * g) / (4 * PI * pow(max(1 + g * g - 2 * g * c, 1e-4), 1.5)); }

    // Light that reaches a point inside a cloud from the Sun or Moon: through the cloud above it toward the light
    // (several orders of scattering, after Wrenninge), and through the air before that.
    inline float3 cloudLight(float3 q, float3 l, float c, float tau, constant Sky &k, texture2d<float> T) {
        float r = length(q);
        float3 light = lightAt(T, r, dot(q / r, l));
        float3 sum = 0;
        float a = 1, b = 1, g = 1;
        for (int o = 0; o < 4; o++) {
            float phase = mix(hg(c, 0.8 * g), hg(c, -0.3 * g), 0.3);
            sum += a * exp(-b * tau) * phase;
            a *= 0.55; b *= 0.4; g *= 0.6;
        }
        return light * sum;
    }

    // MARK: The march

    struct CloudOut { float3 L; float T; float depth; };

    fragment float4 cloudMarch(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                               texture3d<float> N [[texture(0)]], texture2d<float> W [[texture(1)]],
                               texture2d<float> T [[texture(2)]], texture3d<float> haze [[texture(3)]],
                               texture2d<float> previous [[texture(4)]], const device float4 *amb [[buffer(2)]]) {
        float2 p = centered(in.uv, u.res);
        float3 rd = camRay(p, k);
        float jitter = fract(hash12(in.pos.xy) + float(u.frame % 64u) * 0.618034);
        float3 o = float3(0, RG + k.atmos.y, 0);
        float3 L = 0;
        float Tr = 1, dsum = 0, wsum = 0;
        bool sunLit = k.lightDir.w > 0.5;
        float3 l = k.lightDir.xyz;
        float E = k.lightDir.w > 0.5 ? k.sun.w : k.moon.w;
        float3 lightTint = sunLit ? float3(1) : float3(1.0, 0.95, 0.86);
        float c = dot(rd, l);
        float3 skyUp = amb[0].rgb, bounce = amb[2].rgb;
        float darken = k.cover.w;
        if (rd.y > -0.03) {
            for (int layer = 0; layer < 2; layer++) {
                float base = layer == 0 ? k.cumulus.x : k.extra.z;
                float top = layer == 0 ? k.cumulus.x + k.cumulus.y : k.extra.z + 0.6;
                float cover = layer == 0 ? k.cover.x : k.cover.y;
                if (cover < 0.01 || Tr < 0.02) continue;
                float t0 = hitSphere(o, rd, RG + base).y, t1 = hitSphere(o, rd, RG + top).y;
                if (t0 < 0 || t0 > 90) continue;
                // Near the horizon the ray grazes the layer for tens of kilometres; far away, detail averages out.
                t1 = min(t1, t0 + (layer == 0 ? 32.0 : 20.0));
                int steps = layer == 0 ? 30 : 10;
                float dt = (t1 - t0) / steps;
                float sigmaMax = layer == 0 ? (k.cumulus.z > 1.5 ? 22.0 : 20.0) : 7.0;
                for (int i = 0; i < steps; i++) {
                    float t = t0 + (i + jitter) * dt;
                    float3 q = o + rd * t;
                    float h = length(q) - RG;
                    bool fine = t < 25;
                    float d = layer == 0 ? lowCloud(q, h, k, N, W, fine) : midCloud(q, h, k, N, W, fine);
                    if (d < 0.003) continue;
                    float sigma = d * sigmaMax;
                    // Optical depth toward the light: four steps that grow with distance, as a cone.
                    float tau = 0;
                    float s0 = 0;
                    const float stepsTo[4] = { 0.08, 0.25, 0.7, 1.5 };
                    for (int j = 0; j < 4; j++) {
                        float sj = stepsTo[j];
                        float3 lq = q + l * sj;
                        float lh = length(lq) - RG;
                        float ld = layer == 0 ? lowCloud(lq, lh, k, N, W, false) : midCloud(lq, lh, k, N, W, false);
                        tau += ld * sigmaMax * (sj - s0);
                        s0 = sj;
                    }
                    float hf = saturate((h - base) / (top - base));
                    float3 direct = E * lightTint * cloudLight(q, l, c, tau, k, T);
                    // Dark, wet bases in rain; a little powder darkening deep inside.
                    float powder = 1 - 0.6 * exp(-sigma * 0.4);
                    float3 ambient = (skyUp * mix(0.45, 1.05, hf) * (1 - 0.6 * darken) + bounce * (1 - hf) * 0.6) * (0.35 + 0.65 * hf);
                    float3 S = sigma * (direct * powder * (1 - 0.75 * darken) + ambient);
                    // Lightning glows inside the storm.
                    if (k.weather.w > 0) S += sigma * float3(0.75, 0.8, 1.0) * k.weather.w * 6.0 * exp(-length(q.xz - k.seed.zw) * 0.4) * (1 - hf * 0.5);
                    float stepT = exp(-sigma * dt);
                    L += Tr * (S - S * stepT) / sigma;
                    float took = Tr * (1 - stepT);
                    dsum += took * t;
                    wsum += took;
                    Tr *= stepT;
                    if (Tr < 0.01) break;
                }
            }
            // Cirrus: thin fibres at about 9 km, as a single sheet.
            if (k.cover.z > 0.01 && Tr > 0.02) {
                float t = hitSphere(o, rd, RG + 9.0).y;
                float3 q = o + rd * t;
                float2 xz = q.xz + k.wind.xy * 2.5;
                float hd = k.wind.z;
                float2 along = float2(sin(hd), cos(hd)), across = float2(along.y, -along.x);
                float2 s = float2(dot(xz, along) / 5.0, dot(xz, across) / 2.6);
                s.x += (W.sample(wrap3, xz / 60.0).g - 0.5) * 0.7;
                float fibres = N.sample(wrap3, float3(s.x * 0.5, 0.3, s.y * 1.1)).r * 0.55 + N.sample(wrap3, float3(s.x * 1.5, 0.7, s.y * 3)).g * 0.45;
                float patch = W.sample(wrap3, xz / 70.0 + 0.6).r;
                fibres = mix(fibres, N.sample(wrap3, float3(xz / 1.3, 0.2)).g, 0.4);
                float d = saturate(remap(fibres * (0.4 + 0.6 * patch), 1 - k.cover.z * 0.75, 1.05, 0, 1)) * (1 - 0.7 * k.cover.x);
                d *= smoothstep(0.0, 0.08, rd.y + 0.02);
                if (d > 0.002) {
                    float tau = d * 0.9;
                    float3 light = E * lightTint * lightAt(T, length(q), dot(normalize(q), l));
                    float3 S = light * (hg(c, 0.75) * 0.8 + hg(c, -0.2) * 0.2) * 0.9 + skyUp * 0.9;
                    float a = 1 - exp(-tau);
                    L += Tr * S * a;
                    dsum += Tr * a * t;
                    wsum += Tr * a;
                    Tr *= 1 - a;
                }
            }
        }
        // The haze between us and the clouds.
        if (wsum > 1e-4) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float depth = dsum / wsum;
            float slice = sliceOf(depth);
            float4 h = haze.sample(s, float3(in.uv, (slice + 0.5) / SLICES));
            L = L * k.cam.w * h.a + h.rgb * (1 - Tr);
        }
        float4 now = float4(L, Tr);
        // Average with the last frames to smooth the jittered sampling; clouds drift slowly enough not to smear.
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float4 before = previous.sample(s, in.uv);
        return u.a.x > 0.5 ? mix(before, now, 0.12) : now;
    }
    """
}
