extension SkyWindow {
    /// Metal for what's outside: the sky's own picture (the scattered light, the real Milky Way and stars, the Sun and
    /// the Moon), then the land in receding layers, each seen through the haze at its own distance: mountains 25 km off,
    /// wooded hills at 6 km, a town at 1.5 km and 450 m, rooftops below, and trees and telephone wires close by.
    /// Rain and snow fall in front.
    static let land = """
    // MARK: Sky, stars, Sun and Moon

    fragment float4 skyBase(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                            texture2d<float> sky [[texture(0)]], texture2d<float> T [[texture(1)]],
                            texture2d<float> milky [[texture(2)]], texture2d<float> moonMap [[texture(3)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float3 rd = camRay(p, k);
        float pixel = 2 / (u.res.y * k.cam.z);
        float alt = asin(clamp(rd.y, -1.0, 1.0));
        float E = k.cam.w;
        float3 c = sky.sample(s, in.uv).rgb / E + nightGlow(alt, k);
        // What survives the trip through the whole atmosphere along this ray: starlight dims and reddens low down.
        float3 through = lightAt(T, RG + k.atmos.y, rd.y);
        if (k.night.x > 0 && milky.get_width() > 1) {
            // ESO's photograph of the real Milky Way, two mip levels soft so its own stars become glow; the crisp
            // stars are drawn on top from the catalogue.
            float3 mw = milkyWay(milky, toEquatorial(rd, k), pixel * 4);
            mw = max(mw - 0.012, 0.0);
            c += mw * 1.6e-6 * k.night.x * through;
        }
        // The Sun: limb-darkened, as bright as the sensor allows.
        float sr = 0.00465;
        float sd = acos(clamp(dot(rd, k.sun.xyz), -1.0, 1.0)) / sr;
        if (sd < 1.5) {
            float mu = sqrt(max(1 - sd * sd, 0.0));
            float disc = 1 - smoothstep(1 - pixel / sr, 1 + pixel / sr, sd);
            float3 L = k.sun.w * through / (PI * sr * sr) * (0.4 + 0.6 * mu);
            c += min(L, float3(90 / E)) * disc;

        }
        // The Moon, held a little under clipping so its seas still show.
        float cover;
        float3 moon = moonDisc(rd, k, pixel, moonMap, moonMap.get_width() > 1, cover) * k.sun.w / PI * through;
        float cap = 3.5 / E;
        moon = moon * cap / (cap + moon);
        c = mix(c, c * 0.3 + moon * 1.6, cover);
        // Out exposed, like everything in the frame buffer.
        return float4(c * E, 1);
    }

    // Catalogue stars and the planets as point sprites, with their real brightness and colour, dimmed and reddened by
    // the air near the horizon and twinkling a little there.
    vertex P skyStar(uint id [[vertex_id]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                     const device float4 *stars [[buffer(2)]], texture2d<float> T [[texture(0)]]) {
        float4 a = stars[id * 2], b = stars[id * 2 + 1];
        bool planet = b.w > 0.5;
        // Stars are stored as J2000 directions; planets already in the horizon frame.
        float3 d = planet ? a.xyz : a.x * k.eq0.xyz + a.y * k.eq1.xyz + a.z * k.eq2.xyz;
        P o;
        float3 f = forward(k);
        float z = dot(d, f);
        o.pos = float4(dot(d, float3(f.z, 0, -f.x)) * k.cam.z / (u.res.x / u.res.y), d.y * k.cam.z - lensShift(k) * z, z * 0.001, z);
        float3 color = planet ? b.xyz : blackbody(4600 * (1 / (0.92 * b.x + 1.7) + 1 / (0.92 * b.x + 0.62)));
        float3 through = lightAt(T, RG + k.atmos.y, d.y);
        float low = 1 - saturate(d.y * 4);
        float twinkle = planet ? 1 : 1 + 0.25 * low * sin(u.time * (2.1 + fract(id * 0.618) * 2.5) + id);
        // Illuminance from magnitude (the Sun is −26.74), spread over the sprite so its total light is right.
        float pixel = 2 / (u.res.y * k.cam.z);
        float flux = k.sun.w * pow(10.0, -0.4 * (a.w + 26.74)) / (pixel * pixel);
        float size = clamp(2.2 + 1.4 * log2(max(flux * k.cam.w, 1.0)) * 0.5, 2.2, 9.0) * u.res.y / 1000;
        o.size = size;
        float vis = k.night.x * (1 - 0.7 * k.weather.z);
        o.color = float4(color * through * flux * twinkle * vis * k.cam.w * 6.0 / (0.14 * size * size), 0);
        if (o.pos.w <= 0 || vis <= 0 || d.y < -0.01) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    // MARK: Profiles

    inline float n1(float x, float period) {
        float i = floor(x), f = fract(x);
        float w = f * f * (3 - 2 * f);
        return mix(hash11(fmod(i, period) + 0.5), hash11(fmod(i + 1, period) + 0.5), w);
    }

    inline float f1(float t, float base, int octaves, float seed) {
        float v = 0, a = 0.5, f = base;
        for (int i = 0; i < octaves; i++) { v += a * n1(t * f + seed * 7.1, f); f *= 2; a *= 0.5; }
        return v;
    }

    inline float ridged(float t, float base, float seed) {
        float v = 0, a = 0.55, f = base;
        for (int i = 0; i < 6; i++) { float n = 1 - abs(n1(t * f + seed * 3.3, f) * 2 - 1); v += a * n * n; f *= 2.1; a *= 0.5; }
        return v;
    }

    // One layer of land: how much of the pixel it covers, its colour before haze, and its distance.
    struct Layer { float cover; float3 color; float dist; float3 glow; };

    // Light falling on the land: the sky above, the sky behind the camera on faces turned toward us, and the Sun or
    // Moon where a face turns toward it.
    inline float3 lit(float3 albedo, float3 n, constant Sky &k, const device float4 *amb) {
        float3 sun = amb[3].rgb;
        float3 l = k.lightDir.xyz;
        float sunFacing = saturate(dot(n, l)) / max(l.y, 0.08);
        // Seen from indoors the outside is a stop or two under the sky, but never black: a gain stands in for light bounced from the ground and neighbours.
        return 1.5 * albedo * (amb[0].rgb * saturate(n.y * 0.5 + 0.5) + amb[1].rgb * saturate(-dot(n, forward(k)) * 0.8 + 0.2)
                         + sun * sunFacing / PI + amb[2].rgb * 0.25);
    }

    // Distant mountains, an elevation angle (radians) for an azimuth in turns.
    float mountains(float t, float seed, float amount) { return rad(0.08 + 1.5 * amount * ridged(t, 5, seed)); }

    // Wooded hills with a ragged canopy edge.
    float hills(float t, float seed) {
        float v = f1(t, 9, 4, seed);
        return rad(0.12 + 0.75 * v * v) + rad(0.025) * n1(t * 1800 + seed, 1800) + rad(0.012) * n1(t * 5200, 5200);
    }

    // A strip of town: houses with pitched or flat roofs, chimneys, the odd taller block or spire, trees between.
    // Returns the signed angle above the roofline (positive inside), and which building and part this is.
    float town(float t, float alt, float cells, float scale, float seed, thread float3 &info) {
        float x = t * cells;
        float c = floor(x);
        float best = rad(0.06 * scale) - alt;
        info = float3(-1, 0, 0);
        float span = TAU / cells;
        for (int j = -1; j <= 1; j++) {
            float id = fmod(c + j + cells, cells);
            float lx = x - (c + j);
            float h = hash11(id * 3.17 + seed);
            float kind = hash11(id * 5.11 + seed + 1);
            if (h < 0.12) continue;
            if (kind < 0.3) {
                // A tree: a ragged crown on a short trunk.
                float cx = 0.5 + (hash11(id + 2.0 + seed) - 0.5) * 0.5;
                float R = rad(scale * (0.3 + 0.3 * hash11(id + 5.0 + seed)));
                float top = R * 2.1 + rad(scale * 0.25 * hash11(id + 6.0 + seed));
                float2 q = float2((lx - cx) * span, alt - (top - R)) / R;
                float ragged = 1 + 0.55 * (fbm(q * 3.0 + id * 3.1, 4) - 0.5) + 0.3 * (fbm(q * 9.0 + id * 1.7, 3) - 0.5);
                float crown = (ragged - length(q * float2(0.9, 1.05))) * R;
                if (crown > best) { best = crown; info = float3(id, 2, 0); }
                continue;
            }
            float w0 = 0.04 + 0.2 * hash11(id + 1.0 + seed), w1 = 0.96 - 0.2 * hash11(id + 4.0 + seed);
            if (lx < w0 || lx > w1) continue;
            bool tall = kind > 0.93;
            bool spire = kind > 0.905 && kind <= 0.93;
            float wall = rad(scale * (0.35 + 0.45 * h * h + (tall ? 1.1 : 0)));
            float mid = (w0 + w1) * 0.5, hw = (w1 - w0) * 0.5;
            float across = 1 - abs(lx - mid) / hw;
            float roof = wall;
            float style = hash11(id * 9.7 + seed);
            if (tall) roof += across > 0.05 ? rad(scale * 0.03) : 0;
            else if (style < 0.6) roof += rad(scale * (0.22 + 0.25 * hash11(id + 8.0 + seed))) * across;
            else if (style < 0.8) roof += rad(scale * 0.2) * min(1.0, across * 2.5);
            if (spire) roof = max(roof, wall + rad(scale * 1.6) * saturate(1 - abs(lx - mid) * 9) );
            float chim = mid + hw * (hash11(id + 9.0 + seed) > 0.5 ? 0.45 : -0.4);
            if (!tall && hash11(id + 10.0 + seed) > 0.45 && abs(lx - chim) * span < rad(scale * 0.05)) roof = max(roof, wall + rad(scale * 0.38));
            if (hash11(id + 11.0 + seed) > 0.8 && abs(lx - mid - 0.1) * span < rad(scale * 0.008)) roof = max(roof, wall + rad(scale * 0.75));
            // Television aerials, water tanks on stilts and ragged parapets, as on any real skyline.
            float2 ex = float2(hash11(id + 12.0 + seed), hash11(id + 13.0 + seed));
            if (!tall && ex.x > 0.7 && abs(lx - mid - (ex.y - 0.5) * hw) * span < rad(scale * 0.035)) roof = max(roof, wall + rad(scale * (0.4 + 0.9 * ex.y)));
            if (!tall && ex.x > 0.35 && ex.x < 0.5 && abs(lx - mid + 0.2 * hw) * span < rad(scale * 0.14)) roof = max(roof, wall + rad(scale * (0.14 + 0.1 * ex.y)));
            roof += rad(scale * 0.02) * (n1(lx * 14 + id * 5.0, 14) - 0.5);
            float inside = min(roof - alt, min(lx - w0, w1 - lx) * span);
            if (inside > best) {
                best = inside;
                // Which part: wall (with its height above the roof eaves) or roof.
                info = float3(id, alt < wall ? 0 : 1, (alt - wall) / max(roof - wall, 1e-4));
                info.z = alt < wall ? (wall - alt) / rad(scale) : info.z;
            }
        }
        return best;
    }

    // Windows lit in the evening: a grid on each wall, some warm, a few cool.
    float3 windowsLit(float t, float alt, float cells, float scale, float id, float hour, float amount) {
        float x = fract(t * cells) * 6;
        float y = alt / rad(scale * 0.14);
        float2 cell = floor(float2(x, y)), f = fract(float2(x, y));
        if (f.x < 0.3 || f.x > 0.7 || f.y < 0.3 || f.y > 0.75) return 0;
        float on = hash12(cell + id * 13.1);
        if (on > amount) return 0;
        float warm = hash12(cell + 7.3 + id);
        return mix(float3(1.0, 0.62, 0.3), float3(0.75, 0.85, 1.0), step(0.85, warm)) * (0.5 + 0.5 * hash12(cell + 3.1));
    }

    // MARK: Near trees and wires

    // How much a thin segment covers a pixel when the lens is focused on the sky and the branch is out of focus.
    inline float cover(float d, float w, float soft) {
        return smoothstep(w + soft, max(w - soft, 0.0), d) * min(1.0, (w + soft) / (soft * 2.0));
    }
    inline float segD(float2 p, float2 a, float2 b, thread float &h) {
        float2 pa = p - a, ba = b - a;
        h = saturate(dot(pa, ba) / dot(ba, ba));
        return length(pa - ba * h);
    }
    inline float2 tip(float2 a, float ang, float len) { return a + float2(cos(ang), sin(ang)) * len; }

    // A branch drawn as two straight stretches with a small kink between them; `t` is the position along it, 0 to 1.
    inline float2 chainAt(float2 a, float ang, float len, float kink, float t) {
        float2 m = tip(a, ang, len * 0.55);
        return t < 0.55 ? mix(a, m, t / 0.55) : mix(m, tip(m, ang + kink, len * 0.45), (t - 0.55) / 0.45);
    }
    inline float chainD(float2 p, float2 a, float ang, float len, float kink, thread float &t) {
        float2 m = tip(a, ang, len * 0.55), e = tip(m, ang + kink, len * 0.45);
        float h1, h2;
        float d1 = segD(p, a, m, h1), d2 = segD(p, m, e, h2);
        if (d1 < d2) { t = h1 * 0.55; return d1; }
        t = 0.55 + h2 * 0.45;
        return d2;
    }
    // Wood coverage of a tapering branch of base half-width w.
    inline float wood(float d, float w, float t, float soft) { return cover(d, w * (1.0 - 0.7 * t), soft); }

    // Boughs of a tree near the window, entering from the upper right corner: two limbs branching into branches, twigs
    // and sprigs that taper, bend and kink where shoots leave, with a few crisp leaves on the finest twigs. Everything
    // sways in the wind, the tips most; the lens is focused on the sky, so it is softly out of focus.
    // p is in screen heights with y down. Returns (wood, leaf) coverage.
    float2 boughs(float2 p, float aspect, float seed, float time, float sway, float soft) {
        float wd = 0, leaf = 0;
        p += 0.012 * (float2(fbm(p * 3.5 + seed, 3), fbm(p * 3.5 + seed + 9.1, 3)) - 0.5);
        float2 anchor = float2(aspect + 0.05, 0.03 + 0.14 * hash11(seed));
        for (int a = 0; a < 2; a++) {
            float h0 = hash11(seed + a * 3.3);
            float ang0 = PI - 0.2 - 0.6 * float(a) + (h0 - 0.5) * 0.25;
            float L = 0.34 + 0.14 * h0;
            float2 c0 = anchor;
            for (int s = 0; s < 3; s++) {
                float ha = hash11(seed + a * 11.0 + s * 1.7);
                float ang = ang0 + (ha - 0.5) * 0.5 + sway * sin(time * (0.4 + ha * 0.3) + ha * 9.0) * (0.4 + 0.35 * float(s));
                ang0 = ang;
                float2 c1 = tip(c0, ang, L / 3);
                float h;
                float d = segD(p, c0, c1, h);
                wd = max(wd, wood(d, 0.0085, (float(s) + h) / 3.0, soft));
                if (d < L * 0.9) {
                    for (int sd = 0; sd < 2; sd++) {
                        float hb = hash11(seed + a * 5.0 + s * 3.1 + sd * 1.3 + 7.0);
                        float angb = ang + (sd == 0 ? -1.0 : 1.0) * (0.5 + 0.5 * hb) + sway * sin(time * 0.6 + hb * 8.0) * 0.9;
                        float Lb = L * (0.5 + 0.2 * hb), kb = (hash11(hb * 91.0) - 0.5) * 0.9;
                        float2 b0 = mix(c0, c1, 0.4 + 0.5 * hb);
                        d = chainD(p, b0, angb, Lb, kb, h);
                        wd = max(wd, wood(d, 0.0042, h, soft));
                        if (d < Lb * 0.9) {
                            for (int sb = 0; sb < 2; sb++) {
                                float hc = hash11(seed + a * 13.0 + s * 4.3 + sd * 2.9 + sb * 1.1 + 21.0);
                                float angc = angb + (sb == 0 ? -1.0 : 1.0) * (0.45 + 0.5 * hc) + sway * sin(time * 0.8 + hc * 7.0) * 1.2;
                                float Lc = Lb * (0.5 + 0.15 * hc), kc = (hash11(hc * 77.0) - 0.5) * 1.0;
                                float2 e0 = chainAt(b0, angb, Lb, kb, 0.3 + 0.55 * hc);
                                d = chainD(p, e0, angc, Lc, kc, h);
                                wd = max(wd, wood(d, 0.0021, h, soft));
                                if (d < Lc * 0.9) {
                                    for (int tw = 0; tw < 3; tw++) {
                                        float hd = hash11(seed + a * 17.0 + s * 6.1 + sd * 3.7 + sb * 2.3 + tw * 1.9 + 33.0);
                                        float angd = angc + (float(tw) - 1.0) * 0.7 + (hd - 0.5) * 0.4 + sway * sin(time * 1.1 + hd * 6.0) * 1.6;
                                        float Ld = Lc * (0.5 + 0.25 * hd);
                                        float2 f0 = chainAt(e0, angc, Lc, kc, 0.25 + 0.3 * float(tw));
                                        float kd = (hd - 0.5) * 0.8;
                                        float dd = chainD(p, f0, angd, Ld, kd, h);
                                        wd = max(wd, wood(dd, 0.0011, h, soft) * 0.85);
                                        if (dd < 0.02) {
                                            // A few crisp leaves along the twig, alternate sides.
                                            for (int l = 0; l < 4; l++) {
                                                float hl = hash11(hd * 53.0 + float(l) * 7.7);
                                                if (hl < 0.3) continue;
                                                float2 base = chainAt(f0, angd, Ld, kd, 0.3 + 0.2 * float(l));
                                                float la = angd + (l % 2 == 0 ? -1.0 : 1.0) * (0.7 + 0.5 * hl) + sway * sin(time * 1.3 + hl * 9.0) * 2.0;
                                                float2 c = base + float2(cos(la), sin(la)) * 0.0075;
                                                float2 q = p - c;
                                                q = float2(cos(la) * q.x + sin(la) * q.y, -sin(la) * q.x + cos(la) * q.y);
                                                float e = length(q / float2(0.0085 * (0.8 + 0.4 * hl), 0.0036));
                                                leaf = max(leaf, smoothstep(1.0 + soft * 40.0, 1.0 - soft * 40.0, e));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                c0 = c1;
            }
        }
        return float2(wd, leaf);
    }

    // MARK: Falling weather

    float rainStreaks(float2 p, float time, float slant, float amount) {
        float a = 0;
        for (int i = 0; i < 4; i++) {
            float s = 22 + i * 17;
            float2 q = float2(p.x + p.y * slant, p.y) * float2(s, s * 0.09);
            q.y += time * (2.4 - i * 0.45) * s * 0.09;
            float col = floor(q.x);
            float r = hash11(col * 1.13 + i * 17);
            if (r > amount * 0.7 + 0.08) continue;
            float x = fract(q.x) - (0.3 + 0.4 * hash11(col + 3.3));
            float y = fract(q.y + r * 13);
            float width = 0.035 + 0.05 * (3 - i) / 3;
            a += smoothstep(width, 0.0, abs(x)) * smoothstep(0, 0.1, y) * smoothstep(0.35, 0.12, y) * (0.22 - i * 0.04);
        }
        return a;
    }

    // Snowflakes, the near ones big and soft because the lens is focused far beyond them.
    float snowFlakes(float2 p, float time, float drift, float amount) {
        float a = 0;
        for (int i = 0; i < 5; i++) {
            float s = 3 + i * 3.5;
            float2 q = p * s;
            q.y += time * (0.35 - i * 0.05) * s * 0.4;
            q.x += time * drift * s * 0.1 + sin(q.y * 0.7 + i) * 0.3;
            float2 cell = floor(q);
            float h = hash12(cell + i * 31);
            if (h > amount * 0.7 + 0.08) continue;
            float2 at = cell + 0.2 + 0.6 * hash22(cell + i * 7);
            at.x += 0.15 * sin(time * (0.5 + h) + h * 30);
            float r = (0.05 + 0.04 * hash12(cell + 2.1)) * (i == 0 ? 2.2 : 1.0);
            float soft = i == 0 ? 0.0 : 0.4;
            a += smoothstep(r, r * soft, length(q - at)) * (i == 0 ? 0.35 : 0.85 - i * 0.12);
        }
        return a;
    }

    // MARK: Everything in front of the sky

    inline float4 hazeAt(texture3d<float> haze, float2 uv, float d) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        return haze.sample(s, float3(uv, (sliceOf(d) + 0.5) / SLICES));
    }

    // Adds a layer behind what's already in front (front-to-back): colour through the haze at its distance. The
    // haze is stored exposed; `E` exposes the colour to match.
    inline void under(thread float4 &acc, float cover, float3 color, float d, float2 uv, texture3d<float> haze, float E) {
        if (cover <= 0 || acc.a >= 0.999) return;
        float4 h = hazeAt(haze, uv, d);
        float3 c = color * E * h.a + h.rgb;
        acc.rgb += (1 - acc.a) * cover * c;
        acc.a += (1 - acc.a) * cover;
    }

    fragment float4 skyFront(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                             texture2d<float> clouds [[texture(0)]], texture3d<float> haze [[texture(1)]],
                             texture2d<float> sky [[texture(2)]], const device float4 *amb [[buffer(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = in.uv;
        float2 p = centered(uv, u.res);
        float3 rd = camRay(p, k);
        float pixel = 2 / (u.res.y * k.cam.z);
        float alt = asin(clamp(rd.y, -1.0, 1.0));
        float turn = fract(atan2(rd.x, rd.z) / TAU + 1);
        float seed = k.seed.x;
        float edge = pixel * 0.75;
        float night = k.night.x;
        float hour = k.night.w;
        float E = k.cam.w;

        // Front to back: boughs, wires, the town, hills, mountains, then the clouds.
        float4 acc = 0;
        float3 glow = 0;
        {
            float aspect = u.res.x / u.res.y;
            float2 sp = float2(uv.x * aspect, uv.y);
            if (hash11(seed * 1.7 + 0.3) > 0.5) sp.x = aspect - sp.x;
            float windy = clamp(k.wind.w / 40, 0.05, 1.0);
            float2 bl = boughs(sp, aspect, seed * 2.3, u.time, 0.03 + 0.1 * windy, 0.0018);
            float cov = max(bl.x, bl.y);
            if (cov > 0.002) {
                float3 l = k.lightDir.xyz;
                float3 up = float3(0, 1, 0), facing = -forward(k);
                float through = pow(saturate(dot(rd, l)), 12.0) * k.lightDir.w;
                float3 leafCol = lit(float3(0.016, 0.024, 0.01), normalize(facing + up * 0.5), k, amb) + amb[3].rgb * float3(0.3, 0.45, 0.08) * through * 0.1;
                float3 barkCol = lit(float3(0.012, 0.01, 0.009), normalize(facing + up * 0.3), k, amb);
                under(acc, cov, mix(barkCol, leafCol, bl.y / max(bl.x + bl.y, 1e-3)), 0.05, uv, haze, E);
            }
        }
        if (alt < rad(24)) {
            float3 leafAlb = float3(0.04, 0.055, 0.028);
            float3 up = float3(0, 1, 0), facing = -forward(k);

            // Telephone wires sagging between poles along the street, about 25 m away.
            if (k.seed.y > 0.5) {
                float a0 = hash11(seed * 1.3) * TAU;
                float rel = atan2(sin(atan2(rd.x, rd.z) - a0), cos(atan2(rd.x, rd.z) - a0));
                float spanW = 0.9;
                float x = fract(rel / spanW + 0.5) - 0.5;
                for (int w = 0; w < 3; w++) {
                    float h0 = rad(3.2 + w * 0.55 + 0.6 * hash11(seed + w));
                    float sag = rad(2.4 + 0.5 * w) * (1 - 4 * x * x);
                    float y = h0 - sag;
                    float width = max(pixel * 0.55, rad(0.025));
                    float cov = smoothstep(width + pixel * 0.6, width - pixel * 0.2, abs(alt - y)) * min(width / pixel, 1.0);
                    under(acc, cov * 0.85, lit(float3(0.03), up, k, amb), 0.025, uv, haze, E);
                }
                // The pole itself.
                float pole = smoothstep(pixel * 1.5, 0, abs(x * spanW) - rad(0.35)) * step(alt, rad(6.5));
                under(acc, pole, lit(float3(0.08, 0.07, 0.06), facing, k, amb), 0.03, uv, haze, E);
            }

            float3 info;
            // The town at 900 m and 2.8 km.
            if (alt < rad(4.5)) {
                float r = town(turn, alt + rad(0.3), 420, 0.6, seed + 7, info);
                float cov = smoothstep(-edge, edge, r);
                float3 alb = info.y == 2 ? leafAlb * 1.3 : info.y == 1 ? mix(float3(0.2, 0.1, 0.07), float3(0.12, 0.12, 0.13), hash11(info.x + seed))
                                                              : mix(float3(0.45, 0.42, 0.36), float3(0.32, 0.2, 0.15), hash11(info.x * 1.3 + seed));
                float3 col = lit(alb, info.y == 1 ? normalize(up + facing) : info.y == 2 ? normalize(up + facing) : facing, k, amb);
                if (info.y == 0 && night > 0) glow += windowsLit(turn, alt + rad(0.3), 420, 0.6, info.x, hour, k.extra.x) * 0.006 * cov * (1 - acc.a);
                // Street lamps along the foot of the buildings.
                float lamp = smoothstep(rad(0.1), 0.0, length(float2(fract(turn * 700 + 0.3) - 0.5, (alt + rad(0.45)) / rad(0.05)) * float2(1, 0.4)));
                glow += float3(1.0, 0.6, 0.25) * lamp * night * k.extra.x * 0.02 * hash11(floor(turn * 700) + seed) * (1 - acc.a);
                under(acc, cov, col, 0.9, uv, haze, E);

                r = town(turn, alt + rad(0.1), 1300, 0.28, seed + 3, info);
                cov = smoothstep(-edge, edge, r);
                alb = info.y == 2 ? leafAlb * 1.4 : mix(float3(0.35, 0.33, 0.3), float3(0.2, 0.12, 0.1), hash11(info.x + seed));
                if (night > 0) glow += windowsLit(turn, alt + rad(0.1), 1300, 0.28, info.x, hour, k.extra.x) * 0.004 * cov * step(info.y, 0.5) * (1 - acc.a);
                under(acc, cov, lit(alb, normalize(up + facing), k, amb), 2.8, uv, haze, E);
            }
            // Wooded hills at 6 km, mountains at 25 km.
            if (alt < rad(3.5)) {
                float cov = smoothstep(-edge, edge, hills(turn, seed + 3) - alt);
                under(acc, cov, lit(leafAlb * 1.2, normalize(up + facing), k, amb), 6, uv, haze, E);
                float amount = 0.25 + 0.75 * hash11(seed * 2.7);
                cov = smoothstep(-edge, edge, mountains(turn, seed + 9, amount) - alt);
                float gullies = 0.85 + 0.3 * fbm(float2(turn * 900, alt * 300), 3);
                under(acc, cov, lit(float3(0.07, 0.075, 0.06) * gullies, normalize(up * 0.7 + facing), k, amb), 25, uv, haze, E);
            }
        }
        // The clouds and the sky behind everything else are drawn already; clouds are blended in here.
        float4 cl = clouds.sample(s, uv);
        float coverC = 1 - cl.a;
        acc.rgb += (1 - acc.a) * cl.rgb;
        acc.a += (1 - acc.a) * coverC;

        acc.rgb += glow;

        // Rain or snow in front of everything, lit by the sky.
        float3 fall = (amb[0].rgb * 0.9 + amb[1].rgb * 0.3) * E * 1.2;
        float slant = clamp(k.wind.w / 60, 0.0, 0.5) * (sin(k.wind.z - k.cam.x) > 0 ? 1 : -1);
        if (k.weather.x > 0) {
            float r = min(rainStreaks(p, u.time, slant, k.weather.x) * (0.4 + 0.5 * k.weather.x), 0.6);
            acc = float4(fall * r, r) + acc * (1 - r);
        }
        if (k.weather.y > 0) {
            float sn = min(snowFlakes(p, u.time, slant * 2, k.weather.y), 1.0) * 0.85;
            acc = float4(fall * 1.5 * sn, sn) + acc * (1 - sn);
        }
        return acc;
    }
    """
}
