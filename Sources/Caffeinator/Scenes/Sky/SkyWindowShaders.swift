extension SkyWindow {
    /// Metal for the window: the atmosphere, the night sky, clouds, the skyline, falling rain and snow, and the glass.
    static let source = """
    struct Sky {
        float4 cam;        // azimuth, pitch, focal length, exposure
        float4 sun;        // direction (x east, y up, z north), intensity
        float4 moon;       // direction, intensity
        float4 moonInfo;   // lit fraction, drawn radius, waxing, latitude
        float4 sunGround;  // sunlight left at the ground
        float4 lightLow, lightMid, lightHigh;  // the light reaching each cloud layer
        float4 lightDir;   // its direction; w is 1 for the Sun, 0 for the Moon
        float4 cover;      // low, mid, high cloud, extra darkness for rain clouds
        float4 wind;       // drift x, z (km), heading, speed (km/h)
        float4 weather;    // rain, snow, fog, lightning
        float4 night;      // star visibility, local sidereal time, haze, local hour
        float4 seed;       // skyline seed
    };

    inline float rad(float d) { return d * PI / 180; }

    constant float RE = 6360.0, RA = 6420.0;
    constant float3 BR = float3(5.802e-3, 13.558e-3, 33.1e-3);
    constant float3 BO = float3(1.3e-3, 1.881e-3, 0.085e-3);
    constant float BM = 3.996e-3;

    // A level camera with its lens shifted up, like an architect's: verticals stay vertical and the horizon stays flat.
    float3 forward(constant Sky &k) { return float3(sin(k.cam.x), 0, cos(k.cam.x)); }
    float lensShift(constant Sky &k) { return tan(k.cam.y) * k.cam.z; }

    float3 camRay(float2 p, constant Sky &k) {
        float3 f = forward(k), r = float3(f.z, 0, -f.x);
        return normalize(p.x * r + (p.y + lensShift(k)) * float3(0, 1, 0) + k.cam.z * f);
    }

    float2 hitSphere(float3 o, float3 d, float r) {
        float b = dot(o, d), c = dot(o, o) - r * r, h = b * b - c;
        if (h < 0) return float2(-1);
        h = sqrt(h);
        return float2(-b - h, -b + h);
    }

    float3 extinction(float h, float haze) {
        return BR * exp(-h / 8) + BM * 1.11 * haze * exp(-h / 1.2) + BO * max(0.0, 1 - abs(h - 25) / 15);
    }

    // Optical depth from p toward the light, or a large number when the planet is in the way.
    float3 depthToLight(float3 p, float3 l, float haze) {
        if (hitSphere(p, l, RE).x > 0) return float3(1e4);
        float t = hitSphere(p, l, RA).y;
        float dt = t / 8;
        float3 od = 0;
        for (int i = 0; i < 8; i++) od += extinction(length(p + l * (i + 0.5) * dt) - RE, haze) * dt;
        return od;
    }

    // Single scattering of one light along a view ray (Rayleigh, Mie and ozone absorption), with steps that grow
    // with distance so the thick lower air gets most of them.
    float3 scatter(float3 rd, float3 l, float power, float haze) {
        if (power <= 0) return 0;
        float3 o = float3(0, RE + 0.05, 0);
        float tMax = hitSphere(o, rd, RA).y;
        float2 g = hitSphere(o, rd, RE);
        if (g.x > 0) tMax = g.x;
        float mu = dot(rd, l);
        float pr = 3 / (16 * PI) * (1 + mu * mu);
        float gm = 0.76;
        float pm = 3 / (8 * PI) * ((1 - gm * gm) * (1 + mu * mu)) / ((2 + gm * gm) * pow(1 + gm * gm - 2 * gm * mu, 1.5));
        float3 sum = 0, od = 0;
        float prev = 0;
        for (int i = 1; i <= 16; i++) {
            float t = tMax * pow(i / 16.0, 2.0);
            float dt = t - prev, tm = (t + prev) * 0.5;
            prev = t;
            float3 q = o + rd * tm;
            float h = length(q) - RE;
            od += extinction(h, haze) * dt;
            float3 tr = exp(-(od + depthToLight(q, l, haze)));
            sum += tr * (BR * exp(-h / 8) * pr + BM * haze * exp(-h / 1.2) * pm) * dt;
        }
        return sum * power;
    }

    // Airglow and a little light pollution: a moonless night is deep blue, never black.
    float3 nightFloor(float alt, float haze) {
        return float3(2.0e-6, 3.4e-6, 8.0e-6) + float3(1.0e-5, 7.0e-6, 4.6e-6) * exp(-max(alt, 0.0) * 6) * (0.6 + 0.4 * haze);
    }

    // The sky's light for each pixel at low resolution: it varies slowly, so this is cheap and smooth.
    fragment float4 skyLUT(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]]) {
        float3 rd = camRay(centered(in.uv, u.res), k);
        float3 c = scatter(rd, k.sun.xyz, k.sun.w, k.night.z) + scatter(rd, k.moon.xyz, k.moon.w, k.night.z);
        // A rough stand-in for multiple scattering, which keeps twilight from going grey.
        c += float3(0.7, 0.95, 1.45) * length(c) * 0.07 * saturate(rd.y * 2 + 0.2);
        return float4(c, 1);
    }

    // The log-average brightness of the sky above the horizon, for exposure (read back a frame later).
    kernel void skyMeter(constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]], device float *out [[buffer(2)]],
                         texture2d<float> lut [[texture(0)]], uint id [[thread_position_in_grid]]) {
        if (id > 0) return;
        float horizon = 0.5 + tan(k.cam.y) * k.cam.z * 0.5;
        float sum = 0, n = 0, peak = 0;
        for (int j = 0; j < 12; j++) for (int i = 0; i < 20; i++) {
            float2 uv = float2((i + 0.5) / 20, (j + 0.5) / 12 * (horizon - 0.03));
            float3 c = lut.read(uint2(uv * float2(lut.get_width(), lut.get_height()))).rgb;
            float3 rd = camRay(centered(uv, u.res), k);
            c += nightFloor(asin(clamp(rd.y, -1.0, 1.0)), k.night.z);
            float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
            sum += log(lum + 1e-9);
            peak = max(peak, lum);
            n += 1;
        }
        // Lean toward the brightest part when the light is uneven, as in twilight, so the glow on the horizon keeps its colour.
        float w = mix(0.3, 0.08, smoothstep(-0.05, 0.1, k.sun.y));
        out[0] = exp(sum / n * (1 - w) + log(peak + 1e-9) * w);
    }

    // Direction to right ascension and declination (radians).
    float2 equatorial(float3 v, float lst, float lat) {
        float3 pole = float3(0, sin(lat), cos(lat));
        float3 meridian = float3(0, cos(lat), -sin(lat));
        float dec = asin(clamp(dot(v, pole), -1.0, 1.0));
        float ha = atan2(dot(v, float3(-1, 0, 0)), dot(v, meridian));
        return float2(lst - ha, dec);
    }

    // Faint stars below the catalogue's limit and the Milky Way, fixed to the turning celestial sphere.
    float3 deepSky(float3 rd, constant Sky &k, float pixel) {
        float2 eq = equatorial(rd, k.night.y, k.moonInfo.w);
        float3 e = float3(cos(eq.y) * cos(eq.x), cos(eq.y) * sin(eq.x), sin(eq.y));
        // J2000 equatorial to galactic coordinates.
        float3 g = float3(dot(e, float3(-0.0548755604, -0.8734370902, -0.4838350155)),
                          dot(e, float3(0.4941094279, -0.4448296300, 0.7469822445)),
                          dot(e, float3(-0.8676661490, -0.1980763734, 0.4559837762)));
        float b = asin(clamp(g.z, -1.0, 1.0)), l = atan2(g.y, g.x);
        float core = exp(-l * l * 1.2);
        float band = exp(-b * b / (0.012 + 0.025 * core)) * (0.3 + 0.7 * core);
        float clumps = fbm(float2(l * 9, b * 14) + 3.7, 5);
        float dust = smoothstep(0.45, 0.7, fbm(float2(l * 7, b * 22) + 11.0, 4)) * exp(-b * b / 0.004);
        float3 milky = float3(0.85, 0.85, 1.0) * band * (0.3 + clumps) * (1 - 0.7 * dust) * 0.03;
        float3 a = abs(e);
        float2 f = a.x > a.y && a.x > a.z ? e.yz / e.x : a.y > a.z ? e.xz / e.y : e.xy / e.z;
        float face = a.x > a.y && a.x > a.z ? sign(e.x) : a.y > a.z ? 2 + sign(e.y) : 4 + sign(e.z);
        float cells = 110;
        float2 cell = floor(f * cells);
        float3 s = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float2 id = cell + float2(i, j);
            float h = hash12(id + face * 131.7);
            if (h > 0.12) continue;
            float2 at = (id + hash22(id + face * 17.1)) / cells;
            float d = length(f - at) / pixel;
            float m = pow(hash12(id + 5.3 + face), 6.0);
            float3 tint = mix(float3(1.0, 0.85, 0.7), float3(0.75, 0.85, 1.0), hash12(id + 9.1));
            s += tint * (0.035 + m * 0.4) * exp(-d * d * 1.6);
        }
        return (s + milky) * k.night.x;
    }

    // The Moon: a sphere lit from the Sun's direction, so its phase and tilt are true, with darker maria.
    float3 moonDisc(float3 rd, constant Sky &k, float pixel, thread float &cover) {
        float3 m = k.moon.xyz;
        float radius = k.moonInfo.y;
        cover = 0;
        if (dot(rd, m) < cos(radius * 6) || m.y < -0.05) return 0;
        float3 pole = float3(0, sin(k.moonInfo.w), cos(k.moonInfo.w));
        float3 up = normalize(pole - m * dot(pole, m));
        float3 right = cross(up, m);
        float2 q = float2(dot(rd, right), dot(rd, up)) / radius;
        float r = length(q);
        float edge = 1 - smoothstep(1 - pixel / radius, 1 + pixel / radius, r);
        float halo = exp(-max(r - 1, 0.0) * 1.6) * 0.03 * k.moonInfo.x * step(1.0, r);
        if (edge <= 0) return float3(0.85, 0.9, 1.0) * halo;
        float3 n = normalize(right * q.x + up * q.y - m * sqrt(max(1 - r * r, 0.0)));
        float lambert = dot(n, k.sun.xyz);
        float lit = smoothstep(-0.03, 0.08, lambert) * (0.4 + 0.6 * max(lambert, 0.0));
        float maria = 0;
        const float3 seas[9] = { float3(-0.32, 0.45, 0.26), float3(0.18, 0.4, 0.15), float3(0.36, 0.13, 0.18), float3(0.72, 0.3, 0.1),
                                  float3(0.6, -0.12, 0.13), float3(0.33, -0.28, 0.08), float3(-0.62, 0.05, 0.33), float3(-0.22, -0.35, 0.15),
                                  float3(0.0, 0.72, 0.12) };
        for (int i = 0; i < 9; i++) maria += smoothstep(1.0, 0.35, length(q - seas[i].xy) / seas[i].z);
        float albedo = (0.78 - 0.3 * saturate(maria)) * (0.88 + 0.22 * fbm(q * 8 + 2, 4));
        albedo += 0.2 * exp(-length(q - float2(-0.1, -0.7)) * 22);
        cover = edge;
        return float3(1.0, 0.97, 0.92) * albedo * (lit + 0.015 * (1 - k.moonInfo.x)) * edge;
    }

    fragment float4 skyBase(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                            texture2d<float> lut [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float3 rd = camRay(p, k);
        float pixel = 2 / (u.res.y * k.cam.z);
        float E = k.cam.w;
        float alt = asin(clamp(rd.y, -1.0, 1.0));
        float3 c = (lut.sample(s, in.uv).rgb + nightFloor(alt, k.night.z)) * E;
        if (k.night.x > 0) c += deepSky(rd, k, pixel) * saturate(rd.y * 8);
        // The Sun's disc, limb-darkened.
        float sr = 0.0047 * 1.3;
        float sd = acos(clamp(dot(rd, k.sun.xyz), -1.0, 1.0)) / sr;
        if (sd < 1.3) {
            float mu = sqrt(max(1 - sd * sd, 0.0));
            c += k.sunGround.rgb * 30 * (1 - 0.6 * (1 - mu)) * (1 - smoothstep(1 - pixel / sr, 1 + pixel / sr, sd)) * E;
        }
        // The Moon: brilliant at night, a pale ghost by day.
        float cover;
        float3 moon = moonDisc(rd, k, pixel, cover);
        c = c * (1 - cover * k.night.x) + moon * mix(0.12, 1.8, k.night.x);
        return float4(c, 1);
    }

    // Catalogue stars and the planets as point sprites, fading in as the sky darkens and twinkling gently near the horizon.
    vertex P skyStar(uint id [[vertex_id]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                     const device float4 *stars [[buffer(2)]]) {
        float4 a = stars[id * 2], b = stars[id * 2 + 1];
        float3 d = a.xyz;
        P o;
        float3 f = forward(k);
        float z = dot(d, f);
        o.pos = float4(dot(d, float3(f.z, 0, -f.x)) * k.cam.z / (u.res.x / u.res.y), d.y * k.cam.z - lensShift(k) * z, z * 0.001, z);
        bool planet = b.w > 0.5;
        float3 color = planet ? b.xyz : mix(float3(1), blackbody(4600 * (1 / (0.92 * b.x + 1.7) + 1 / (0.92 * b.x + 0.62))), 0.6);
        float flux = pow(10.0, -0.4 * (a.w - 1.0));
        float low = 1 - saturate(d.y * 3);
        float twinkle = planet ? 1 : 1 + 0.15 * low * sin(u.time * (1.1 + fract(id * 0.618) * 1.5) + id);
        float vis = k.night.x * saturate(d.y * 25 - 0.2) * (1 - 0.6 * k.weather.z) * (1 - 0.45 * low);
        float scale = u.res.y / 900;
        o.size = (2.2 + 3.2 * sqrt(min(flux, 12.0))) * scale;
        o.color = float4(color * min(flux * 0.9, 3.0) * twinkle * vis, 0);
        if (o.pos.w <= 0 || vis <= 0) o.pos = float4(0, 0, -1, 1);
        return o;
    }

    // MARK: Skyline

    float n1(float x, float period) {
        float i = floor(x), f = fract(x);
        float w = f * f * (3 - 2 * f);
        return mix(hash11(fmod(i, period) + 0.5), hash11(fmod(i + 1, period) + 0.5), w);
    }

    float f1(float t, float base, int octaves, float seed) {
        float v = 0, a = 0.5, f = base;
        for (int i = 0; i < octaves; i++) { v += a * n1(t * f + seed * 7.1, f); f *= 2; a *= 0.5; }
        return v;
    }

    // Distant hills, an elevation angle in radians for an azimuth t in turns.
    float hills(float t, float seed) { float v = f1(t, 7, 5, seed); return rad(0.15 + 1.9 * v * v); }

    // A tree line, ragged and leafy, with gaps where fields open out.
    float treeline(float t, float seed) {
        float open = smoothstep(0.35, 0.55, f1(t, 11, 2, seed + 4));
        float canopy = rad(0.55 + 1.1 * f1(t, 40, 3, seed)) + rad(0.25) * (n1(t * 700 + seed, 700) - 0.5) + rad(0.12) * n1(t * 2400, 2400);
        return mix(rad(0.1), canopy, open);
    }

    // The near skyline as a signed distance in radians (positive inside): garden walls and hedges, houses with gables, hips or flat roofs, chimneys, the odd aerial,
    // a taller block now and then, round ragged trees, and windows that light up in the evening.
    float rooftops(float t, float alt, float seed, float hour, thread float &window) {
        window = 0;
        float best = rad(0.3 + 0.25 * n1(t * 260 + seed, 260) + 0.1 * n1(t * 1300, 1300)) - alt;
        float cells = 48, span = 360 / cells;
        float x = t * cells;
        float c = floor(x);
        for (int k = -1; k <= 1; k++) {
            float id = fmod(c + k + cells, cells);
            float lx = x - (c + k);
            float h = hash11(id * 3.17 + seed);
            float kind = hash11(id * 5.11 + seed + 1);
            if (h < 0.3) continue;
            if (h < 0.45) {
                // A tree: a ragged round crown on a short trunk.
                float cx = 0.5 + (hash11(id + 2.0 + seed) - 0.5) * 0.4;
                float R = rad(1.0 + 0.9 * hash11(id + 5.0 + seed));
                float top = R * 2 + rad(0.8 + 1.2 * hash11(id + 6.0 + seed));
                float2 q = float2((lx - cx) * rad(span), alt - (top - R)) / R;
                float ragged = 1 + 0.3 * (fbm(q * 2.2 + id * 3.1, 4) - 0.5);
                best = max(best, (ragged - length(q * float2(0.95, 1.05))) * R);
                best = max(best, min(rad(0.06) - abs(lx - cx) * rad(span), top - R * 1.5 - alt));
                continue;
            }
            float w0 = 0.05 + 0.25 * hash11(id + 1.0 + seed), w1 = 0.95 - 0.25 * hash11(id + 4.0 + seed);
            if (lx < w0 || lx > w1) continue;
            bool tall = kind > 0.88;
            float wall = rad(0.7 + 1.5 * h * h + (tall ? 1.8 : 0));
            float mid = (w0 + w1) * 0.5, hw = (w1 - w0) * 0.5;
            float across = 1 - abs(lx - mid) / hw;
            float roof = wall;
            if (tall) roof += across > 0.04 ? rad(0.1) : 0;
            else if (kind < 0.55) roof += rad(0.5 + 0.7 * hash11(id + 8.0 + seed)) * across;
            else if (kind < 0.72) roof += rad(0.45) * min(1.0, across * 3);
            else roof += across < 0.03 ? rad(0.12) : 0;
            float chim = mid + hw * (hash11(id + 9.0 + seed) > 0.5 ? 0.5 : -0.45);
            if (!tall && hash11(id + 10.0 + seed) > 0.4 && abs(lx - chim) * span < 0.25) roof = max(roof, wall + rad(0.75));
            if (hash11(id + 11.0 + seed) > 0.75 && abs(lx - mid - 0.08) * span < 0.04) roof = max(roof, wall + rad(1.6));
            best = max(best, min(roof - alt, min(lx - w0, w1 - lx) * rad(span)));
            if (alt < wall - rad(0.15)) {
                float floors = tall ? 6 : 2;
                float2 g = float2((lx - w0) / (w1 - w0) * (tall ? 7 : 4), (alt + rad(2.5)) / (wall + rad(2.35)) * floors);
                float2 cell = floor(g), fr = fract(g);
                if (cell.y >= 0 && cell.y < floors && fr.x > 0.32 && fr.x < 0.68 && fr.y > 0.3 && fr.y < 0.7) {
                    float on = hash11(id * 7 + cell.x * 3 + cell.y * 11 + seed);
                    float evening = smoothstep(17, 19, hour) * (1 - smoothstep(22.5, 24.5, hour)) + (hour < 1 ? 0.5 : 0)
                                  + (hour > 5.5 && hour < 7.5 ? 0.35 : 0);
                    window = on < evening * 0.45 ? 0.5 + 0.5 * hash11(on * 91) : 0;
                }
            }
        }
        return best;
    }

    // MARK: Clouds

    float cloudDensity(float2 q, int kind, float octaves, float time) {
        if (kind == 0) {
            // Cumulus: clusters a few kilometres apart, eroded at the edges into cauliflower puffs.
            float2 w = float2(noise(q * 1.3 + float2(time * 0.003, 0)), noise(q * 1.3 + float2(4.1, 9.2))) - 0.5;
            float2 r = q + w * 0.15;
            float base = fbm(r * 3.0, 3);
            float detail = fbm(r * 16.0 + 2.3, int(octaves));
            return base - (1 - detail) * 0.3 + 0.15;
        }
        if (kind == 1) {
            // Altocumulus: a dappled sheet of small puffs.
            float sheet = fbm(q * 1.6 + 1.7, 3);
            float puffs = fbm(q * 14 + 3.3, int(max(octaves - 1, 2.0)));
            return sheet - (1 - puffs) * 0.35 + 0.17;
        }
        // Cirrus: long fibres (q is already stretched along the wind), gently bent.
        q.x += (noise(q * float2(0.4, 3.0) + 7.7) - 0.5) * 1.2;
        return fbm(q, int(octaves));
    }

    // One cloud layer seen from below: premultiplied colour (before exposure) and opacity.
    float4 cloudLayer(float3 rd, float h, float cover, float2 drift, float heading, int kind, float scale, float4 light, float3 ldir,
                      float3 ambient, float3 skyHere, float pixel, float time, float darken) {
        if (cover < 0.01 || rd.y <= 0.0) return 0;
        float3 o = float3(0, RE + 0.05, 0);
        float t = hitSphere(o, rd, RE + h).y;
        float3 hit = o + rd * t;
        float2 q = (hit.xz + drift) * scale;
        if (kind == 2) q = float2(dot(q, float2(cos(heading), -sin(heading))) * 2.4, dot(q, float2(sin(heading), cos(heading))) * 0.3);
        float footprint = t * pixel * scale;
        float octaves = clamp(6.5 - log2(max(footprint * 40, 1.0)), 2.0, 6.0);
        float d = cloudDensity(q, kind, octaves, time);
        float thresh = kind == 2 ? mix(0.7, 0.35, cover) : mix(0.66, 0.16, cover);
        float soft = kind == 2 ? 0.3 : 0.07;
        float dens = smoothstep(thresh, thresh + soft, d);
        // How deep into the cloud this is: thin edges glow, thick cores go grey.
        float core = saturate((d - thresh) / 0.3);
        // Nearly full cover closes into a deck.
        dens = max(dens, smoothstep(0.85, 1.0, cover) * 0.85);
        core = max(core, smoothstep(0.85, 1.0, cover) * 0.6);
        // Far off, detail averages out to the cover itself, which keeps the horizon calm.
        dens = mix(dens, cover * 0.85, smoothstep(30, 120, t));
        float thick = (kind == 0 ? 4.0 : kind == 1 ? 1.8 : 0.45) * (1 + 1.5 * darken);
        float od = dens * thick;
        float alpha = 1 - exp(-od);
        if (alpha < 0.002) return 0;
        // Light from the Sun's side, sampled twice through the cloud.
        float2 sd = normalize(ldir.xz + 1e-5);
        float s1 = smoothstep(thresh, thresh + soft, cloudDensity(q + sd * 0.04, kind, min(octaves, 4.0), time));
        float s2 = smoothstep(thresh, thresh + soft, cloudDensity(q + sd * 0.14, kind, 3.0, time));
        float through = exp(-(s1 * 0.6 + s2 * 0.4) * thick * 0.8);
        float mu = dot(rd, ldir);
        float g = 0.75;
        float hg = (1 - g * g) / pow(1 + g * g - 2 * g * mu, 1.5) / (4 * PI);
        // Bases catch direct light only when the Sun is low; then sunset paints them.
        float under = mix(0.55, 1.5, smoothstep(0.3, -0.04, ldir.y));
        float side = 0.6 + 0.4 * (1 - saturate(rd.y * 3));
        // The Sun as seen from this cloud: higher toward it, lower away from it; below the cloud's own horizon it's in Earth's shadow.
        float2 flat = hit.xz;
        float reach = length(flat);
        float toward = reach > 1e-3 ? dot(flat / reach, sd) : 0;
        float localAlt = asin(clamp(ldir.y, -1.0, 1.0)) + reach / RE * toward;
        float dip = sqrt(2 * h / RE);
        float3 sun = light.rgb * 20 * smoothstep(-dip - 0.005, -dip + 0.006, localAlt);
        float3 c = ambient * mix(1.05, 0.5, saturate(core * (1 + darken)))
                 + sun * ((0.1 * through * under * side * (1 - 0.6 * core) + 0.02) * (1 - 0.7 * darken)
                          + hg * pow(1 - core, 2.0) * 0.6 * (1 - darken));
        // Haze between us and distant clouds.
        float far = 1 - exp(-t / 60);
        c = mix(c, skyHere / max(alpha, 0.05), far * 0.8);
        return float4(c * alpha, alpha);
    }

    // MARK: Falling weather

    float rainStreaks(float2 p, float time, float slant, float amount) {
        float a = 0;
        for (int i = 0; i < 3; i++) {
            float s = 16 + i * 14;
            float2 q = float2(p.x + p.y * slant, p.y) * float2(s, s * 0.11);
            q.y += time * (2.6 - i * 0.6) * s * 0.11;
            float col = floor(q.x);
            float r = hash11(col * 1.13 + i * 17);
            if (r > amount * 0.85 + 0.1) continue;
            float x = fract(q.x) - (0.3 + 0.4 * hash11(col + 3.3));
            float y = fract(q.y + r * 13);
            a += smoothstep(0.05, 0.0, abs(x)) * smoothstep(0, 0.08, y) * smoothstep(0.3, 0.12, y) * (0.5 - i * 0.12);
        }
        return a;
    }

    float snowFlakes(float2 p, float time, float drift, float amount) {
        float a = 0;
        for (int i = 0; i < 4; i++) {
            float s = 5 + i * 4;
            float2 q = p * s;
            q.y += time * (0.42 - i * 0.07) * s * 0.4;
            q.x += time * drift * s * 0.1 + sin(q.y * 0.7 + i) * 0.3;
            float2 cell = floor(q);
            float h = hash12(cell + i * 31);
            if (h > amount * 0.75 + 0.1) continue;
            float2 at = cell + 0.2 + 0.6 * hash22(cell + i * 7);
            at.x += 0.15 * sin(time * (0.5 + h) + h * 30);
            float r = 0.05 + 0.04 * hash12(cell + 2.1);
            a += smoothstep(r, r * 0.3, length(q - at)) * (0.85 - i * 0.15);
        }
        return a;
    }

    fragment float4 skyFront(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                             texture2d<float> lut [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float3 rd = camRay(p, k);
        float E = k.cam.w;
        float pixel = 2 / (u.res.y * k.cam.z);
        float alt = asin(clamp(rd.y, -1.0, 1.0));
        float turn = fract(atan2(rd.x, rd.z) / TAU + 1);
        float seed = k.seed.x;
        // The sky's colour where the horizon crosses this column, and high up.
        float hv = 0.5 + tan(k.cam.y) * k.cam.z * 0.5;
        float3 horizon = lut.sample(s, float2(in.uv.x, hv - 0.01)).rgb + nightFloor(0.02, k.night.z);
        float3 around = 0;
        for (int i = 0; i < 6; i++) around += lut.sample(s, float2((i + 0.5) / 6, hv - 0.02)).rgb;
        around = around / 6 + nightFloor(0.02, k.night.z);
        float3 upper = (lut.sample(s, float2(0.2, 0.03)).rgb + lut.sample(s, float2(0.8, 0.03)).rgb) * 0.5 + nightFloor(0.6, k.night.z);
        // Under a full deck of cloud, the low sky takes the cloud's grey instead of the clear sky's glow.
        float deck = saturate(k.cover.x * 1.2 - 0.15);
        float3 grey = (upper * 0.9 + around * 0.3) * (0.7 - 0.25 * k.cover.w);
        horizon = mix(horizon, grey * 1.1, deck);
        around = mix(around, grey, deck);
        float3 here = lut.sample(s, in.uv).rgb + nightFloor(alt, k.night.z);
        float night = k.night.x;
        float fog = k.weather.z;
        float3 cityGlow = float3(1.0e-5, 7e-6, 4.6e-6) * night;
        float3 ambient = around * 0.4 + upper * 0.85;

        // Everything is light before exposure, except the glow of lit windows.
        float window = 0;
        bool low = alt < rad(8);
        float edge = pixel * 0.8;
        float near = low ? smoothstep(-edge, edge, rooftops(turn, alt, seed, k.night.w, window)) : 0;
        float trees = low ? smoothstep(-edge, edge, treeline(turn, seed + 3.0) - alt) : 0;
        float hill = low ? smoothstep(-edge, edge, hills(turn, seed + 9.0) - alt) : 0;
        // The landscape is lit by the sky above it, and a little by the Sun or Moon.
        float3 land = around * 0.05 + upper * 0.04 + k.lightLow.rgb * 20 * 0.008 * saturate(k.lightDir.y + 0.1);
        float3 mistColor = mix(float3(dot(horizon, float3(0.3, 0.5, 0.2))), horizon, 0.45) * 1.1;
        float4 acc = 0;
        if ((1 - near) * (1 - trees) * (1 - hill) > 0.001) {
            float3 ldir = k.lightDir.xyz;
            float darken = k.cover.w;
            float3 amb = ambient * mix(1.0, 0.75, darken) + cityGlow * 1.8;
            float heading = k.wind.z;
            // High ice cloud sees mostly the upper sky.
            float3 ambHigh = upper * 0.9 + around * 0.1 + cityGlow * 0.15;
            float4 hi = cloudLayer(rd, 9.0, k.cover.z, k.wind.xy * 2.2, heading, 2, 0.08, k.lightHigh, ldir, ambHigh, here, pixel, u.time, 0);
            float4 mi = cloudLayer(rd, 4.0, k.cover.y, k.wind.xy * 1.6, heading, 1, 0.14, k.lightMid, ldir, amb, here, pixel, u.time, darken * 0.5);
            float4 lo = cloudLayer(rd, 1.6, k.cover.x, k.wind.xy, heading, 0, 0.2, k.lightLow, ldir, amb, here, pixel, u.time, darken);
            acc = hi;
            acc = mi + acc * (1 - mi.a);
            acc = lo + acc * (1 - lo.a);
            // Lightning glows softly inside the clouds.
            float flash = k.weather.w * k.weather.w;
            if (flash > 0) {
                float spot = exp(-length(p - float2(0.5 * sin(floor(u.time / 20) * 3.1), 0.35)) * 1.6);
                acc.rgb += float3(0.8, 0.85, 1.0) * flash * spot * acc.a * 0.35 / E;
            }
            float mist = fog * exp(-max(alt, 0.0) * 9) * 0.85;
            acc = float4(mistColor * mist, mist) + acc * (1 - mist);
        }
        acc = mix(acc, float4(mix(land * 1.1, mistColor * 0.95, 0.45 + 0.5 * fog), 1), hill);
        acc = mix(acc, float4(mix(land * 0.75, mistColor * 0.85, 0.15 + 0.55 * fog), 1), trees);
        // The foreground darkens toward the sill, with a little haze where it meets the horizon.
        float depth = 0.55 + 0.45 * smoothstep(-rad(9), rad(0.5), alt);
        acc = mix(acc, float4(mix(land * 0.5 * depth, mistColor * 0.8, fog * 0.4), 1), near);
        float3 lamps = float3(1.0, 0.64, 0.33) * window * 0.55 * night * near;
        acc.rgb = acc.rgb * E + lamps;

        // Rain or snow in front of everything, lit by the sky.
        float3 fall = (around * 0.5 + upper * 0.5) * E * 1.4 + 0.02;
        float slant = clamp(k.wind.w / 60, 0.0, 0.5) * (sin(k.wind.z - k.cam.x) > 0 ? 1 : -1);
        if (k.weather.x > 0) {
            float r = min(rainStreaks(p, u.time, slant, k.weather.x) * (0.35 + 0.45 * k.weather.x), 0.8);
            acc = float4(fall * r, r) + acc * (1 - r);
        }
        if (k.weather.y > 0) {
            float sn = min(snowFlakes(p, u.time, slant * 2, k.weather.y), 1.0) * 0.8;
            acc = float4(fall * 1.6 * sn, sn) + acc * (1 - sn);
        }
        return acc;
    }

    // MARK: Glass and frame

    // Rain on the glass: beads that come and go, and a few drops sliding down; returns a refraction offset.
    float3 drops(float2 p, float time, float amount) {
        float3 n = 0;
        float2 g = p * 15;
        float2 cell = floor(g);
        float2 f = fract(g) - 0.5;
        float h = hash12(cell);
        if (h < amount) {
            float2 c = (hash22(cell) - 0.5) * 0.6;
            float life = fract(time * (0.02 + 0.03 * hash12(cell + 2)) + h * 7);
            float r = (0.1 + 0.16 * hash12(cell + 5)) * smoothstep(0.0, 0.1, life) * smoothstep(1.0, 0.8, life);
            float2 d = f - c;
            if (length(d / float2(1, 1.15)) < r) n = float3(d / r, 1);
        }
        float2 sg = p * float2(5, 1.2);
        float col = floor(sg.x);
        float sh = hash11(col * 3.7);
        if (sh < amount * 0.6) {
            float y = fract(sg.y + time * (0.04 + 0.07 * sh) + sh * 10);
            float x = fract(sg.x) - 0.5 + 0.06 * sin(sg.y * 7 + sh * 20);
            float2 d = float2(x * 5, (y - 0.5) * 6);
            if (length(d) < 0.3) n = float3(d / 0.3, 1);
            if (y > 0.5 && abs(x) < 0.025) {
                float2 bead = float2(x * 5, fract(y * 24) - 0.5);
                if (length(bead) < 0.18 * smoothstep(1.0, 0.5, y)) n = float3(bead / 0.18 * 0.5, 0.6);
            }
        }
        return n;
    }

    fragment float4 skyFinish(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                              texture2d<float> hdr [[texture(0)]], texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = in.uv;
        float aspect = u.res.x / u.res.y;
        float2 p = float2(uv.x * aspect, uv.y);
        // A slim, dark window frame and a sill.
        float side = 0.026, sill = 0.055;
        float2 inner0 = float2(side, side), inner1 = float2(aspect - side, 1 - sill);
        float rad0 = 0.008;
        float2 cq = clamp(p, inner0 + rad0, inner1 - rad0);
        float frameDist = max(max(inner0.x - p.x, p.x - inner1.x), max(inner0.y - p.y, p.y - inner1.y));
        frameDist = max(frameDist, length(p - cq) - rad0);
        float px = 1 / u.res.y;
        float inFrame = smoothstep(-px, px, frameDist);

        float rain = k.weather.x;
        float3 n = rain > 0.05 ? drops(p, u.time, rain * 0.85) : float3(0);
        float2 off = n.xy * n.z * 0.01 * float2(1, -1);
        float3 c = hdr.sample(s, uv + off).rgb + glow.sample(s, uv + off).rgb * 0.16;
        c = mix(c, glow.sample(s, uv).rgb * 1.05, rain * 0.2);
        c += n.z * 0.015 * smoothstep(0.3, 0.8, n.y);
        // Inside the reveal the glass is a touch darker, like the deep edge of a real window.
        c *= 1 - 0.25 * exp(min(frameDist, 0.0) * 90);

        // The frame catches only a little of the window's light.
        float3 light = (glow.sample(s, float2(0.5, 0.3)).rgb + glow.sample(s, float2(0.25, 0.45)).rgb + glow.sample(s, float2(0.75, 0.45)).rgb) / 3;
        float lum = min(dot(light, float3(0.3, 0.5, 0.2)), 1.5);
        float3 wood = float3(0.06, 0.05, 0.045);
        float edge = exp(-max(frameDist, 0.0) * 300);
        float3 fc = wood * (0.15 + 0.5 * lum) + light * edge * 0.06;
        if (p.y > 1 - sill && p.y < 1 - sill + 0.012 && p.x > side * 0.5 && p.x < aspect - side * 0.5) fc += light * 0.05;
        c = mix(c, fc, inFrame);
        c = film(c);
        // Fog drains the colour out of everything.
        c = mix(float3(dot(c, float3(0.3, 0.55, 0.15))), c, 1 - 0.6 * k.weather.z);
        float2 v = uv - 0.5;
        c *= 1 - 0.2 * dot(v, v);
        return float4(c, 1);
    }
    """
}
