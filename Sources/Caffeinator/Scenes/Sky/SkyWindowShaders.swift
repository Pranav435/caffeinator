extension SkyWindow {
    /// Metal for the window, in four parts: the atmosphere and the night sky (here), the clouds, the land outside,
    /// and the glass and frame.
    static var source: String { atmosphere + clouds + land + glass }

    /// The atmosphere after Hillaire (2020): a transmittance table, a multiple-scattering table, then each frame the
    /// sky along every view ray at low resolution with the haze in front of things at sixteen distances, all for
    /// sunlight and moonlight. Rayleigh and Mie scattering, ozone absorption and the planet's shadow give the blue
    /// day, the reddened Sun, blue-hour twilight, the Earth's shadow and the Belt of Venus without any painting.
    static let atmosphere = """
    struct Sky {
        float4 cam;        // azimuth, pitch, focal length, exposure
        float4 sun;        // direction (x east, y up, z north), illuminance
        float4 moon;       // direction, illuminance
        float4 moonInfo;   // lit fraction, angular radius, waxing, latitude
        float4 atmos;      // aerosol scattering at the ground (per km), camera height (km), ground albedo, aerosol scale height
        float4 cover;      // low, mid, high cloud, rain darkness
        float4 cumulus;    // low cloud base and thickness (km), kind (0 cumulus, 1 stratocumulus, 2 nimbostratus, 3 cumulonimbus), puffiness
        float4 wind;       // drift x, z (km), heading, speed (km/h)
        float4 weather;    // rain, snow, fog, lightning
        float4 night;      // star visibility, light pollution, condensation, local hour
        float4 seed;       // landscape seed, wires, lightning position
        float4 eq0, eq1, eq2;  // rows of the rotation from the horizon frame to J2000 equatorial
        float4 room;       // room light colour, reflection strength
        float4 extra;      // lit windows, dust on the glass, mid cloud base (km), frame light
        float4 lightDir;   // the light the clouds are lit by; w is 1 for the Sun, 0 for the Moon
    };

    inline float rad(float d) { return d * PI / 180; }

    constant float RG = 6360.0, RT = 6460.0;
    constant float3 RAYLEIGH = float3(5.802e-3, 13.558e-3, 33.1e-3);
    constant float3 OZONE = float3(0.650e-3, 1.881e-3, 0.085e-3);

    // A level camera with its lens shifted up, like an architect's: verticals stay vertical and the horizon stays flat.
    float3 forward(constant Sky &k) { return float3(sin(k.cam.x), 0, cos(k.cam.x)); }
    float lensShift(constant Sky &k) { return tan(k.cam.y) * k.cam.z; }
    float3 camRay(float2 p, constant Sky &k) {
        float3 f = forward(k), r = float3(f.z, 0, -f.x);
        return normalize(p.x * r + (p.y + lensShift(k)) * float3(0, 1, 0) + k.cam.z * f);
    }
    // Where a direction lands on screen, in uv.
    float2 screenOf(float3 d, constant Sky &k, float aspect) {
        float3 f = forward(k), r = float3(f.z, 0, -f.x);
        float z = dot(d, f);
        float2 p = float2(dot(d, r) * k.cam.z / z, d.y * k.cam.z / z - lensShift(k));
        return float2(p.x / aspect * 0.5 + 0.5, 0.5 - p.y * 0.5);
    }

    float2 hitSphere(float3 o, float3 d, float r) {
        float b = dot(o, d), c = dot(o, o) - r * r, h = b * b - c;
        if (h < 0) return float2(-1);
        h = sqrt(h);
        return float2(-b - h, -b + h);
    }

    struct Air { float3 rayleigh; float3 mie; float3 scatter; float3 extinct; };
    inline Air air(float h, constant Sky &k) {
        Air a;
        h = max(h, 0.0);
        a.rayleigh = RAYLEIGH * exp(-h / 8.0);
        a.mie = float3(k.atmos.x * exp(-h / k.atmos.w));
        a.scatter = a.rayleigh + a.mie;
        // Haze absorbs about a tenth of what it scatters; ozone absorbs in a layer around 25 km.
        a.extinct = a.rayleigh + a.mie * 1.11 + OZONE * max(0.0, 1 - abs(h - 25) / 15);
        return a;
    }

    inline float rayleighPhase(float c) { return 3 / (16 * PI) * (1 + c * c); }
    // Cornette-Shanks with g = 0.8: the bright glow around the Sun.
    inline float miePhase(float c) {
        const float g = 0.8;
        return 3 / (8 * PI) * ((1 - g * g) * (1 + c * c)) / ((2 + g * g) * pow(max(1 + g * g - 2 * g * c, 1e-4), 1.5));
    }

    // MARK: Tables

    inline float2 transUV(float r, float mu) {
        float H = sqrt(RT * RT - RG * RG);
        float rho = sqrt(max(r * r - RG * RG, 0.0));
        float d = max(0.0, -r * mu + sqrt(max(r * r * (mu * mu - 1) + RT * RT, 0.0)));
        float dmin = RT - r, dmax = rho + H;
        return float2((d - dmin) / max(dmax - dmin, 1e-4), rho / H);
    }

    // Sunlight left at radius r with the light at cosine mu from the zenith, fading out as it sets behind the planet.
    inline float3 lightAt(texture2d<float> T, float r, float mu) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float horizon = -sqrt(max(1 - RG * RG / (r * r), 0.0));
        float seen = smoothstep(horizon - 0.0047, horizon + 0.0047, mu);
        return seen > 0 ? T.sample(s, transUV(r, mu)).rgb * seen : float3(0);
    }

    inline float3 multiAt(texture2d<float> M, float r, float mu) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        return M.sample(s, float2(mu * 0.5 + 0.5, saturate((r - RG) / (RT - RG)))).rgb;
    }

    fragment float4 atmTrans(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]]) {
        float H = sqrt(RT * RT - RG * RG);
        float rho = H * in.uv.y, r = sqrt(rho * rho + RG * RG);
        float dmin = RT - r, dmax = rho + H, d = dmin + in.uv.x * (dmax - dmin);
        float mu = d <= 0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2 * r * d), -1.0, 1.0);
        float3 depth = 0;
        float dt = d / 40;
        for (int i = 0; i < 40; i++) {
            float t = (i + 0.5) * dt;
            depth += air(sqrt(r * r + t * t + 2 * r * mu * t) - RG, k).extinct * dt;
        }
        return float4(exp(-depth), 1);
    }

    // Light scattered more than once, as one isotropic source per height and Sun angle (Hillaire's Ψms).
    fragment float4 atmMulti(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                             texture2d<float> T [[texture(0)]]) {
        float mus = in.uv.x * 2 - 1;
        float r = RG + in.uv.y * (RT - RG) + 0.01;
        float3 sun = float3(0, mus, sqrt(max(1 - mus * mus, 0.0)));
        float3 p = float3(0, r, 0);
        float3 total = 0, transfer = 0;
        const int N = 8;
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
            float cz = 1 - 2 * (j + 0.5) / N, ph = TAU * (i + 0.5) / N, sz = sqrt(1 - cz * cz);
            float3 d = float3(sz * cos(ph), cz, sz * sin(ph));
            float2 g = hitSphere(p, d, RG);
            bool ground = g.x > 0;
            float tMax = ground ? g.x : hitSphere(p, d, RT).y;
            float3 L = 0, F = 0, thr = 1;
            float dt = tMax / 20;
            for (int s = 0; s < 20; s++) {
                float3 q = p + d * ((s + 0.5) * dt);
                float rq = length(q);
                Air a = air(rq - RG, k);
                float3 step = exp(-a.extinct * dt);
                float3 S = lightAt(T, rq, dot(q / rq, sun)) * a.scatter / (4 * PI);
                L += thr * (S - S * step) / a.extinct;
                F += thr * (a.scatter - a.scatter * step) / a.extinct;
                thr *= step;
            }
            if (ground) {
                float3 n = normalize(p + d * tMax);
                L += thr * lightAt(T, RG, dot(n, sun)) * saturate(dot(n, sun)) * k.atmos.z / PI;
            }
            total += L;
            transfer += F;
        }
        float n = N * N;
        return float4(total / n / (1 - transfer / n), 1);
    }

    // MARK: The sky along each view ray

    constant int SLICES = 16;
    // Distance to each slice of haze, from 100 m to 100 km.
    inline float sliceDistance(float i) { return 0.1 * pow(1000.0, i / (SLICES - 1)); }
    inline float sliceOf(float d) { return log(max(d, 0.1) / 0.1) / log(1000.0) * (SLICES - 1); }

    struct Scatter { float3 L; float3 T; };

    // Scattered light and transmittance along a stretch of a ray, for the Sun and the Moon.
    inline void march(thread Scatter &acc, float3 o, float3 rd, float t0, float t1, int steps, constant Sky &k,
                      texture2d<float> T, texture2d<float> M) {
        float dt = (t1 - t0) / steps;
        float cs = dot(rd, k.sun.xyz), cm = dot(rd, k.moon.xyz);
        float prs = rayleighPhase(cs), pms = miePhase(cs), prm = rayleighPhase(cm), pmm = miePhase(cm);
        for (int i = 0; i < steps; i++) {
            float3 q = o + rd * (t0 + (i + 0.5) * dt);
            float rq = length(q);
            float3 up = q / rq;
            Air a = air(rq - RG, k);
            float3 step = exp(-a.extinct * dt);
            float3 S = 0;
            if (k.sun.w > 0) {
                float mu = dot(up, k.sun.xyz);
                S += k.sun.w * (lightAt(T, rq, mu) * (a.rayleigh * prs + a.mie * pms) + multiAt(M, rq, mu) * a.scatter);
            }
            if (k.moon.w > 0) {
                float mu = dot(up, k.moon.xyz);
                // Moonlight is sunlight reflected off grey-brown rock: a little redder.
                S += k.moon.w * float3(1.0, 0.95, 0.86) * (lightAt(T, rq, mu) * (a.rayleigh * prm + a.mie * pmm) + multiAt(M, rq, mu) * a.scatter);
            }
            acc.L += acc.T * (S - S * step) / a.extinct;
            acc.T *= step;
        }
    }

    // One thread per cell of the low-resolution view: the whole sky along the ray, and the haze in front of anything at
    // each slice distance (inscattered light, and how much of what's behind gets through).
    kernel void atmView(constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                        texture2d<float> T [[texture(0)]], texture2d<float> M [[texture(1)]],
                        texture2d<float, access::write> sky [[texture(2)]], texture3d<float, access::write> haze [[texture(3)]],
                        uint2 g [[thread_position_in_grid]]) {
        uint w = sky.get_width(), h = sky.get_height();
        if (g.x >= w || g.y >= h) return;
        float2 uv = (float2(g) + 0.5) / float2(w, h);
        float3 rd = camRay(centered(uv, u.res), k);
        float3 o = float3(0, RG + k.atmos.y, 0);
        float2 ground = hitSphere(o, rd, RG);
        float tMax = ground.x > 0 ? ground.x : hitSphere(o, rd, RT).y;
        Scatter acc = { float3(0), float3(1) };
        float t = 0;
        for (int i = 0; i < SLICES; i++) {
            float next = min(sliceDistance(i), tMax);
            if (next > t) march(acc, o, rd, t, next, i == 0 ? 3 : 2, k, T, M);
            t = max(t, next);
            // Stored already exposed: night light is far too faint for half floats otherwise.
            haze.write(float4(acc.L * k.cam.w, dot(acc.T, float3(0.2126, 0.7152, 0.0722))), uint3(g, i));
        }
        if (tMax > t) march(acc, o, rd, t, tMax, 10, k, T, M);
        sky.write(float4(acc.L * k.cam.w, 1), g);
    }

    // Light from the sky as a whole, for clouds and the land: what falls on a surface facing up, on one facing back
    // toward the camera, and what the ground sends back up. One thread does it all; it's small.
    kernel void atmAmbient(constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                           texture2d<float> T [[texture(0)]], texture2d<float> M [[texture(1)]],
                           device float4 *out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
        if (id > 0) return;
        float3 o = float3(0, RG + k.atmos.y, 0);
        float3 back = -forward(k);
        float3 upSum = 0, backSum = 0;
        const int N = 24;
        for (int i = 0; i < N; i++) {
            // A cosine-weighted spiral over the upper hemisphere.
            float v = (i + 0.5) / N, ph = i * 2.39996;
            float sz = sqrt(v), cz = sqrt(1 - v);
            float3 d = float3(sz * cos(ph), cz, sz * sin(ph));
            Scatter acc = { float3(0), float3(1) };
            march(acc, o, d, 0, hitSphere(o, d, RT).y, 8, k, T, M);
            upSum += acc.L;
            // The same pattern turned to face back toward the camera, with the half below the horizon left dark.
            float3 b = normalize(back * cz + float3(0, 1, 0) * sz * cos(ph) + cross(back, float3(0, 1, 0)) * sz * sin(ph));
            if (b.y > 0.01) {
                Scatter bb = { float3(0), float3(1) };
                march(bb, o, b, 0, hitSphere(o, b, RT).y, 8, k, T, M);
                backSum += bb.L;
            }
        }
        float3 sunGround = k.sun.w * lightAt(T, RG + k.atmos.y, k.sun.y) * max(k.sun.y, 0.0)
                         + k.moon.w * lightAt(T, RG + k.atmos.y, k.moon.y) * max(k.moon.y, 0.0);
        out[0] = float4(upSum / N, 0);
        out[1] = float4(backSum / N * 0.5 + upSum / N * 0.15, 0);
        out[2] = float4((upSum / N + sunGround / PI) * k.atmos.z, 0);
        out[3] = float4(sunGround, 0);
    }

    // The log-average brightness of the sky above the horizon, for exposure (read back a frame later).
    kernel void skyMeter(constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]], device float *out [[buffer(2)]],
                         texture2d<float> sky [[texture(0)]], uint id [[thread_position_in_grid]]) {
        if (id > 0) return;
        float horizon = 0.5 + lensShift(k) * 0.5;
        float sum = 0, n = 0, peak = 0;
        for (int j = 0; j < 12; j++) for (int i = 0; i < 20; i++) {
            float2 uv = float2((i + 0.5) / 20, (j + 0.5) / 12 * (horizon - 0.05));
            float3 c = sky.read(uint2(uv * float2(sky.get_width(), sky.get_height()))).rgb / k.cam.w;
            float lum = dot(c, float3(0.2126, 0.7152, 0.0722)) + 2e-7;
            sum += log(lum);
            peak = max(peak, lum);
            n += 1;
        }
        // Lean toward the brightest part when the light is uneven, as in twilight, so the glow keeps its colour.
        float w = mix(0.3, 0.08, smoothstep(-0.05, 0.1, k.sun.y));
        out[0] = exp(sum / n * (1 - w) + log(peak) * w);
    }

    // MARK: The night sky and the discs

    // Airglow, and the glow of towns reflected off the haze near the horizon.
    float3 nightGlow(float alt, constant Sky &k) {
        float3 airglow = float3(2.2e-7, 3.4e-7, 4.6e-7) * (1 + 1.5 * exp(-max(alt, 0.0) * 8));
        float3 towns = float3(1.0, 0.72, 0.45) * 2.6e-6 * k.night.y * exp(-max(alt, 0.0) * 5) * (0.5 + 2 * k.atmos.x);
        return airglow + towns;
    }

    float3 toEquatorial(float3 v, constant Sky &k) { return float3(dot(k.eq0.xyz, v), dot(k.eq1.xyz, v), dot(k.eq2.xyz, v)); }

    // The Moon, lit from the Sun's direction so its phase and tilt are true, from the LRO map when it has arrived.
    float3 moonDisc(float3 rd, constant Sky &k, float pixel, texture2d<float> map, bool hasMap, thread float &cover) {
        float3 m = k.moon.xyz;
        float radius = k.moonInfo.y;
        cover = 0;
        if (dot(rd, m) < cos(radius * 3) || m.y < -0.02) return 0;
        // Disc coordinates with up toward the celestial pole, so the near side faces us the right way up.
        float3 pole = float3(0, sin(k.moonInfo.w), cos(k.moonInfo.w));
        float3 up = normalize(pole - m * dot(pole, m));
        float3 right = cross(m, up);
        float2 q = float2(dot(rd, right), dot(rd, up)) / radius;
        float r = length(q);
        float edge = 1 - smoothstep(1 - pixel / radius, 1 + pixel / radius, r);
        if (edge <= 0) return 0;
        float z = sqrt(max(1 - r * r, 0.0));
        float3 n = right * q.x + up * q.y - m * z;
        // The near side's longitude and latitude, for the map.
        float lon = atan2(q.x, z), lat = asin(clamp(q.y, -1.0, 1.0));
        float3 albedo = float3(0.12);
        if (hasMap) {
            constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
            albedo = map.sample(s, float2(0.5 + lon / TAU, 0.5 - lat / PI)).rgb;
        }
        float mu0 = dot(n, k.sun.xyz), mu = z;
        // Lommel-Seeliger with a little opposition surge: no limb darkening at full Moon, as the real one.
        float lit = mu0 > 0 ? mu0 / (mu0 + mu) * 2 : 0;
        cover = edge;
        // Earthshine on the dark side.
        float earthshine = 0.0006 * (1 - k.moonInfo.x);
        return albedo * (lit + earthshine) * edge;
    }
    """
}
