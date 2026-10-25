extension Globe {
    /// Metal shared by the globe scenes: the camera, projection, occlusion and antialiased ribbons for lines.
    /// Scenes compile it in front of their own source: `gpu.library(Globe.metal + source)`.
    static let metal = """
    struct GlobeCamera {
        float3x3 basis;   // right, up, forward
        float3 eye;       // Earth radii, Earth-fixed
        float3 sun;       // unit vector toward the Sun
        float2 shift;     // the Earth's centre on screen, centered units
        float zoom;
        float aspect;
        float gmst;       // Greenwich sidereal angle, radians
    };

    // Earth-fixed position: latitude and longitude in degrees, r in Earth radii.
    inline float3 geo(float lat, float lon, float r) {
        float a = lat * (PI / 180), o = lon * (PI / 180);
        return r * float3(cos(a) * cos(o), cos(a) * sin(o), sin(a));
    }

    inline float3 globeRay(float2 uv, float2 res, constant GlobeCamera &c) {
        float2 p = centered(uv, res) - c.shift;
        return normalize(p.x * c.basis[0] + p.y * c.basis[1] + c.zoom * c.basis[2]);
    }

    inline float4 globeClip(float3 q, constant GlobeCamera &c) {
        float3 d = q - c.eye;
        float z = dot(d, c.basis[2]);
        float2 s = float2(dot(d, c.basis[0]), dot(d, c.basis[1])) * c.zoom + c.shift * z;
        return float4(s.x / c.aspect, s.y, z * 0.5, z);
    }

    // An Earth-fixed direction in the sky's frame (J2000 equatorial; precession since 2000 is a third of a degree).
    inline float3 globeSky(float3 d, constant GlobeCamera &c) {
        float cg = cos(c.gmst), sg = sin(c.gmst);
        return float3(cg * d.x - sg * d.y, sg * d.x + cg * d.y, d.z);
    }

    // Size of one pixel, in Earth radii, at distance z along the view.
    inline float globePixel(float z, float2 res, constant GlobeCamera &c) { return 2 * z / (res.y * c.zoom); }

    // 1 when no part of the Earth lies between the camera and q, fading to 0 just behind the limb.
    inline float globeShown(float3 q, constant GlobeCamera &c) {
        float3 d = q - c.eye;
        float l = length(d);
        float3 rd = d / l;
        float b = -dot(c.eye, rd);
        float m = length(c.eye + rd * b);
        if (m >= 1 || b >= l) return 1;
        float entry = b - sqrt(1 - m * m);
        if (entry > l - 2e-3) return 1;
        return 1 - saturate((1 - m) / 0.012);
    }

    // For points on the surface: 1 on the near side, fading out at the limb.
    inline float globeFacing(float3 q, constant GlobeCamera &c) {
        return smoothstep(-0.02, 0.06, dot(normalize(q), normalize(c.eye - q)));
    }

    // Antialiased lines: six vertices per segment from a to b (clip space), expanded to `width` pixels on screen.
    // `along` is free for the caller, for dashes or gradients. Colours are premultiplied, so the same ribbon works
    // added into light or alpha-blended over the developed picture.
    struct Ribbon { float4 pos [[position]]; float across; float half_width; float along; float4 color; };

    inline Ribbon ribbon(uint corner, float4 a, float4 b, float width, float4 color, float2 res) {
        Ribbon o;
        float w = max(width, 1.0) * 0.5 + 1;
        o.along = 0;
        if (a.w <= 1e-3 || b.w <= 1e-3) { o.pos = float4(0, 0, -1, 1); o.color = 0; o.across = 0; o.half_width = 1; return o; }
        float2 pa = a.xy / a.w, pb = b.xy / b.w;
        float2 dir = (pb - pa) * res;
        float len = length(dir);
        dir = len > 1e-5 ? dir / len : float2(1, 0);
        float2 normal = float2(-dir.y, dir.x);
        // corners 0..5 make two triangles: (a-, a+, b-) and (b-, a+, b+)
        bool atB = corner == 2 || corner == 3 || corner == 5;
        float side = (corner == 1 || corner == 4 || corner == 5) ? 1 : -1;
        float4 p = atB ? b : a;
        float2 ext = dir * (atB ? 1 : -1) * 0.5;
        p.xy += (normal * side * w + ext) / res * 2 * p.w;
        o.pos = p;
        o.across = side * w;
        o.half_width = max(width, 1.0) * 0.5;
        o.color = color * min(width, 1.0);
        return o;
    }

    fragment float4 ribbonFill(Ribbon in [[stage_in]]) {
        return in.color * saturate(in.half_width + 0.5 - abs(in.across));
    }

    """

    /// The Earth and its air, and the passes that turn yesterday's satellite mosaic into a cloud layer.
    static let passes = """
    constant float KM = 1.0 / 6371.0;
    constant float AIR_TOP = 1.0 + 90.0 / 6371.0;
    constant float CLOUD_TOP = 1.0 + 6.0 / 6371.0;
    constant float HR = 8.0 / 6371.0;          // Rayleigh scale height
    constant float HM = 1.2 / 6371.0;          // haze (Mie) scale height
    // Scattering and extinction per Earth radius at sea level (Bruneton's and Hillaire's values), and ozone absorption.
    constant float3 BETA_R = float3(36.965, 86.378, 210.88);
    constant float BETA_MS = 25.458;
    constant float BETA_ME = 28.287;
    constant float3 BETA_O = float3(4.141, 11.984, 0.5415);
    constant float OZONE = 15.0 / 6371.0;      // the ozone layer's thickness at its peak density

    // Schüler's approximation of the Chapman function: optical depth to space, in scale heights, from height h
    // (scale heights) for a planet of radius X (scale heights), at cosine mu from the zenith. Rays that meet the
    // ground come out enormous, so the Earth's own shadow falls out of it.
    inline float chapman(float X, float h, float mu) {
        float c = sqrt(X + h);
        if (mu >= 0) return c / (c * mu + 1) * exp(-h);
        float x0 = sqrt(1 - mu * mu) * (X + h);
        return 2 * sqrt(x0) * exp(min(X - x0, 60.0)) - c / (1 - c * mu) * exp(-h);
    }

    // Optical depth from radius r to space, toward a direction at cosine mu from straight up.
    inline float3 airDepth(float r, float mu) {
        float h = max(r - 1, 0.0);
        float cr = chapman(1 / HR, h / HR, mu), cm = chapman(1 / HM, h / HM, mu);
        float mass = min(cr * exp(h / HR), 40.0);
        float ozone = OZONE * saturate((40 * KM - h) / (30 * KM));
        return BETA_R * HR * cr + BETA_ME * HM * cm + BETA_O * ozone * mass;
    }

    struct Air { float3 light; float3 through; };

    // Sunlight scattered toward the camera by air (Rayleigh) and haze (Mie) along a ray, up to tEnd, with a little
    // more for light scattered more than once. `through` is how much of what lies behind gets through.
    inline Air airAlong(float3 ro, float3 rd, float tEnd, float3 sun) {
        Air o;
        o.light = 0;
        o.through = 1;
        float2 top = sphere(ro, rd, float3(0), AIR_TOP);
        if (top.y <= 0) return o;
        float t0 = max(top.x, 0.0), t1 = min(top.y, tEnd);
        if (t1 <= t0) return o;
        // Samples crowd toward the lowest point on the ray, where the air is densest.
        float tLow = clamp(-dot(ro, rd), t0, t1);
        float len0 = tLow - t0, len1 = t1 - tLow;
        const int N = 16;
        int n0 = len1 <= 1e-6 ? N : (len0 <= 1e-6 ? 0 : clamp(int(N * len0 / (len0 + len1) + 0.5), 1, N - 1));
        float cv = dot(rd, sun);
        float pr = 3 / (16 * PI) * (1 + cv * cv);
        const float g = 0.76;
        float pm = 3 / (8 * PI) * (1 - g * g) * (1 + cv * cv) / ((2 + g * g) * pow(1 + g * g - 2 * g * cv, 1.5));
        float3 depth = 0, sum = 0;
        for (int i = 0; i < N; i++) {
            bool first = i < n0;
            float k = first ? float(n0) : float(N - n0);
            float x = ((first ? float(i) : float(i - n0)) + 0.5) / k;
            float t, dt;
            if (first) { float w = 1 - x; t = tLow - len0 * w * w; dt = 2 * len0 * w / k; }
            else { t = tLow + len1 * x * x; dt = 2 * len1 * x / k; }
            float3 q = ro + rd * t;
            float r = length(q), h = max(r - 1, 0.0);
            float dr = exp(-h / HR), dm = exp(-h / HM), dz = max(1 - abs(h - 25 * KM) / (15 * KM), 0.0);
            float3 seg = (BETA_R * dr + BETA_ME * dm + BETA_O * dz) * dt;
            float3 view = exp(-(depth + 0.5 * seg));
            float mus = dot(q, sun) / r;
            float3 sunT = exp(-airDepth(r, mus));
            float3 single = sunT * (BETA_R * dr * pr + BETA_MS * dm * pm);
            float3 multi = (BETA_R * dr + BETA_MS * dm) * 0.022 * smoothstep(-0.22, 0.3, mus);
            sum += view * (single + multi) * dt;
            depth += seg;
        }
        o.light = sum * PI;
        o.through = exp(-depth);
        return o;
    }

    // u.a: city lights, atmosphere, Milky Way, airglow   u.b: sunglint, clouds, has photo, has sky
    // u.c: Moon direction, moonlight   u.d: time, night-sky glow
    fragment float4 globeSurface(V in [[stage_in]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                                 texture2d<float> surface [[texture(0)]], texture2d<float> clouds [[texture(1)]],
                                 texture2d<float> lights [[texture(2)]], texture2d<float> land [[texture(3)]],
                                 texture2d<float> sky [[texture(4)]]) {
        float3 rd = globeRay(in.uv, u.res, c);
        float3 ro = c.eye;
        float3 sun = c.sun;
        float b = -dot(ro, rd);
        float3 closest = ro + rd * b;
        float m = length(closest);
        float pix = globePixel(max(b, 0.01), u.res, c);
        float cover = b > 0 ? smoothstep(1 + pix * 0.6, 1 - pix * 0.6, m) : 0;
        float h = sqrt(max(1 - m * m, 0.0));
        bool hit = m < 1 && b > 0;
        Air air = airAlong(ro, rd, hit ? b - h : 1e9, sun);
        air.light *= u.a.y;
        air.through = mix(1.0, air.through, saturate(u.a.y));

        // The surface point; rays that miss use the limb point under them so derivatives stay continuous.
        float3 n = normalize(hit ? closest - rd * h : closest);
        float lat = asin(clamp(n.z, -1.0, 1.0)), lon = atan2(n.y, n.x);
        float2 tuv = float2(lon / TAU + 0.5, 0.5 - lat / PI);
        float3 dx = dfdx(n), dy = dfdy(n);
        float rxy2 = max(n.x * n.x + n.y * n.y, 1e-6), rxy = sqrt(rxy2);
        float2 gx = float2((n.x * dx.y - n.y * dx.x) / (rxy2 * TAU), -dx.z / (rxy * PI));
        float2 gy = float2((n.x * dy.y - n.y * dy.x) / (rxy2 * TAU), -dy.z / (rxy * PI));

        float3 col = 0;
        if (cover > 0) {
            constexpr sampler aniso(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge, max_anisotropy(8));

            float cv = land.sample(aniso, tuv, gradient2d(gx, gy)).r;
            float water = 1 - smoothstep(0.3, 0.7, cv);
            float3 albedo = u.b.z > 0 ? surface.sample(aniso, tuv, gradient2d(gx, gy)).rgb
                                      : mix(float3(0.006, 0.016, 0.04), float3(0.13, 0.12, 0.085), cv);
            // The Blue Marble's seas are brighter and bluer than they look from orbit, where the air adds the blue.
            albedo *= mix(float3(1), float3(0.55, 0.62, 0.66), water);

            // Clouds sit 6 km up: where the ray meets that shell, and where the sunlight crossed it on its way down.
            float2 shell = sphere(ro, rd, float3(0), CLOUD_TOP);
            float3 nc = shell.x > 0 ? normalize(ro + rd * shell.x) : n;
            float2 cuv = float2(atan2(nc.y, nc.x) / TAU + 0.5, 0.5 - asin(clamp(nc.z, -1.0, 1.0)) / PI);
            float cloud = saturate(clouds.sample(aniso, cuv, gradient2d(gx, gy)).r * u.b.y);
            float mus = dot(n, sun);
            float3 ns = normalize(n + sun * (CLOUD_TOP - 1) / max(mus, 0.08));
            float2 suv = float2(atan2(ns.y, ns.x) / TAU + 0.5, 0.5 - asin(clamp(ns.z, -1.0, 1.0)) / PI);
            float shade = 1 - 0.78 * saturate(clouds.sample(aniso, suv, gradient2d(gx * 3, gy * 3)).r * u.b.y);

            float3 sunT = exp(-airDepth(1, mus));
            float lit = max(mus, 0.0);
            // Skylight: about half of what the air scatters out of the beam comes down as blue light from the sky.
            float3 skylight = (1 - exp(-airDepth(1, 1))) * 0.5 * smoothstep(-0.15, 0.5, mus) * (0.35 + 0.65 * lit) + 0.004;
            float3 ground = albedo * (sunT * lit * shade + skylight);

            // Sunglint: wind-roughened sea as an anisotropic GGX surface. Waves are steeper along the Sun-viewer plane than
            // across it, so the patch stretches along that plane; wind streaks and gusts modulate the roughness, and the
            // highlight rolls off softly so it never blooms into a disc.
            float3 hv = normalize(sun - rd);
            float nv = max(dot(n, -rd), 1e-3);
            float3 planeN = cross(sun, -rd);
            float3 tA = cross(n, planeN);
            tA = length(tA) > 1e-4 ? normalize(tA) : float3(1, 0, 0);
            float3 tC = cross(n, tA);
            float gust = fbm(float3(n * 140.0 + float3(0, 0, u.d.x * 0.01)), 4);
            float streak = noise(float2(dot(n, tC) * 900.0, dot(n, tA) * 90.0));
            float rough = 0.75 + 0.5 * gust + 0.25 * (streak - 0.5);
            float ax = 0.17 * rough, ay = 0.07 * rough;
            float hx = dot(hv, tA), hy = dot(hv, tC), hz = max(dot(hv, n), 1e-3);
            float qd = hx * hx / (ax * ax) + hy * hy / (ay * ay) + hz * hz;
            float D = 1 / (PI * ax * ay * qd * qd);
            float F = 0.02 + 0.98 * pow(1 - saturate(dot(hv, -rd)), 5.0);
            const float k = 0.07;
            float G = lit / (lit * (1 - k) + k) / (nv * (1 - k) + k);
            float3 gl = float3(sunT * shade * PI * D * F * G / 4 * u.b.x);
            gl = 0.75 * gl / (1 + gl / 2.5);
            ground += water * gl * (1 - cloud);

            // Cloud tops: bright, lit a little past the terminator, reddened by the long path at sunset, and lumpy:
            // thicker cloud stands taller, so the opacity's slope tilts the tops toward or away from a low Sun.
            float muc = dot(nc, sun), shaded = muc;
            if (cloud > 0.02) {
                const float nudge = 0.15 / 360;
                float ce = clouds.sample(aniso, cuv + float2(nudge, 0), gradient2d(gx, gy)).r;
                float cn = clouds.sample(aniso, cuv - float2(0, nudge * 2), gradient2d(gx, gy)).r;
                float clat = max(sqrt(1 - nc.z * nc.z), 0.05);
                float3 east = normalize(float3(-nc.y, nc.x, 0) + float3(1e-6, 0, 0)), north = cross(nc, east);
                const float relief = 12.0 / 6371 / (nudge * TAU);
                float2 slope = float2((ce - cloud) / clat, cn - cloud) * relief * u.b.y;
                shaded = dot(normalize(nc - slope.x * east - slope.y * north), sun) * 0.6 + muc * 0.4;
            }
            float3 sunC = exp(-airDepth(CLOUD_TOP, muc));
            float3 tops = 0.8 * (sunC * saturate(shaded * 0.88 + 0.12) * smoothstep(-0.1, 0.03, muc) + skylight * 1.4);
            col = mix(ground, tops, cloud);

            // The night side: city lights, dimmed and spread by the clouds over them, and moonlight.
            float night = smoothstep(0.04, -0.1, mus);
            float raw = lights.sample(aniso, tuv, gradient2d(gx, gy)).r;
            float soft = lights.sample(aniso, tuv, gradient2d(gx * 6, gy * 6)).r;
            float city = mix(max(fromSRGB(float3(raw)).r - 0.075, 0.0), max(fromSRGB(float3(soft)).r - 0.075, 0.0) * 0.6, cloud) / 0.925;
            col += float3(1.0, 0.6, 0.28) * city * 1.4 * (1 - 0.65 * cloud) * night * u.a.x;
            // Moonlight, and the faint glow of the night sky itself, so a long exposure shows clouds on the night side.
            float mum = max(dot(n, u.c.xyz), 0.0);
            col += mix(albedo, float3(0.8), cloud) * (mum * u.c.w + u.d.y) * night * float3(0.8, 0.88, 1.0);

            col = col * air.through + air.light;
        }

        float3 space = air.light;
        if (cover < 1) {
            if (u.b.w > 0 && u.a.z > 0) {
                float spread = 2 / (u.res.y * c.zoom);
                space += milkyWay(sky, globeSky(rd, c), spread) * u.a.z * air.through;
            }
            // Airglow: oxygen's faint green line about 97 km up, seen edge-on above the night-side limb.
            if (b > 0 && u.a.w > 0) {
                const float ra = 1 + 97 * KM;
                float w = max(6 * KM, pix);
                float edge = exp(-pow((m - ra) / w, 2.0)) * (6 * KM / w);
                float dark = smoothstep(0.05, -0.2, dot(normalize(closest), sun));
                space += float3(0.3, 1.0, 0.4) * 0.035 * edge * dark * u.a.w;
            }
        }
        return float4(mix(space, col, cover), 1);
    }

    // Cloud opacity from a day's VIIRS mosaic: how far each pixel is from the cloud-free Blue Marble toward white
    // cloud. Where the mosaic has no data (polar night), NASA's older cloud map stands in. Writes opacity, its square
    // (for the local spread), and whether it was filled in. u.a: has Blue Marble, has fallback
    kernel void cloudOpacity(constant U &u [[buffer(0)]], texture2d<float> viirs [[texture(0)]], texture2d<float> surface [[texture(1)]],
                             texture2d<float> fallback [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
        float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float3 v = viirs.sample(s, uv, level(0)).rgb;
        float lod = max(log2(float(surface.get_width()) / float(out.get_width())), 0.0);
        float3 g = u.a.x > 0 ? surface.sample(s, uv, level(lod)).rgb : float3(0.03, 0.04, 0.06);
        const float3 white = float3(0.86, 0.88, 0.92);
        float3 a3 = saturate((v - g) / max(white - g, 0.04));
        // Over snow and ice a cloud can't be told from the ground, so trust the difference less there.
        float a = dot(a3, float3(0.25, 0.3, 0.45)) * smoothstep(0.04, 0.25, white.b - g.b);
        bool none = max(v.r, max(v.g, v.b)) < 0.0035;
        if (none) a = u.a.y > 0 ? smoothstep(0.1, 0.9, fallback.sample(s, uv, level(0)).r) : 0;
        out.write(float4(a, a * a, none ? 1 : 0, 0), gid);
    }

    // Takes the VIIRS sunglint out: smooth grey bands over the sea where each swath caught the Sun's reflection. Over
    // open water in the tropics, the part of the opacity the neighbourhood shares evenly (its mean less 1.5 standard
    // deviations, across about a degree) is removed; real clouds are lumpy and survive. u.a: subsolar latitude
    kernel void cloudFinish(constant U &u [[buffer(0)]], texture2d<float> raw [[texture(0)]], texture2d<float> land [[texture(1)]],
                            texture2d<float, access::write> out [[texture(2)]], uint2 gid [[thread_position_in_grid]]) {
        float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float4 here = raw.sample(s, uv, level(0));
        float4 near = raw.sample(s, uv, level(3.5));
        float sigma = sqrt(max(near.y - near.x * near.x, 0.0));
        float lat = 90 - uv.y * 180;
        float water = 1 - land.sample(s, uv, level(1.5)).r;
        float band = exp(-pow((lat - u.a.x) / 28, 2.0)) * water * (1 - here.z);
        float glint = clamp(near.x - 1.5 * sigma, 0.0, 0.35) * band;
        float a = saturate((here.x - glint) / (1 - glint));
        out.write(float4(smoothstep(0.03, 0.95, a)), gid);
    }
    """
}
