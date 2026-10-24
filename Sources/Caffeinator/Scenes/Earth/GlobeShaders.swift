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
    // `along` is free for the caller, for dashes or gradients.
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

    /// The globe itself and the final tone map.
    static let passes = """
    // u.a: coast, land, graticule, city lights   u.b: atmosphere, stars, night, glint
    // u.c: coast colour   u.d: graticule spacing, daylight
    fragment float4 globeSurface(V in [[stage_in]], constant U &u [[buffer(0)]], constant GlobeCamera &c [[buffer(1)]],
                                 texture2d<float> land [[texture(0)]], texture2d<float> lights [[texture(1)]]) {
        float3 rd = globeRay(in.uv, u.res, c);
        float3 ro = c.eye;
        float b = -dot(ro, rd);
        float3 closest = ro + rd * b;
        float m = length(closest);
        float pix = globePixel(max(b, 0.01), u.res, c);
        float cover = b > 0 ? smoothstep(1 + pix * 0.6, 1 - pix * 0.6, m) : 0;

        // The surface point; rays that miss use the limb point under them so derivatives stay continuous.
        float h = sqrt(max(1 - m * m, 0.0));
        float3 n = normalize(m < 1 ? closest - rd * h : closest);
        float lat = asin(clamp(n.z, -1.0, 1.0)), lon = atan2(n.y, n.x);
        float2 tuv = float2(lon / TAU + 0.5, 0.5 - lat / PI);
        float3 dx = dfdx(n), dy = dfdy(n);
        float rxy2 = max(n.x * n.x + n.y * n.y, 1e-6), rxy = sqrt(rxy2);
        float2 gx = float2((n.x * dx.y - n.y * dx.x) / (rxy2 * TAU), -dx.z / (rxy * PI));
        float2 gy = float2((n.x * dy.y - n.y * dy.x) / (rxy2 * TAU), -dy.z / (rxy * PI));
        constexpr sampler aniso(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge, max_anisotropy(8));
        float cv = land.sample(aniso, tuv, gradient2d(gx, gy)).r;
        float soft = land.sample(aniso, tuv, gradient2d(gx * 5, gy * 5)).r;
        float wide = land.sample(aniso, tuv, gradient2d(gx * 14, gy * 14)).r;

        // Coastline: the half-coverage contour, about a pixel wide whatever the zoom, plus a soft glow. The mask's
        // edge spans about a texel of the mip level sampled, so its gradient per pixel is at least that; fwidth alone
        // misses edges that fall between 2x2 quads.
        float2 texels = float2(land.get_width(), land.get_height());
        float perPixel = min(length(float2(gx.x, gx.y) * texels), length(float2(gy.x, gy.y) * texels));
        float fw = max(fwidth(cv), 0.8 * saturate(perPixel));
        float dpx = abs(cv - 0.5) / fw;
        float line = exp(-dpx * dpx * 1.4) * smoothstep(0.0, 0.12, cv) * smoothstep(1.0, 0.88, cv);
        float glow = 4 * soft * (1 - soft) * 0.35 + 4 * wide * (1 - wide) * 0.12;

        // Graticule lines, fading where they crowd together.
        float spacing = u.d.x;
        float latd = lat * 180 / PI, lond = lon * 180 / PI;
        float perLon = max(length(float2(gx.x, gy.x)) * 360, 1e-5), perLat = max(length(float2(gx.y, gy.y)) * 180, 1e-5);
        float dLat = abs(fract(latd / spacing + 0.5) - 0.5) * spacing / perLat;
        float dLon = abs(fract(lond / spacing + 0.5) - 0.5) * spacing / perLon;
        float grid = saturate(1 - dLat) * smoothstep(3.0, 10.0, spacing / perLat);
        grid = max(grid, saturate(1 - dLon) * smoothstep(3.0, 10.0, spacing / perLon) * smoothstep(84.0, 70.0, abs(latd)));
        float equator = saturate(1.2 - abs(latd) / perLat) * 0.6;

        // Sunlight, with a civil-twilight band at the terminator.
        float sd = dot(n, c.sun);
        float day = smoothstep(-0.1, 0.12, sd) * u.d.y;
        float lambert = max(sd, 0.0) * u.d.y;
        float nightLevel = u.b.z;
        float3 ocean = mix(float3(0.0025, 0.0045, 0.011) * nightLevel, float3(0.008, 0.026, 0.07) * (0.35 + 0.65 * lambert), day);
        float3 ground = mix(float3(0.011, 0.015, 0.024) * nightLevel, float3(0.034, 0.056, 0.082) * (0.3 + 0.7 * lambert), day) * u.a.y;
        float3 col = mix(ocean, ground, cv);
        float3 refl = reflect(rd, n);
        float glint = max(dot(refl, c.sun), 0.0);
        col += (1 - cv) * pow(glint, 500.0) * 0.03 * day * u.b.w * float3(0.75, 0.85, 1.0);

        constexpr sampler mip(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge, max_anisotropy(8));
        float lit = lights.sample(mip, tuv, gradient2d(gx, gy)).r;
        col += float3(1.0, 0.58, 0.25) * lit * lit * 1.6 * (1 - day) * u.a.w;

        float3 coast = u.c.rgb * u.a.x * mix(0.55, 1.0, day);
        col += coast * (line * 0.32 + glow * 0.12);
        float3 gridColor = float3(0.22, 0.4, 0.85);
        col += gridColor * (grid * 0.022 + equator * 0.02) * u.a.z * mix(0.6, 1.0, day);

        // Haze toward the limb, blue by day.
        float mu = saturate(dot(n, -rd));
        float3 sky = float3(0.18, 0.42, 1.0);
        col += sky * pow(1 - mu, 4.0) * (0.03 + 0.35 * day) * u.b.x;

        // Space: a faint glow of atmosphere around the limb, brighter on the day side, and optional stars.
        float3 bg = float3(0.0012, 0.0016, 0.0032);
        float altitude = max(m - 1, 0.0);
        float dayRim = smoothstep(-0.35, 0.45, dot(normalize(closest), c.sun)) * u.d.y;
        float rim = exp(-altitude / 0.025) * 0.3 + exp(-altitude / 0.1) * 0.045;
        bg += sky * rim * (0.08 + 0.9 * dayRim) * u.b.x * (b > 0 ? 1 : 0);
        if (u.b.y > 0) {
            float3 q = rd * 180;
            float3 cell = floor(q);
            float3 hc = hash33(cell);
            if (hc.x > 0.93) {
                float3 at = normalize(cell + 0.25 + 0.5 * hash33(cell + 17.3));
                float pa = 2 / (u.res.y * c.zoom);
                float d = length(rd - at) / pa;
                bg += mix(float3(0.7, 0.8, 1.0), float3(1.0, 0.85, 0.7), hc.z) * exp(-d * d * 1.2) * pow(hc.y, 4.0) * 0.5 * u.b.y;
            }
        }
        return float4(mix(bg, col, cover), 1);
    }

    fragment float4 globeFinish(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                                texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = light.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * u.a.x;
        float2 p = in.uv - 0.5;
        c *= u.a.y * (1 - dot(p, p) * 0.35);
        return float4(film(max(c, 0.0)), 1);
    }
    """
}
