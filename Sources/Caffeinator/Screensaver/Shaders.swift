/// Metal code shared by every scene: the uniform block, a full-screen triangle, noise, colour and camera helpers,
/// and the compositor that puts a scene and its text overlay on screen.
enum Shaders {
    static let header = """
    #include <metal_stdlib>
    using namespace metal;

    struct U {
        float2 res;   // render size in pixels
        float time;   // seconds since the scene began
        float dt;
        uint frame;
        float seed;
        float2 pad;
        float4 a, b, c, d;
    };

    struct V { float4 pos [[position]]; float2 uv; };

    // One triangle that covers the target. uv runs 0...1 with (0, 0) at the top left.
    vertex V fullscreen(uint id [[vertex_id]]) {
        float2 p = float2((id << 1) & 2, id & 2);
        V v;
        v.pos = float4(p * 2 - 1, 0, 1);
        v.uv = float2(p.x, 1 - p.y);
        return v;
    }

    constant float PI = 3.14159265358979;
    constant float TAU = 6.28318530717959;

    // Centered coordinates: y runs -1...1, x scales with the aspect ratio, +y is up.
    inline float2 centered(float2 uv, float2 res) {
        float2 p = (uv * 2 - 1) * float2(res.x / res.y, -1);
        return p;
    }

    inline float2x2 rot(float a) { float c = cos(a), s = sin(a); return float2x2(c, s, -s, c); }

    // MARK: Hashes

    inline uint pcg(uint v) {
        uint state = v * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    inline float rnd(thread uint &s) { s = pcg(s); return float(s) / 4294967296.0; }
    inline float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
    inline float hash12(float2 p) { float3 p3 = fract(float3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    inline float2 hash22(float2 p) { float3 p3 = fract(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
    inline float3 hash33(float3 p3) { p3 = fract(p3 * float3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yxx) * p3.zyx); }

    // MARK: Noise

    inline float noise(float2 p) {
        float2 i = floor(p), f = fract(p);
        float2 u = f * f * (3 - 2 * f);
        return mix(mix(hash12(i), hash12(i + float2(1, 0)), u.x), mix(hash12(i + float2(0, 1)), hash12(i + float2(1, 1)), u.x), u.y);
    }
    inline float noise(float3 p) {
        float3 i = floor(p), f = fract(p);
        float3 u = f * f * (3 - 2 * f);
        float n000 = hash33(i).x, n100 = hash33(i + float3(1, 0, 0)).x, n010 = hash33(i + float3(0, 1, 0)).x, n110 = hash33(i + float3(1, 1, 0)).x;
        float n001 = hash33(i + float3(0, 0, 1)).x, n101 = hash33(i + float3(1, 0, 1)).x, n011 = hash33(i + float3(0, 1, 1)).x, n111 = hash33(i + float3(1, 1, 1)).x;
        return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
    }
    inline float fbm(float2 p, int octaves = 5) {
        float v = 0, a = 0.5;
        for (int i = 0; i < octaves; i++) { v += a * noise(p); p = rot(0.5) * p * 2.02 + 17.1; a *= 0.5; }
        return v;
    }
    inline float fbm(float3 p, int octaves = 5) {
        float v = 0, a = 0.5;
        for (int i = 0; i < octaves; i++) { v += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; }
        return v;
    }
    // Divergence-free flow from the curl of a noise field, for smoke-like motion.
    inline float2 curl(float2 p) {
        float e = 0.01;
        float n1 = fbm(p + float2(0, e), 3), n2 = fbm(p - float2(0, e), 3);
        float n3 = fbm(p + float2(e, 0), 3), n4 = fbm(p - float2(e, 0), 3);
        return float2(n1 - n2, n4 - n3) / (2 * e);
    }

    // MARK: Colour

    // Inigo Quilez's cosine palettes.
    inline float3 palette(float t, float3 a, float3 b, float3 c, float3 d) { return a + b * cos(TAU * (c * t + d)); }
    inline float3 hsv(float h, float s, float v) {
        float3 k = clamp(abs(fract(h + float3(0, 2, 1) / 3) * 6 - 3) - 1, 0, 1);
        return v * mix(float3(1), k, s);
    }
    // Colour of a black body at a temperature in kelvin, normalised to its brightest channel.
    inline float3 blackbody(float k) {
        float t = clamp(k, 1000.0, 40000.0) / 100;
        float r = t <= 66 ? 1 : clamp(1.29293618606 * pow(t - 60, -0.1332047592), 0.0, 1.0);
        float g = t <= 66 ? clamp(0.39008157876 * log(t) - 0.63184144378, 0.0, 1.0) : clamp(1.12989086089 * pow(t - 60, -0.0755148492), 0.0, 1.0);
        float b = t >= 66 ? 1 : (t <= 19 ? 0 : clamp(0.54320678911 * log(t - 10) - 1.19625408914, 0.0, 1.0));
        return float3(r, g, b);
    }
    // A smooth rainbow for t in 0...1, from deep red through violet.
    inline float3 spectrum(float t) {
        return clamp(palette(t, float3(0.5), float3(0.5), float3(0.9), float3(0.0, 0.33, 0.67)), 0.0, 1.0);
    }
    // Stephen Hill's fit of the ACES filmic curve. Linear in, display out.
    inline float3 aces(float3 x) {
        const float3x3 inM = float3x3(float3(0.59719, 0.07600, 0.02840), float3(0.35458, 0.90834, 0.13383), float3(0.04823, 0.01566, 0.83777));
        const float3x3 outM = float3x3(float3(1.60475, -0.10208, -0.00327), float3(-0.53108, 1.10813, -0.07276), float3(-0.07367, -0.00605, 1.07602));
        float3 v = inM * x;
        float3 a = v * (v + 0.0245786) - 0.000090537;
        float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
        return clamp(outM * (a / b), 0.0, 1.0);
    }
    inline float3 toSRGB(float3 c) { return select(1.055 * pow(c, 1 / 2.4) - 0.055, c * 12.92, c <= 0.0031308); }
    inline float3 fromSRGB(float3 c) { return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045); }
    // Tone maps linear HDR light for display.
    inline float3 film(float3 c) { return toSRGB(aces(c)); }

    // MARK: 3D

    // A camera ray for centered coordinates p, looking from `eye` toward `target` with focal length `zoom`.
    inline float3 ray(float2 p, float3 eye, float3 target, float zoom) {
        float3 f = normalize(target - eye);
        float3 r = normalize(cross(float3(0, 1, 0), f));
        float3 u = cross(f, r);
        return normalize(p.x * r + p.y * u + zoom * f);
    }
    // Projects a world point to clip space for the same camera, so point sprites line up with ray-marched scenes.
    inline float4 project(float3 q, float3 eye, float3 target, float zoom, float aspect) {
        float3 f = normalize(target - eye);
        float3 r = normalize(cross(float3(0, 1, 0), f));
        float3 u = cross(f, r);
        float3 d = q - eye;
        float z = dot(d, f);
        return float4(dot(d, r) * zoom / aspect, dot(d, u) * zoom, z * 0.001, z);
    }
    inline float2 sphere(float3 ro, float3 rd, float3 c, float r) {
        float3 oc = ro - c;
        float b = dot(oc, rd), h = b * b - dot(oc, oc) + r * r;
        if (h < 0) return float2(-1);
        h = sqrt(h);
        return float2(-b - h, -b + h);
    }

    // MARK: The real sky

    // Directions are unit vectors: equatorial (J2000) with x toward right ascension 0h and z toward the north celestial
    // pole; galactic with x toward the galactic centre and z toward the north galactic pole. Multiply by
    // transpose(EQ_TO_GAL) to go back.
    constant float3x3 EQ_TO_GAL = float3x3(float3(-0.0548755604, 0.4941094279, -0.8676661490),
                                           float3(-0.8734370902, -0.4448296300, -0.1980763734),
                                           float3(-0.4838350155, 0.7469822445, 0.4559837762));
    // ESO's Milky Way photograph is nearly galactic, but tilted by about three degrees. This frame, fitted to 393
    // catalogue stars (0.1 degree median error), puts its stars where the real ones are.
    constant float3x3 EQ_TO_PHOTO = float3x3(float3(-0.0015634, 0.4797950, -0.8773792),
                                             float3(-0.8908374, -0.3992797, -0.2167588),
                                             float3(-0.4543195, 0.7812634, 0.4280436));

    // Linear light from the Milky Way photograph (see `MilkyWay`) toward equatorial direction `d`. `spread` is the angle
    // one screen pixel covers, in radians, which picks the mip level so the sky never shimmers. Jupiter, caught in the
    // photograph in 2009, is painted over with the sky just above it.
    inline float3 milkyWay(texture2d<float> t, float3 d, float spread) {
        constexpr sampler s(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float3 g = EQ_TO_PHOTO * d;
        float2 uv = float2(0.5 - atan2(g.y, g.x) / TAU, 0.5 - asin(clamp(g.z, -1.0, 1.0)) / PI);
        float lod = log2(max(spread * float(t.get_width()) / TAU, 1e-4));
        float3 c = t.sample(s, uv, level(lod)).rgb;
        float jupiter = smoothstep(0.99985, 0.99992, dot(g, float3(0.19237, 0.48919, -0.85068)));
        if (jupiter > 0) c = mix(c, t.sample(s, uv - float2(0, 0.012), level(lod)).rgb, jupiter);
        return c;
    }

    // A soft round point sprite. Vertex shaders that draw particles output this.
    struct P { float4 pos [[position]]; float size [[point_size]]; float4 color; };
    fragment float4 sprite(P in [[stage_in]], float2 pc [[point_coord]]) {
        float d = length(pc - 0.5) * 2;
        float a = smoothstep(1.0, 0.0, d);
        return in.color * a * a;
    }
    fragment float4 solid(P in [[stage_in]]) { return in.color; }
    // With GPU.Blend.fade, scales the target by u.a.x: draw it before new particles to leave trails.
    fragment float4 fadeOut(V in [[stage_in]], constant U &u [[buffer(0)]]) { return float4(0, 0, 0, u.a.x); }

    """

    /// Puts the scene (sampled at its own resolution) and the CoreGraphics overlay on screen, with the fade and a little
    /// dither so dark gradients don't band.
    static let compositor = """
    fragment float4 composite(V in [[stage_in]], constant U &u [[buffer(0)]],
                              texture2d<float> scene [[texture(0)]], texture2d<float> overlay [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = max(scene.sample(s, in.uv).rgb, 0.0);
        float4 o = overlay.sample(s, in.uv);
        c = (c * (1 - o.a) + o.rgb) * u.a.x;
        float n = hash12(in.pos.xy + fract(u.time * 7.3) * 113) + hash12(in.pos.xy * 1.37 + 41.7) - 1;
        return float4(clamp(c + n / 255, 0.0, 1.0), 1);
    }
    """
}
