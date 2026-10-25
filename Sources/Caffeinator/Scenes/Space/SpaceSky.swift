import Metal
import simd

/// The real sky behind the space scenes, seen through any pinhole camera: ESO's Milky Way photograph, sampled a little
/// softer than the screen so its own stars melt into the glow of the star clouds, with the 5,044 catalogue stars drawn
/// crisp on top at their true brightness. Directions are J2000 equatorial. Until the photograph arrives, or offline,
/// the band of the Milky Way comes from the catalogue's outline instead.
@MainActor final class SkyBackdrop {
    /// A pinhole camera: unit axes in equatorial coordinates, and `focal`, the distance to the image plane in units of
    /// half the image height (2 gives a 53° vertical field). Matches `ray` in `SpaceSky.metal`.
    struct Camera {
        var right: SIMD3<Float>
        var up: SIMD3<Float>
        var forward: SIMD3<Float>
        var focal: Float
    }

    private(set) var texture: MTLTexture?
    /// True once the photograph has loaded or failed to.
    private(set) var loaded = false
    private let back: MTLRenderPipelineState
    private let starPipe: MTLRenderPipelineState
    private let stars: MTLBuffer
    private let starCount: Int
    private let outline: MTLTexture

    init(gpu: GPU) throws {
        let library = try gpu.library(SpaceSky.metal + Self.source)
        back = try gpu.pipeline(library, "skyBackdrop")
        starPipe = try gpu.pipeline(library, "skyStarFragment", vertex: "skyStarVertex", blend: .add)
        let list = SkyCatalog.stars.flatMap { s -> [SIMD4<Float>] in
            let v = SkyCatalog.unit(ra: Double(s.ra), dec: Double(s.dec))
            return [SIMD4(Float(v.x), Float(v.y), Float(v.z), s.mag), SIMD4(SpaceSky.tint(SkyCatalog.temperature(bv: Double(s.bv))), 0)]
        }
        starCount = list.count / 2
        stars = gpu.buffer(list)
        var levels = SkyCatalog.milkyWay.map { $0 * 51 }
        outline = levels.withUnsafeMutableBytes { gpu.texture(720, 360, .r8Unorm, bytes: $0.baseAddress!, bytesPerRow: 720) }
        Task { @MainActor [weak self] in
            let t = await MilkyWay.texture(gpu)
            self?.texture = t
            self?.loaded = true
        }
    }

    /// Draws the sky into `target`. `exposure` scales the light: at 1 the galactic centre's star clouds come out near
    /// 0.2 and a magnitude-0 star carries the light it really has relative to them. `soften` is how many mip levels
    /// blurrier than the screen to sample the photograph; `stars` scales the catalogue stars alone.
    func draw(_ cmd: MTLCommandBuffer, into target: MTLTexture, camera c: Camera, exposure: Float, soften: Float = 1,
              stars starLevel: Float = 1, clear: Bool = true) {
        let res = SIMD2(Float(target.width), Float(target.height))
        let u = U(res: res, a: SIMD4(c.right, c.focal), b: SIMD4(c.up, exposure), c: SIMD4(c.forward, soften),
                  d: SIMD4(starLevel, texture == nil ? 1 : 0, res.y / 1000, 0))
        cmd.draw(back, into: target, clear: clear, uniforms: u, textures: [texture ?? outline, outline])
        cmd.draw(starPipe, into: target, clear: false, uniforms: u, buffers: [stars], vertices: 4, instances: starCount,
                 primitive: .triangleStrip)
    }

    private static let source = """
    // a: right, focal   b: up, exposure   c: forward, soften   d: star level, no photograph, pixel density
    fragment float4 skyBackdrop(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> photo [[texture(0)]],
                                texture2d<float> outline [[texture(1)]]) {
        float2 p = centered(in.uv, u.res);
        float3 d = skyRay(p, u.a.xyz, u.b.xyz, u.c.xyz, u.a.w);
        float spread = skySpread(u.res.y, u.a.w, p) * exp2(u.c.w);
        float3 c = u.d.y > 0.5 ? milkyWayOutline(outline, d) : milkyWay(photo, d, spread);
        return float4(c * u.b.w, 1);
    }

    struct SkyStarOut { float4 pos [[position]]; float2 offset; float4 color; float sigma; };

    // Catalogue stars as Gaussian points of light, sized by the pixel density rather than by brightness: brighter stars
    // look bigger only as the lens and the curve spread their light, as in a photograph.
    vertex SkyStarOut skyStarVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                    const device float4 *stars [[buffer(1)]]) {
        float4 s = stars[2 * iid], col = stars[2 * iid + 1];
        SkyStarOut o;
        o.pos = float4(0, 0, 2, 1);
        o.color = 0;
        o.offset = 0;
        o.sigma = 1;
        float3 c = float3(dot(s.xyz, u.a.xyz), dot(s.xyz, u.b.xyz), dot(s.xyz, u.c.xyz));
        if (c.z <= 0.01) return o;
        float2 p = c.xy / c.z * u.a.w;
        float2 px = float2((p.x / (u.res.x / u.res.y) * 0.5 + 0.5) * u.res.x, (0.5 - p.y * 0.5) * u.res.y);
        float flux = starFlux(s.w, skySpread(u.res.y, u.a.w, p)) * u.b.w * u.d.x;
        float sigma = 0.7 * u.d.z;
        float peak = flux / (TAU * sigma * sigma);
        if (peak < 0.0015) return o;
        float extent = sigma * clamp(sqrt(2 * log(peak / 0.0015)), 1.5, 9.0);
        float2 corner = float2(vid & 1, vid >> 1) * 2 - 1;
        o.pos = float4((px.x + corner.x * extent) / u.res.x * 2 - 1, 1 - (px.y + corner.y * extent) / u.res.y * 2, 0, 1);
        o.offset = corner * extent;
        o.color = float4(col.rgb * peak, 0);
        o.sigma = sigma;
        return o;
    }

    fragment float4 skyStarFragment(SkyStarOut in [[stage_in]]) {
        return in.color * exp(-dot(in.offset, in.offset) / (2 * in.sigma * in.sigma));
    }
    """
}

enum SpaceSky {
    /// Colour of a star's light at a temperature, as linear RGB with unit luminance.
    static func tint(_ kelvin: Double) -> SIMD3<Float> {
        let c = SkyCatalog.blackbody(kelvin)
        return c / max((c * SIMD3(0.2126, 0.7152, 0.0722)).sum(), 0.05)
    }

    /// Rotates an ecliptic (J2000) direction into equatorial coordinates.
    static func equatorial(ecliptic v: SIMD3<Double>) -> SIMD3<Double> {
        let e = Astro.rad(23.4392911)
        return SIMD3(v.x, v.y * cos(e) - v.z * sin(e), v.y * sin(e) + v.z * cos(e))
    }

    /// Rotates a galactic direction into equatorial coordinates.
    static func equatorial(galactic g: SIMD3<Double>) -> SIMD3<Double> {
        let r = SkyCatalog.galactic
        return SIMD3(r[0].x * g.x + r[1].x * g.y + r[2].x * g.z,
                     r[0].y * g.x + r[1].y * g.y + r[2].y * g.z,
                     r[0].z * g.x + r[1].z * g.y + r[2].z * g.z)
    }

    /// Metal shared by the space scenes that show the real sky or a star up close.
    static let metal = """
    // The ray through centered point p for a pinhole camera with these axes and focal length.
    inline float3 skyRay(float2 p, float3 right, float3 up, float3 forward, float focal) {
        return normalize(forward * focal + p.x * right + p.y * up);
    }

    // The angle one pixel covers at centered point p, in radians.
    inline float skySpread(float height, float focal, float2 p) {
        return 2 / (height * focal) / (1 + dot(p, p) / (focal * focal));
    }

    // Light from a star of magnitude m for one pixel covering `spread` radians, in the units `milkyWay` returns:
    // the photograph's galactic-centre star clouds (0.19) are about 20 magnitudes per square arcsecond.
    inline float starFlux(float m, float spread) {
        float pixel = spread * 206265.0;
        return 0.19 * pow(10.0, -0.4 * (m - 20.0)) / (pixel * pixel);
    }

    // The Milky Way from the catalogue's half-degree outline, for when the photograph isn't available.
    inline float3 milkyWayOutline(texture2d<float> t, float3 d) {
        constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
        float3 g = EQ_TO_GAL * d;
        float2 uv = float2(fract(atan2(g.y, g.x) / TAU), asin(clamp(g.z, -1.0, 1.0)) / PI + 0.5);
        float level = 0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -1; j <= 1; j++) level += t.sample(s, uv + float2(i, j) * float2(1.0 / 720, 1.0 / 360)).r;
        }
        level /= 15;
        float core = exp(-pow(acos(clamp(g.x, -1.0, 1.0)) / 0.6, 2.0));
        float3 tint = mix(float3(0.8, 0.82, 0.9), float3(1.0, 0.85, 0.65), core);
        return tint * (0.0025 + 0.06 * level * level * (1 + 2 * core) * (0.75 + 0.5 * fbm(g * 14.0, 4)));
    }

    // Colour of a black body at `k` kelvin, with unit luminance.
    inline float3 starTint(float k) {
        float3 c = blackbody(k);
        return c / max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.05);
    }

    // Limb darkening in visible light, I(mu) / I(1): the Sun's law (Neckel and Labs), stronger for cooler stars.
    inline float limbDarkening(float mu, float teff) {
        float k = clamp(5772.0 / teff, 0.4, 2.5);
        float u1 = clamp(0.47 * pow(k, 0.6), 0.15, 0.85), u2 = 0.23;
        float x = 1 - mu;
        return max(1 - u1 * x - u2 * x * x * k * 0.6, 0.05);
    }

    // Cellular noise on a sphere: distance to the nearest and second-nearest seed, the seeds drifting with time.
    inline float2 cells(float3 p, float time) {
        float3 i = floor(p), f = fract(p);
        float d1 = 8, d2 = 8;
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    float3 o = float3(x, y, z);
                    float3 h = hash33(i + o);
                    float3 seed = o + 0.5 + 0.38 * sin(time * (0.3 + 0.4 * h) + TAU * h);
                    float d = length(seed - f);
                    if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
                }
            }
        }
        return float2(d1, d2);
    }

    // Brightness of granulation at a point n on the unit sphere: bright cells parted by dark lanes, `count` cells
    // across a radius. Fades to its mean where cells shrink below a few pixels (`pixel` is a pixel in radii).
    inline float granulation(float3 n, float count, float time, float pixel) {
        float visible = smoothstep(4.0, 9.0, 1 / max(pixel * count, 1e-4));
        if (visible <= 0) return 1;
        float2 c = cells(n * count, time);
        float lane = smoothstep(0.0, 0.22, c.y - c.x);
        float g = 0.8 + 0.3 * lane - 0.12 * c.x;
        return mix(1.0, g / 0.98, visible);
    }

    """
}
