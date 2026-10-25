import Metal

/// The camera the realistic scenes are filmed with. A scene renders linear HDR light, where 1 is a well-exposed white,
/// and `develop` finishes it like a photograph: exposure and white balance, bloom and warm film halation around
/// highlights, natural vignetting, a touch of lateral colour fringing toward the corners, the ACES curve, and fine grain
/// that moves every frame. Sharing one camera keeps every scene looking like real footage.
@MainActor final class Lens {
    struct Look {
        var exposure: Float = 1
        /// White balance, multiplied into the light.
        var white: SIMD3<Float> = [1, 1, 1]
        /// Share of the light scattered into a wide glow.
        var bloom: Float = 0.04
        /// Warm glow that film shows around bright highlights.
        var halation: Float = 0.02
        /// Darkening at the corners, 0 to 1.
        var vignette: Float = 0.35
        /// Red and blue fringes at the corners, in output pixels.
        var aberration: Float = 1.2
        /// Film grain; 1 is clearly visible on mid grey, 0 is none.
        var grain: Float = 0.5
    }

    private let bloom: Bloom
    private let pass: MTLRenderPipelineState

    init(gpu: GPU) throws {
        bloom = try Bloom(gpu: gpu)
        pass = try gpu.pipeline(gpu.library(Self.source), "lensDevelop")
    }

    /// Develops `light` (linear HDR, any size) into the frame's target.
    func develop(_ f: Frame, _ light: MTLTexture, _ look: Look = Look()) {
        let glow = look.bloom > 0 || look.halation > 0 ? bloom.glow(f.cmd, light) : light
        f.cmd.draw(pass, into: f.target, uniforms: f.uniforms([look.exposure, look.bloom, look.vignette, look.aberration],
                                                              [look.halation, look.grain, 0, 0], SIMD4(look.white, 0)),
                   textures: [light, glow])
    }

    private static let source = """
    fragment float4 lensDevelop(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]],
                                texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 c = in.uv - 0.5;
        float corner = 0.5 * length(u.res);
        float rho = length(c * u.res) / corner;
        float2 shift = c * u.a.w * rho / corner;
        float3 col = float3(light.sample(s, in.uv + shift).r, light.sample(s, in.uv).g, light.sample(s, in.uv - shift).b);
        float3 g = glow.sample(s, in.uv).rgb;
        col = col * (1 - u.a.y) + g * u.a.y + g * u.b.x * float3(1.0, 0.32, 0.1);
        col *= u.a.x * u.c.xyz * pow(max(1 - u.a.z * rho * rho, 0.0), 1.5);
        float3 out = film(max(col, 0.0));
        // Grain is strongest in the mid-tones and vanishes in true black, like film.
        float2 p = in.pos.xy + float2(u.frame % 64u, (u.frame / 64u) % 64u) * 37.0;
        float n = hash12(p) + hash12(p * 1.31 + 7.7) - 1;
        float y = dot(out, float3(0.2126, 0.7152, 0.0722));
        out += n * u.b.y * 0.06 * sqrt(y) * (1.1 - y);
        return float4(max(out, 0.0), 1);
    }
    """
}
