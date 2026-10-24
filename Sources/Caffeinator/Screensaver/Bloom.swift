import Metal

/// Dual-filter (Kawase) bloom: a cheap, wide glow around the bright parts of an HDR image.
/// Scenes add the returned half-resolution texture back in their final pass.
@MainActor final class Bloom {
    private let gpu: GPU
    private let down: MTLRenderPipelineState
    private let up: MTLRenderPipelineState
    private var chain: [MTLTexture] = []
    private var ups: [MTLTexture] = []

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(Self.source)
        down = try gpu.pipeline(library, "bloomDown")
        up = try gpu.pipeline(library, "bloomUp")
    }

    /// Blurs `source` over `levels` octaves and returns the glow at half its resolution.
    func glow(_ cmd: MTLCommandBuffer, _ source: MTLTexture, levels: Int = 5) -> MTLTexture {
        if chain.first?.width != max(source.width / 2, 1) || chain.count != levels {
            chain = (1...levels).map { gpu.texture(max(source.width >> $0, 1), max(source.height >> $0, 1)) }
            ups = (1..<levels).map { gpu.texture(max(source.width >> $0, 1), max(source.height >> $0, 1)) }
        }
        var input = source
        for t in chain {
            cmd.draw(down, into: t, uniforms: U(res: SIMD2(Float(input.width), Float(input.height))), textures: [input])
            input = t
        }
        // Walk back up, adding each level so the glow has both a tight core and a wide halo.
        var small = chain[levels - 1]
        for i in stride(from: levels - 2, through: 0, by: -1) {
            cmd.draw(up, into: ups[i], uniforms: U(res: SIMD2(Float(small.width), Float(small.height)), a: [1 / Float(levels), 0, 0, 0]),
                     textures: [small, chain[i]])
            small = ups[i]
        }
        return small
    }

    private static let source = """
    fragment float4 bloomDown(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> t [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 h = 1 / u.res;
        float4 c = t.sample(s, in.uv) * 4;
        c += t.sample(s, in.uv + h * float2(-1, -1)) + t.sample(s, in.uv + h * float2(1, -1));
        c += t.sample(s, in.uv + h * float2(-1, 1)) + t.sample(s, in.uv + h * float2(1, 1));
        return min(c / 8, 64.0);
    }

    fragment float4 bloomUp(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> t [[texture(0)]],
                            texture2d<float> same [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 h = 1 / u.res;
        float4 c = 0;
        c += t.sample(s, in.uv + h * float2(-2, 0)) + t.sample(s, in.uv + h * float2(2, 0));
        c += t.sample(s, in.uv + h * float2(0, -2)) + t.sample(s, in.uv + h * float2(0, 2));
        c += 2 * (t.sample(s, in.uv + h * float2(-1, -1)) + t.sample(s, in.uv + h * float2(1, -1)));
        c += 2 * (t.sample(s, in.uv + h * float2(-1, 1)) + t.sample(s, in.uv + h * float2(1, 1)));
        return c / 12 + same.sample(s, in.uv) * u.a.x;
    }
    """
}
