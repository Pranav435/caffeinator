import Metal

@MainActor extension SceneKind {
    static let fieldScenes: [SceneKind] = [
        SceneKind(id: "black-hole", name: "Black Hole", family: .physics,
                  about: "A Schwarzschild black hole seen from about thirty radii out, ray traced against the real Milky Way photograph. A black shadow sits inside a thin, glowing disk of gas that tilts toward you: one side is blazing white where it rushes at you, the other dim and ochre, and the stars and dust lanes behind are bent into arcs and a faint ring around the hole.",
                  make: BlackHole.init, gain: 0.65),
        SceneKind(id: "ink", name: "Ink in Water", family: .physics,
                  about: "A glass tank of still water, seen from the side and lit from behind by a dim, warm softbox. Drops of Prussian blue, vermilion and gold ink fall in, sink as ragged plumes, roll into mushroom-shaped vortex rings and fold into marbled veils, while mixed inks darken or turn green. Tiny bubbles cling to the glass and dust drifts in the light.",
                  make: InkInWater.init, gain: 0.6),
        SceneKind(id: "slime", name: "Slime Mold", family: .physics,
                  about: "A macro photograph, from above, of the slime mold Physarum polycephalum growing on pale agar in a petri dish: a bright yellow network of glossy tubes that swell and slacken as cytoplasm shuttles through them. Now and then oat flakes appear at the positions of 36 cities around Tokyo, and the mold rebuilds something like the Tokyo rail network between them.",
                  make: SlimeMold.init, gain: 0.8),
        SceneKind(id: "aurora", name: "Northern Lights", family: .physics,
                  about: "A long-exposure photograph of the northern or southern lights over a still fjord or lake ringed by snowy mountains, under the real Milky Way and stars for this place and time. Rayed green curtains with red crowns and pink lower edges fold across the sky and mirror in the water, as lively as the real geomagnetic Kp index right now.",
                  make: AuroraCurtains.init, gain: 0.9),
    ]
}

/// Several compute dispatches in one encoder, in order, with the same bindings as `MTLCommandBuffer.compute`.
/// Forty Jacobi iterations a frame would otherwise mean forty encoders.
struct FieldDispatch {
    let encoder: MTLComputeCommandEncoder

    func callAsFunction(_ p: MTLComputePipelineState, _ w: Int, _ h: Int, _ u: U, _ textures: [MTLTexture], buffers: [MTLBuffer] = []) {
        encoder.setComputePipelineState(p)
        var u = u
        encoder.setBytes(&u, length: MemoryLayout<U>.stride, index: 0)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i + 1) }
        for (i, t) in textures.enumerated() { encoder.setTexture(t, index: i) }
        let tw = p.threadExecutionWidth
        let group = h == 1 ? MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
                           : MTLSize(width: tw, height: max(p.maxTotalThreadsPerThreadgroup / tw / 2, 1), depth: 1)
        encoder.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: group)
    }
}

/// Helpers shared by the scenes in this folder.
enum FieldKit {
    /// Point i of the Halton sequence in `base`, in 0..<1: well-spread sub-pixel jitter for temporal antialiasing.
    static func halton(_ i: Int, _ base: Int) -> Float {
        var f: Float = 1, r: Float = 0, i = i
        while i > 0 { f /= Float(base); r += f * Float(i % base); i /= base }
        return r
    }

    /// Metal for a star field, prepended to the scenes that draw one.
    static let stars = """
    // Stars on a cube-mapped grid, at most one per cell, each a Gaussian at least the pixel's footprint wide.
    // Surface brightness is conserved, so a star squeezed into a smaller patch of sky than one pixel gets dimmer;
    // `ref` is the pixel footprint, in radians, at which a star of `gain` peaks at that brightness.
    inline float3 starLayer(float3 d, float cells, float chance, float fp, float gain, float salt, float ref) {
        float3 a = abs(d);
        float2 uv; float face;
        if (a.x >= a.y && a.x >= a.z) { uv = d.yz / a.x; face = d.x > 0 ? 0 : 1; }
        else if (a.y >= a.z) { uv = d.zx / a.y; face = d.y > 0 ? 2 : 3; }
        else { uv = d.xy / a.z; face = d.z > 0 ? 4 : 5; }
        float2 g = (uv * 0.5 + 0.5) * cells;
        float2 c0 = floor(g);
        float s0 = 0.0006;
        float sigma2 = s0 * s0 + fp * fp;
        float w = sqrt(sigma2) * cells * 0.5;
        float keep = smoothstep(0.7, 0.35, w);
        if (keep <= 0) return 0;
        float3 acc = 0;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                float2 c = c0 + float2(i, j);
                float3 h = hash33(float3(c + face * 157.31, salt));
                if (h.x > chance) continue;
                float2 pos = c + 0.2 + 0.6 * h.yz;
                float2 dd = g - pos;
                float e = dot(dd, dd) / (2 * w * w);
                if (e > 9) continue;
                float3 k = hash33(float3(c * 1.31 + face * 11.7, salt + 7.3));
                float b = gain * pow(k.x, 9.0);
                float3 tint = blackbody(mix(3000.0, 12000.0, k.y * k.y));
                acc += tint * b * exp(-e);
            }
        }
        return acc * keep * (s0 * s0 + ref * ref) / sigma2;
    }
    """
}
