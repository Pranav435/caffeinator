import AppKit
import Metal

/// Text drawn once into a texture, so labels that move every frame cost one quad each on the GPU.
/// Sizes are in points; the atlas is rendered at the scene's pixel density so the text stays crisp.
@MainActor final class Labels {
    struct Style {
        var size: CGFloat
        var weight: NSFont.Weight = .regular
        var tracking: CGFloat = 0
    }

    let texture: MTLTexture
    let size: SIMD2<Float>
    private var rects: [String: SIMD4<Float>] = [:]

    init(gpu: GPU, density: CGFloat, _ items: [(String, Style)]) {
        let width = 2048
        var placed: [(NSAttributedString, CGPoint)] = []
        var x: CGFloat = 1, y: CGFloat = 1, row: CGFloat = 0
        var seen = Set<String>()
        var boxes: [(String, CGRect)] = []
        for (text, style) in items where !text.isEmpty && seen.insert(text).inserted {
            let font = NSFont.systemFont(ofSize: style.size * density, weight: style.weight)
            let s = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: NSColor.white,
                                                                  .kern: style.tracking * density])
            let b = s.size()
            let w = ceil(b.width) + 4, h = ceil(b.height) + 2
            if x + w > CGFloat(width) { x = 1; y += row + 2; row = 0 }
            placed.append((s, CGPoint(x: x + 2, y: y + 1)))
            boxes.append((text, CGRect(x: x, y: y, width: w, height: h)))
            x += w + 2
            row = max(row, h)
        }
        let height = max(Int(ceil(y + row + 2)), 4)
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
        for (s, p) in placed { s.draw(at: p) }
        NSGraphicsContext.restoreGraphicsState()
        texture = gpu.texture(width, height, .r8Unorm, bytes: ctx.data!, bytesPerRow: width)
        size = SIMD2(Float(width), Float(height))
        for (text, r) in boxes { rects[text] = SIMD4(Float(r.minX), Float(r.minY), Float(r.width), Float(r.height)) }
    }

    /// A quad showing `text` with its `anchor` point (0...1 across the text box) at `pos` in target pixels.
    func quad(_ text: String, at pos: SIMD2<Float>, anchor: SIMD2<Float> = [0.5, 0.5], color: SIMD4<Float>) -> LabelQuad? {
        guard let r = rects[text] else { return nil }
        return LabelQuad(pos: pos, anchor: anchor, color: color, rect: r)
    }

    func width(_ text: String) -> Float { rects[text]?.z ?? 0 }
}

/// One label on screen: matches `LabelQuad` in `SpaceKit.metal`. Colour is premultiplied.
struct LabelQuad {
    var pos: SIMD2<Float>
    var anchor: SIMD2<Float>
    var color: SIMD4<Float>
    var rect: SIMD4<Float>
}

/// Per-frame data the CPU writes and the GPU reads, rotated so a frame never overwrites one still in flight.
@MainActor final class Ring {
    private var buffers: [MTLBuffer]
    private var index = 0
    private let gpu: GPU

    init(gpu: GPU, bytes: Int) {
        self.gpu = gpu
        buffers = (0..<4).map { _ in gpu.buffer(bytes) }
    }

    func next<T>(_ items: [T]) -> MTLBuffer {
        index = (index + 1) % buffers.count
        let bytes = MemoryLayout<T>.stride * items.count
        if buffers[index].length < bytes { buffers[index] = gpu.buffer(bytes * 2) }
        items.withUnsafeBytes { raw in
            if let base = raw.baseAddress { buffers[index].contents().copyMemory(from: base, byteCount: bytes) }
        }
        return buffers[index]
    }

    func next<T>(_ item: T) -> MTLBuffer { next([item]) }
}

enum SpaceKit {
    /// Metal shared by the space scenes: label quads, antialiased line quads and a soft glow.
    static let metal = """
    struct LabelQuad { float2 pos; float2 anchor; float4 color; float4 rect; };
    struct LabelOut { float4 pos [[position]]; float2 uv; float4 color; };

    // a.xy: atlas size in pixels.
    vertex LabelOut labelVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                                const device LabelQuad *ls [[buffer(1)]]) {
        LabelQuad l = ls[iid];
        float2 corner = float2(vid & 1, vid >> 1);
        float2 p = l.pos - l.anchor * l.rect.zw + corner * l.rect.zw;   // unsnapped, so drifting labels glide
        LabelOut o;
        o.pos = float4(p.x / u.res.x * 2 - 1, 1 - p.y / u.res.y * 2, 0, 1);
        o.uv = (l.rect.xy + corner * l.rect.zw) / u.a.xy;
        o.color = l.color;
        return o;
    }

    fragment float4 labelFragment(LabelOut in [[stage_in]], texture2d<float> atlas [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        return in.color * atlas.sample(s, in.uv).r;
    }

    struct LineOut { float4 pos [[position]]; float4 color; float across; float halfw; };

    // One corner (0...3, as a triangle strip) of a quad around the segment a-b in pixels (top-left origin),
    // `width` pixels wide plus a pixel of soft edge. Lines thinner than a pixel draw a pixel wide but dimmer.
    inline LineOut lineQuad(float2 a, float2 b, float width, float4 color, uint corner, float2 res) {
        float2 d = b - a;
        float len = length(d);
        float2 t = len > 1e-4 ? d / len : float2(1, 0);
        float2 n = float2(-t.y, t.x);
        float w = max(width, 1.0);
        float h = w * 0.5 + 1;
        float side = (corner & 2) ? 1.0 : -1.0;
        float2 p = ((corner & 1) ? b : a) + n * side * h;
        LineOut o;
        o.pos = float4(p.x / res.x * 2 - 1, 1 - p.y / res.y * 2, 0, 1);
        o.color = color * (width / w);
        o.across = side * h;
        o.halfw = w * 0.5;
        return o;
    }

    fragment float4 lineFragment(LineOut in [[stage_in]]) {
        return in.color * clamp(in.halfw + 0.5 - abs(in.across), 0.0, 1.0);
    }

    // A soft round glow on a quad: the vertex shader passes uv in -1...1.
    struct GlowOut { float4 pos [[position]]; float2 uv; float4 color; float4 extra; };
    fragment float4 glowFragment(GlowOut in [[stage_in]]) {
        float r2 = dot(in.uv, in.uv);
        float core = exp(-r2 * in.extra.x);
        float halo = exp(-sqrt(r2) * in.extra.y) * in.extra.z;
        return in.color * (core + halo) * smoothstep(1.0, 0.8, r2);
    }

    inline float4 pixelToClip(float2 p, float2 res) { return float4(p.x / res.x * 2 - 1, 1 - p.y / res.y * 2, 0, 1); }

    """

    /// A number the way VoiceOver reads it well: grouped thousands, a fixed number of decimals.
    static func number(_ x: Double, decimals: Int = 0) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.minimumFractionDigits = decimals
        f.maximumFractionDigits = decimals
        return f.string(from: NSNumber(value: x)) ?? String(x)
    }
}

extension SIMD4 where Scalar == Float {
    /// A premultiplied colour with alpha.
    static func premultiplied(_ r: Float, _ g: Float, _ b: Float, _ a: Float) -> SIMD4<Float> { SIMD4(r * a, g * a, b * a, a) }
}
