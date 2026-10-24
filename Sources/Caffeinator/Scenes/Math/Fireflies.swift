import Foundation
import Metal

/// Two thousand fireflies over a night meadow, each a phase oscillator flashing at its own natural rate. Kuramoto
/// coupling to their neighbours strengthens slowly until patches fall into step and the whole meadow pulses together,
/// the pulse rolling outward across the grass like the synchronous fireflies of the Great Smoky Mountains; then a
/// breeze scatters their rhythm and it begins again.
@MainActor final class Fireflies: Scene {
    private static let count = 2048
    /// Mean flash rate (radians per second: one flash every 2.1 s) and its spread.
    private static let omega: Float = 2 * .pi / 2.1
    private static let spread: Float = 0.16
    /// Critical coupling for a Gaussian spread of natural frequencies, 2 / (π g(0)).
    private static let critical: Float = 2 * spread * (2 * .pi).squareRoot() / .pi
    private static let cols = 16, rows = 10
    /// Screen height of the horizon, in centered coordinates, seen from a metre above the grass.
    private static let horizon: Float = -0.12
    /// Phase lag per metre from the wave's origin: about two radians across the meadow, so a locked pulse rolls
    /// outward over two thirds of a second.
    private static let waveLag: Float = 0.075

    private struct Fly {
        var phase: Float
        var omega: Float
        /// Ground position (metres across, metres away), height, and wandering phases.
        var x: Float, d: Float, h: Float
        var wx: Float, wy: Float, wz: Float
        var tint: Float
        var lag: Float = 0
    }

    private let scenery: MTLRenderPipelineState
    private let sky: MTLRenderPipelineState
    private let pools: MTLRenderPipelineState
    private let glows: MTLRenderPipelineState
    private let grass: MTLRenderPipelineState
    private let expose: MTLRenderPipelineState
    private let bloom: Bloom
    private var hdr: MTLTexture?
    private var backdrop: MTLTexture?
    private var light: MTLTexture?
    private var buffers: [MTLBuffer]
    private var flies: [Fly] = []
    private var rng: UInt64

    private var cycle: Float = 0
    private var coupling: Float = 0
    private var order: Float = 0
    private var synced: Float = 0
    private var gust: Float = -1
    private var drift: Float = 0
    private var origin = SIMD2<Float>(0, 12)
    private var tokens: Float = 0
    private var pulseCool: Float = 0
    private var shine: Float = 0
    /// Recent flash onsets as a share of the swarm, decaying over a quarter second, and where they were.
    private var burst: Float = 0
    private var burstX: Float = 0

    var scale: Float { 0.75 }

    var facts: String? {
        let r = String(format: "%.2f", order)
        let k = String(format: "%.1f", coupling / Self.critical)
        if gust >= 0 { return "\(Self.count.formatted()) fireflies; a breeze is sweeping the meadow and scattering their rhythm, synchrony \(r)" }
        let state = order > 0.8 ? "flashing together, the pulse rolling across the meadow" : order > 0.4 ? "falling into step" : "flashing on their own"
        return "\(Self.count.formatted()) fireflies \(state); synchrony \(r), coupling \(k) times the critical strength"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        scenery = try gpu.pipeline(library, "meadow")
        sky = try gpu.pipeline(library, "night")
        pools = try gpu.pipeline(library, "pool", vertex: "poolVertex", blend: .add)
        glows = try gpu.pipeline(library, "bokeh", vertex: "flyVertex", blend: .add)
        grass = try gpu.pipeline(library, "blades", blend: .alpha)
        expose = try gpu.pipeline(library, "nightExpose")
        bloom = try Bloom(gpu: gpu)
        buffers = (0..<3).map { _ in gpu.buffer(Self.count * MemoryLayout<SIMD4<Float>>.stride * 2) }
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
        for i in 0..<Self.count {
            // More fireflies far away, where there's more meadow, and a few hovering low right in front of the lens.
            let near = i % 20 == 0
            let d = near ? 1.2 + random() * 1.8 : 3.2 + pow(random(), 1.5) * 26
            let h = near ? 0.1 + random() * 0.45 : 0.12 + random() * random() * 2.2
            flies.append(Fly(phase: random() * 2 * .pi, omega: Self.omega + gaussian() * Self.spread,
                             x: (random() * 2 - 1) * d * 1.8, d: d, h: h,
                             wx: random() * 100, wy: random() * 100, wz: random() * 100, tint: random()))
        }
        cycle = Float(Snapshot.variant ?? 0) * 40
        newOrigin()
    }

    private func random() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(rng >> 40) / Float(1 << 24)
    }

    private func gaussian() -> Float {
        let u = max(random(), 1e-7), v = random()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }

    /// Somewhere in the meadow for the next cycle's pulse to roll out from.
    private func newOrigin() {
        origin = SIMD2((random() * 2 - 1) * 9, 5 + random() * 14)
    }

    func draw(_ f: Frame) {
        if hdr?.width != f.target.width || hdr?.height != f.target.height {
            hdr = f.gpu.texture(f.target.width, f.target.height)
            light = f.gpu.texture(max(f.target.width / 4, 1), max(f.target.height / 4, 1))
            // The sky, woods and meadow stand still, so they're drawn once.
            backdrop = f.gpu.texture(f.target.width, f.target.height)
            f.cmd.draw(scenery, into: backdrop!, uniforms: f.uniforms([f.aspect, 0, 0, f.seed]))
        }
        let dt = f.dt
        cycle += dt

        // The coupling climbs past the critical strength over two minutes; once the meadow has pulsed together for
        // a while, a breeze sweeps across, scattering phases and carrying the fireflies sideways, and the coupling
        // drops back.
        if gust < 0 {
            coupling = Self.critical * 2.6 * min(cycle / 130, 1) * min(cycle / 130, 1)
            synced = order > 0.85 ? synced + dt : max(synced - dt * 0.5, 0)
            if synced > 45 { gust = 0 }
        } else {
            gust += dt
            coupling *= exp(-dt / 2)
            if gust > 14 {
                gust = -1
                cycle = 0
                synced = 0
                newOrigin()
            }
        }
        let wind = gust >= 0 ? sin(min(gust / 14, 1) * .pi) : 0
        drift += wind * 1.4 * dt

        // Each firefly couples to the mean phase of its neighbourhood (a coarse grid, blurred) and a little of
        // everyone's. Coupling acts on the phase plus a lag that grows with distance from the wave's origin, so once
        // locked, the flashes start there and roll outward.
        let cols = Self.cols, rows = Self.rows
        var cell = [SIMD2<Float>](repeating: .zero, count: cols * rows)
        var counts = [Float](repeating: 0, count: cols * rows)
        var total = SIMD2<Float>.zero
        var cellOf = [Int](repeating: 0, count: Self.count)
        var spots = [SIMD2<Float>](repeating: .zero, count: Self.count)
        for i in 0..<Self.count {
            let (p, ground) = place(flies[i], f.time)
            spots[i] = p
            let g = ground - origin
            flies[i].lag = (g * g).sum().squareRoot() * Self.waveLag
            let cx = min(max(Int((p.x / f.aspect * 0.5 + 0.5) * Float(cols)), 0), cols - 1)
            let cy = min(max(Int((p.y * 0.5 + 0.5) * Float(rows)), 0), rows - 1)
            let k = cy * cols + cx
            cellOf[i] = k
            let a = flies[i].phase + flies[i].lag
            let e = SIMD2(cos(a), sin(a))
            cell[k] += e
            counts[k] += 1
            total += e
        }
        var field = [SIMD2<Float>](repeating: .zero, count: cols * rows)
        for y in 0..<rows {
            for x in 0..<cols {
                var s = SIMD2<Float>.zero, n: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let xx = x + dx, yy = y + dy
                        guard xx >= 0, xx < cols, yy >= 0, yy < rows else { continue }
                        let w: Float = dx == 0 && dy == 0 ? 2 : 1
                        s += cell[yy * cols + xx] * w
                        n += counts[yy * cols + xx] * w
                    }
                }
                field[y * cols + x] = s / max(n, 1)
            }
        }
        let global = total / Float(Self.count)
        order = (global * global).sum().squareRoot()

        // Kuramoto: dθ/dt = ω + K Im(Z e^(-i(θ + lag))). The breeze kicks phases at random as its front passes.
        let front = gust >= 0 ? gust / 9 * 2.6 - 1.3 : -9
        var onsets: [Int] = []
        var lit: Float = 0
        let buffer = buffers[Int(f.index % 3)].contents().assumingMemoryBound(to: SIMD4<Float>.self)
        for i in 0..<Self.count {
            var fly = flies[i]
            let z = field[cellOf[i]] * 0.8 + global * 0.2
            let a = fly.phase + fly.lag
            let pull = z.y * cos(a) - z.x * sin(a)
            var next = fly.phase + (fly.omega + coupling * pull) * dt + gaussian() * 0.02 * dt.squareRoot()
            let p = spots[i]
            if gust >= 0 {
                // The front is ragged: each firefly feels it a little early or late.
                let edge = abs(p.x / f.aspect - front + (fly.tint - 0.5) * 0.25)
                if edge < 0.15 { next += gaussian() * 3.3 * dt.squareRoot() * (1 - edge / 0.15) }
            }
            if next >= 2 * .pi {
                next -= 2 * .pi
                onsets.append(i)
            }
            fly.phase = next
            flies[i] = fly
            let flash = Self.flash(next)
            lit += flash
            buffer[2 * i] = SIMD4(p.x, p.y, fly.d, flash)
            buffer[2 * i + 1] = SIMD4(fly.tint, fly.h, 0, 0)
        }
        shine += (lit / Float(Self.count) - shine) * min(dt * 8, 1)

        let u = f.uniforms([f.aspect, wind, shine, f.seed], [front, Self.horizon, 0, 0])
        let flyBuffer = buffers[Int(f.index % 3)]
        f.cmd.draw(pools, into: light!, uniforms: u, buffers: [flyBuffer], vertices: 6, instances: Self.count)
        f.cmd.draw(sky, into: hdr!, uniforms: u, textures: [backdrop!, light!])
        f.cmd.draw(glows, into: hdr!, clear: false, uniforms: u, buffers: [flyBuffer], vertices: 6, instances: Self.count)
        f.cmd.draw(grass, into: hdr!, clear: false, uniforms: u, textures: [light!])
        let glow = bloom.glow(f.cmd, hdr!)
        f.cmd.draw(expose, into: f.target, uniforms: u, textures: [hdr!, glow])

        if let synth = f.sound { play(synth, f, onsets) }
    }

    /// A flash's brightness at a phase: a soft rise over about 0.3 s up to the moment the phase wraps, a fade over
    /// about 0.6 s after it, and dark in between, so a shared pulse stands out against the night.
    static func flash(_ phase: Float) -> Float {
        let before = 2 * .pi - phase
        if before < 0.9 {
            let t = 1 - before / 0.9
            return t * t * (3 - 2 * t)
        }
        if phase < 1.8 {
            let t = 1 - phase / 1.8
            return t * t
        }
        return 0
    }

    /// Screen position (centered coordinates) of a firefly from a camera a metre above the grass, with its slow
    /// wander and the breeze's push, and the ground point beneath it in metres.
    private func place(_ fly: Fly, _ t: Float) -> (SIMD2<Float>, SIMD2<Float>) {
        let d = fly.d + sin(t * 0.04 + fly.wy) * 0.6
        let span = fly.d * 3.6
        var x = fly.x + sin(t * 0.05 + fly.wx) * 1.2 + sin(t * 0.13 + fly.wy) * 0.3 + drift
        x -= span * ((x + span / 2) / span).rounded(.down)
        let h = fly.h + sin(t * 0.09 + fly.wz) * 0.2 * min(fly.h, 1) + 0.06 * sin(t * 0.4 + fly.wx)
        return (SIMD2(x / d, Self.horizon + (h - 1) / d), SIMD2(x, d))
    }

    /// Each flash can chirp: scattered, a few quiet plucks a second from random fireflies; in step, one soft chord
    /// per shared pulse as it rolls past, louder the more fireflies join it. A pad swells with the synchrony;
    /// crickets and wind beneath.
    private func play(_ synth: Synth, _ f: Frame, _ onsets: [Int]) {
        let scale: [Float] = [1, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3, 2, 9.0 / 4, 5.0 / 2, 3, 10.0 / 3]
        let root: Float = 196
        tokens = min(tokens + f.dt * 3.5, 3)
        pulseCool -= f.dt
        // A rolling pulse spreads its onsets over half a second, so they're gathered over a quarter-second decay.
        let share = Float(onsets.count) / Float(Self.count)
        burst = burst * exp(-f.dt / 0.25) + share
        if !onsets.isEmpty {
            let x = onsets.reduce(0) { $0 + flies[$1].x / flies[$1].d } / Float(onsets.count)
            burstX += (x - burstX) * min(share * 20, 1)
        }
        let window = burst
        if window > 0.3, pulseCool < 0 {
            let amp = 0.03 + 0.08 * min(window, 1)
            let pan = max(-0.6, min(0.6, burstX))
            synth.play(.pluck, root * 2, amp: amp, pan: pan - 0.2, length: 1.6)
            synth.play(.pluck, root * 3, amp: amp * 0.7, pan: pan + 0.2, length: 1.6)
            if window > 0.6 { synth.play(.bell, root * 4, amp: amp * 0.35, pan: pan, length: 2.5) }
            pulseCool = 1.2
        } else if let i = onsets.randomElement(), tokens >= 1, window < 0.3 {
            tokens -= 1
            let fly = flies[i]
            let note = scale[min(Int(fly.h / 2.9 * Float(scale.count)), scale.count - 1)]
            let near = min(6 / fly.d, 1)
            synth.play(.pluck, root * 2 * note, amp: 0.012 + 0.025 * near, pan: max(-0.9, min(0.9, fly.x / fly.d)), length: 0.8)
        }
        let wind = gust >= 0 ? sin(min(gust / 14, 1) * .pi) : 0
        synth.drone(0, root / 2, 0.05 + 0.05 * order, pan: -0.2)
        synth.drone(1, root * 0.75, 0.04 * order, pan: 0.2)
        synth.drone(2, root, 0.03 * order * (0.6 + 0.6 * min(shine * 3, 1)))
        synth.noise(0.02 + 0.06 * wind, cutoff: 260 + 900 * wind, crackle: 7, crackleTone: 4600)
    }

    private static let source = """
    // a: aspect, wind (0...1), mean flash brightness, seed   b: breeze front (x over aspect), horizon
    constant float HORIZON = -0.12;
    constant float3 GLOW = float3(0.80, 1.00, 0.28);

    // The tree line's height at x: a lumpy broadleaf canopy, broken by conifers of many heights and widths, each
    // a spire of drooping, slightly ragged tiers.
    inline float treeline(float x, float seed) {
        float top = HORIZON + 0.035 + 0.05 * fbm(float2(x * 1.1 + seed, 0.3), 4);
        // Crowns: one rounded lump per uneven cell.
        float cw = 0.07;
        float id = floor(x / cw);
        for (int k = -1; k <= 1; k++) {
            float c = id + float(k);
            float cx = (c + 0.2 + 0.6 * hash11(c * 1.73 + seed)) * cw;
            float rw = cw * (0.7 + 0.8 * hash11(c * 3.1 + seed));
            float rh = 0.02 + 0.05 * hash11(c * 5.7 + seed);
            float u = (x - cx) / rw;
            if (abs(u) < 1) top = max(top, HORIZON + 0.045 + 0.04 * fbm(float2(cx * 1.1 + seed, 0.3), 4) + rh * sqrt(1 - u * u));
        }
        top += 0.012 * (fbm(float2(x * 45, 1.7), 3) - 0.5);
        // Conifers in unevenly spaced slots, some empty.
        float sw = 0.11;
        float sid = floor(x / sw);
        for (int k = -1; k <= 1; k++) {
            float c = sid + float(k);
            if (hash11(c * 9.13 + seed) < 0.45) continue;
            float cx = (c + 0.15 + 0.7 * hash11(c * 2.37 + seed)) * sw;
            float tall = 0.07 + 0.24 * pow(hash11(c * 4.91 + seed), 1.5);
            float wide = tall * (0.26 + 0.2 * hash11(c * 6.2 + seed));
            float lean = (hash11(c * 8.8 + seed) - 0.5) * 0.25;
            float u = (x - cx) / wide;
            if (abs(u) < 1) {
                float base = HORIZON + 0.04;
                float h = 1 - abs(u + lean * (1 - abs(u))) ;
                float tiers = 5 + 4 * hash11(c * 3.3 + seed);
                float notch = fract(abs(u) * tiers + 0.3 * noise(float2(x * 300, c)));
                h = h * 0.9 - 0.12 * notch * (1 - abs(u) * 0.5) + 0.03 * (noise(float2(x * 400, c)) - 0.5);
                top = max(top, base + tall * pow(max(h, 0.0), 1.15));
            }
        }
        return top;
    }

    fragment float4 meadow(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 p = centered(in.uv, u.res);
        float seed = u.a.w * 0.01;
        // Sky: deep blue fading to a faint glow at the horizon, with a scatter of dim stars.
        float3 c = mix(float3(0.012, 0.018, 0.045), float3(0.0015, 0.002, 0.007), smoothstep(HORIZON, 1.0, p.y));
        c += float3(0.018, 0.016, 0.024) * exp(-max(p.y - HORIZON, 0.0) * 5);
        float2 sp = p * 90;
        float2 sc = floor(sp);
        float star = step(0.985, hash12(sc + 3.1)) * smoothstep(0.25, 0.0, length(fract(sp) - hash22(sc) * 0.6 - 0.2));
        c += float3(0.6, 0.65, 0.8) * star * hash12(sc) * 0.3 * smoothstep(0.1, 0.5, p.y);
        // The meadow: dark grass, a little lighter far away where it catches the sky. Alpha keeps how much
        // firefly light the grass returns, mottled with clumps.
        float albedo = 0;
        if (p.y < HORIZON + 0.01) {
            float far = smoothstep(-0.7, HORIZON, p.y);
            float3 g = mix(float3(0.0015, 0.0028, 0.0020), float3(0.005, 0.008, 0.009), far * far);
            float clumps = fbm(float2(p.x * 3, p.y * 14) / max(HORIZON + 0.02 - p.y, 0.05), 4);
            g *= 0.7 + 0.6 * clumps;
            float m = smoothstep(HORIZON + 0.01, HORIZON - 0.005, p.y);
            c = mix(c, g, m);
            albedo = m * (0.5 + 0.7 * clumps);
        }
        float trees = smoothstep(0.002, -0.002, p.y - treeline(p.x, seed));
        c = mix(c, float3(0.0008, 0.0012, 0.0016), trees);
        return float4(c, albedo * (1 - trees));
    }

    // Pools of light each firefly throws on the grass beneath it, foreshortened on the ground, into a quarter-size
    // texture. Fireflies higher up light the grass more faintly.
    struct L { float4 pos [[position]]; float2 q; float3 color; };

    vertex L poolVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                        const device float4 *flies [[buffer(1)]]) {
        float4 f = flies[2 * iid];
        float h = flies[2 * iid + 1].y;
        float d = f.z;
        float reach = 0.75;
        float2 c = float2(f.x, HORIZON - 1 / d);
        float2 r = float2(reach / d, max(reach / (d * d), 12 / u.res.y));
        const float2 corners[6] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1)};
        float2 q = corners[vid];
        float2 pos = c + q * r * float(f.w > 0.002);
        L o;
        o.pos = float4(pos.x / u.a.x, pos.y, 0, 1);
        o.q = q;
        o.color = GLOW * f.w * 0.045 / (h * h + 0.25);
        return o;
    }

    fragment float4 pool(L in [[stage_in]]) {
        float r2 = dot(in.q, in.q);
        return float4(in.color * exp(-r2 * 3.5) * step(r2, 1.0), 0);
    }

    // Each frame: the still backdrop, firefly light on the grass and in the low mist, and the mist itself.
    fragment float4 night(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> backdrop [[texture(0)]],
                          texture2d<float> light [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float4 b = backdrop.sample(s, in.uv);
        float3 l = light.sample(s, in.uv).rgb;
        float3 c = b.rgb + l * float3(0.55, 0.75, 0.35) * b.a;
        float mist = exp(-pow((p.y - HORIZON + 0.02) / 0.07, 2.0)) * (0.7 + 0.3 * noise(float2(p.x * 2 + u.time * 0.02, 3.0)));
        c += float3(0.005, 0.007, 0.010) * mist;
        // Light scattered in the air over the grass: a soft glow just above where the fireflies are.
        float2 up = in.uv + float2(0, 0.025);
        c += light.sample(s, up).rgb * 0.12 * smoothstep(HORIZON + 0.2, HORIZON - 0.1, p.y);
        return float4(c, 1);
    }

    // Each firefly: far away a bright point with a soft halo, close to the lens a wide, faint bokeh disc.
    struct G { float4 pos [[position]]; float2 q; float4 color; float radius; float soft; };

    vertex G flyVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                       const device float4 *flies [[buffer(1)]]) {
        float4 f = flies[2 * iid];
        float tint = flies[2 * iid + 1].x;
        float d = f.z;
        float px = u.res.y * 0.5;
        float blur = max(1 / d - 1 / 4.5, 0.0);
        float bokeh = blur * 0.3 * px;
        // Dark fireflies collapse to nothing, so they cost no pixels.
        float radius = f.w > 0.002 ? max(bokeh, 13.0) : 0.0;
        const float2 corners[6] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1)};
        float2 q = corners[vid];
        float2 pos = f.xy + q * radius / px;
        G o;
        o.pos = float4(pos.x / u.a.x, pos.y, 0, 1);
        o.q = q;
        float3 warm = mix(float3(0.70, 1.0, 0.20), float3(1.0, 0.78, 0.20), tint * tint);
        o.color = float4(warm * f.w * clamp(10 / d, 0.6, 1.6), 0);
        o.radius = radius;
        o.soft = smoothstep(4.0, 14.0, bokeh);
        return o;
    }

    fragment float4 bokeh(G in [[stage_in]]) {
        float r = length(in.q) * in.radius;
        float point = 2.4 * exp(-r * r / 2.2) + 0.10 * exp(-r * r / 40) + 0.025 * exp(-r * r / 90);
        float edge = length(in.q);
        float disc = smoothstep(1.0, 0.7, edge) * (0.75 + 0.25 * smoothstep(0.5, 0.9, edge)) * 40 / (in.radius * in.radius);
        return in.color * mix(point, disc, in.soft);
    }

    // Foreground grass and tall seed stalks: dark silhouettes against the glowing meadow, faintly lit by flashes
    // around them, leaning with the breeze.
    fragment float4 blades(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> light [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float a = 0, lit = 0;
        for (int layer = 0; layer < 4; layer++) {
            float fl = float(layer);
            float density = 50 + fl * 38;
            float base = -1.02;
            float x = p.x * density;
            float id = floor(x);
            float tall = (0.18 + 0.52 * pow(hash11(id * 1.37 + fl * 11), 1.4)) * (1.25 - fl * 0.2);
            float h = p.y - base;
            float t = saturate(h / tall);
            float sway = (sin(u.time * (0.5 + hash11(id) * 0.4) + id * 1.7) * 0.18 + u.a.y * 2.2) * t * t;
            float centre = id + 0.5 + (hash11(id + 3.3) - 0.5) * 0.6 + sway * density * 0.1;
            float width = 0.38 * (1 - t * 0.92);
            float off = x - centre;
            float blade = smoothstep(width, width - 0.1, abs(off)) * step(h, tall);
            if (blade > a) {
                a = blade;
                lit = 0.6 + 0.4 * saturate(off / max(width, 0.05) + 0.5);
            }
        }
        // Tall stalks with seed heads.
        float x = p.x * 9;
        float id = floor(x);
        if (hash11(id * 4.1) > 0.35) {
            float tall = 0.55 + 0.4 * hash11(id * 2.3);
            float h = p.y + 1.02;
            float t = saturate(h / tall);
            float sway = (sin(u.time * 0.35 + id) * 0.08 + u.a.y * 1.2) * t * t;
            float cx = id + 0.5 + (hash11(id + 9.1) - 0.5) * 0.7 + sway * 9;
            float stem = smoothstep(0.05, 0.03, abs(x - cx)) * step(h, tall);
            float head = smoothstep(1.0, 0.8, length(float2((x - cx) / 0.14, (h - tall) / 0.05)));
            float stalk = max(stem, head);
            if (stalk > a) { a = stalk; lit = 0.8; }
        }
        float3 l = light.sample(s, in.uv).rgb + light.sample(s, in.uv + float2(0, 0.03)).rgb;
        float3 c = float3(0.0010, 0.0016, 0.0022) * lit + l * float3(0.30, 0.42, 0.16) * 0.22 * lit;
        return float4(c * a, a);
    }

    fragment float4 nightExpose(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> hdr [[texture(0)]],
                                texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = hdr.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.7;
        return float4(film(c), 1);
    }
    """
}
