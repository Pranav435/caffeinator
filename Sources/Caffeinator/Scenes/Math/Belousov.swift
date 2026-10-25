import Foundation
import Metal

/// The Belousov–Zhabotinsky reaction in a glass Petri dish, photographed from above on a lab bench. The medium is
/// Barkley's model of an excitable medium, the standard fast model of BZ waves: a fast activator (bromous acid) and a
/// slow recovery variable, read as the share of the iron catalyst oxidised from red ferroin to blue ferriin. Pacemaker
/// spots send out target rings, broken fronts curl into spirals, and the dish is lit through its liquid by the white
/// paper beneath, colour by Beer–Lambert absorption.
@MainActor final class Belousov: Scene {
    /// Pacemakers (negative threshold offset `b`) and CO2 bubbles (obstacles), in grid cells.
    struct Feature { var x: Float; var y: Float; var r: Float; var b: Float }

    private static let grid = 512
    /// Barkley's parameters: a, b, ε, and the grid spacing and time step in the model's units.
    private static let a: Float = 0.75
    private static let b: Float = 0.06
    private static let epsilon: Float = 0.02
    private static let dx: Float = 0.4
    private static let dt: Float = 0.01
    /// Model time units per second of film. One unit is about ten seconds of the real reaction, so the film runs at
    /// 2.5 times real speed.
    private static let pace: Float = 0.25
    /// Millimetres per model length unit: 512 cells of 0.4 span the dish's 86 mm inside.
    private static let millimetres: Float = 86 / (512 * 0.4)
    /// A fresh dish every twelve minutes, as a real one tires and fills with bubbles.
    private static let dishSeconds: Float = 720

    private let step: MTLComputePipelineState
    private let paint: MTLComputePipelineState
    private let breaker: MTLComputePipelineState
    private let probe: MTLComputePipelineState
    private let photo: MTLRenderPipelineState
    private let lens: Lens
    private var states: [MTLTexture] = []
    private var medium: MTLTexture?
    private var light: MTLTexture?
    private let features: MTLBuffer
    private let probes: MTLBuffer
    private let readback: MTLBuffer
    private var current = 0
    private var budget: Float = 0

    private var rng: UInt64
    private var pacemakers: [Feature] = []
    private var bubbles: [Feature] = []
    private var age: Float = 0
    private var level: Float = 0
    private var fresh = true
    private var nextBreak: Float = 0
    private var sinceCount: Float = 0
    private var sincePaint: Float = 0
    private var reading = false
    private var spirals = 0
    private var oxidised: Float = 0
    private var framing = SIMD4<Float>(0, 0, 0, 0)
    private var wasHigh = [Bool](repeating: false, count: 8)
    private var noteCool: Float = 0

    var scale: Float { 0.75 }

    var facts: String? {
        // Ten real seconds per model time unit, after an induction period of about three minutes.
        let minutes = 3 + Int(age * Self.pace * 10 / 60)
        let rings = pacemakers.count == 1 ? "1 pacemaker sends out target rings" : "\(pacemakers.count) pacemakers send out target rings"
        let turning = spirals == 0 ? "" : spirals == 1 ? ", 1 spiral wave turns" : ", \(spirals) spiral waves turn"
        let speed = Int((2.7 * Self.millimetres / 10 * 60).rounded())
        return "Belousov–Zhabotinsky reaction in a 9-centimetre dish, \(minutes) minutes after mixing: \(rings)\(turning), blue ferriin through red ferroin, about \(speed) millimetres a minute. Barkley's excitable-medium model, filmed at 2.5 times speed"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        step = try gpu.kernel(library, "bzStep")
        paint = try gpu.kernel(library, "bzMedium")
        breaker = try gpu.kernel(library, "bzBreak")
        probe = try gpu.kernel(library, "bzProbe")
        photo = try gpu.pipeline(library, "bzPhoto")
        lens = try Lens(gpu: gpu)
        features = gpu.buffer(32 * MemoryLayout<SIMD4<Float>>.stride)
        probes = gpu.buffer(8 * MemoryLayout<SIMD2<Float>>.stride)
        readback = gpu.buffer(Self.grid * Self.grid * MemoryLayout<SIMD2<Float>>.stride)
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
    }

    private func random() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(rng >> 40) / Float(1 << 24)
    }

    /// A point inside the dish, at most `reach` of the radius from its centre, in grid cells.
    private func spot(_ reach: Float) -> SIMD2<Float> {
        let r = reach * sqrt(random()) * Float(Self.grid) / 2, t = random() * 2 * .pi
        return SIMD2(Float(Self.grid) / 2 + r * cos(t), Float(Self.grid) / 2 + r * sin(t))
    }

    /// A fresh dish: new pacemakers of different strengths (so different rhythms), a few bubbles, new framing.
    private func newDish() {
        pacemakers = (0..<(2 + Int(random() * 3))).map { _ in
            let p = spot(0.8)
            return Feature(x: p.x, y: p.y, r: 4 + random() * 2, b: -0.05 - random() * 0.07)
        }
        bubbles = (0..<(3 + Int(random() * 4))).map { _ in
            let p = spot(0.92)
            return Feature(x: p.x, y: p.y, r: 2.5 + random() * 3, b: 9)
        }
        framing = SIMD4((random() - 0.5) * 0.2, (random() - 0.5) * 0.04, (random() - 0.5) * 0.16, random())
        age = 0
        nextBreak = 15 + random() * 10
        spirals = 0
        fresh = true
    }

    private func uploadFeatures() {
        let p = features.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let all = (pacemakers + bubbles).prefix(32)
        for (i, f) in all.enumerated() { p[i] = SIMD4(f.x, f.y, f.r, f.b) }
    }

    func draw(_ f: Frame) {
        let n = Self.grid
        if states.isEmpty {
            states = (0..<2).map { _ in f.gpu.texture(n, n, .rg32Float) }
            medium = f.gpu.texture(n, n, .rg16Float)
            newDish()
        }
        if light?.width != f.target.width || light?.height != f.target.height {
            light = f.gpu.texture(f.target.width, f.target.height)
        }

        // Fade between dishes.
        if age > Self.dishSeconds { level = max(level - f.dt / 3, 0) } else { level = min(level + f.dt / 2, 1) }
        if age > Self.dishSeconds, level == 0 { newDish() }

        // Bubbles of CO2 slowly grow as the malonic acid is oxidised.
        for i in bubbles.indices { bubbles[i].r = min(bubbles[i].r + f.dt * 0.004, 9) }
        let featureCount = Float(min(pacemakers.count + bubbles.count, 32))
        sincePaint += f.dt
        if fresh || sincePaint > 2 {
            sincePaint = 0
            uploadFeatures()
            f.cmd.compute(paint, n, n, uniforms: f.uniforms([featureCount, Float(n), Self.b, framing.w * 100]), textures: [medium!],
                          buffers: [features])
        }

        var steps = 0
        if fresh {
            // A dish that's been going a couple of minutes: clear it, and let the pacemakers and a first broken
            // front get started before the first frame.
            fresh = false
            for t in states { f.cmd.compute(breaker, n, n, uniforms: f.uniforms([0, 0, 1e6, 0], [1, 0, 0, 0]), textures: [t]) }
            simulate(f, 1500, dt: 0.02, breakAt: 700)
            age = 1500 * 0.02 / Self.pace
        }
        age += f.dt
        budget += f.dt * Self.pace / Self.dt
        steps = Int(budget)
        budget -= Float(steps)
        // A disturbance now and then, as when the dish is nudged or a bubble lifts away, breaks a front into spirals.
        nextBreak -= f.dt
        if nextBreak < 0, age < Self.dishSeconds - 120 {
            breakFront(f)
            nextBreak = 120 + random() * 150
        }
        simulate(f, steps, dt: Self.dt, breakAt: -1)

        f.cmd.compute(probe, 8, uniforms: f.uniforms(), textures: [states[current]], buffers: [probes])
        count(f)

        let u = f.uniforms([f.aspect, level, framing.x, framing.y], [framing.z, framing.w, featureCount, 0])
        f.cmd.draw(photo, into: light!, uniforms: u, textures: [states[current], medium!], buffers: [features])
        lens.develop(f, light!, Lens.Look(exposure: 1.0, bloom: 0.02, halation: 0.02, vignette: 0.4, aberration: 1.0, grain: 0.5))

        if let synth = f.sound { play(synth, f) }
    }

    private func simulate(_ f: Frame, _ steps: Int, dt: Float, breakAt: Int) {
        let n = Self.grid
        let u = f.uniforms([Self.a, Self.epsilon, dt, 1 / (Self.dx * Self.dx)])
        for i in 0..<steps {
            if i == breakAt { breakFront(f) }
            f.cmd.compute(step, n, n, uniforms: u, textures: [states[current], states[1 - current], medium!])
            current = 1 - current
        }
    }

    /// Erases the waves in a half-disc, so fronts crossing it get free ends.
    private func breakFront(_ f: Frame) {
        let p = spot(0.6)
        let angle = random() * 2 * .pi
        f.cmd.compute(breaker, Self.grid, Self.grid, uniforms: f.uniforms([p.x, p.y, 30 + random() * 25, angle], [0, 0, 0, 0]),
                      textures: [states[current]])
    }

    /// Every few seconds, counts the spiral tips (where the activator's front meets the recovery's back) from a copy
    /// of the field, and keeps the fraction of catalyst that's oxidised.
    private func count(_ f: Frame) {
        sinceCount += f.dt
        let n = Self.grid
        if reading, sinceCount > 0.5 {
            reading = false
            let p = readback.contents().assumingMemoryBound(to: SIMD2<Float>.self)
            spirals = Self.tips(p, n, v: Self.a / 2 - Self.b, obstacles: bubbles + pacemakers)
            var sum: Float = 0
            for i in stride(from: 0, to: n * n, by: 61) { sum += p[i].y }
            oxidised = sum / Float(n * n / 61) / 0.6
        }
        guard !reading, sinceCount > 4, let blit = f.cmd.makeBlitCommandEncoder() else { return }
        sinceCount = 0
        reading = true
        blit.copy(from: states[current], sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                  sourceSize: MTLSize(width: n, height: n, depth: 1), to: readback, destinationOffset: 0,
                  destinationBytesPerRow: n * 8, destinationBytesPerImage: n * n * 8)
        blit.endEncoding()
    }

    /// Spiral tips: phase singularities, where the phase of (u, v) about the point (½, a/2 − b) winds once round a small
    /// loop of grid points, clustered so each tip counts once.
    static func tips(_ p: UnsafePointer<SIMD2<Float>>, _ n: Int, v: Float, obstacles: [Feature] = []) -> Int {
        var found: [SIMD2<Float>] = []
        let middle = Float(n) / 2
        func phase(_ x: Int, _ y: Int) -> Float {
            let q = p[y * n + x]
            return atan2(q.y - v, q.x - 0.5)
        }
        func turn(_ a: Float, _ b: Float) -> Float {
            var d = b - a
            if d > .pi { d -= 2 * .pi } else if d < -.pi { d += 2 * .pi }
            return d
        }
        let s = 2
        for y in stride(from: 0, to: n - s, by: s) {
            for x in stride(from: 0, to: n - s, by: s) {
                let a = phase(x, y), b = phase(x + s, y), c = phase(x + s, y + s), d = phase(x, y + s)
                let winding = turn(a, b) + turn(b, c) + turn(c, d) + turn(d, a)
                guard abs(winding) > .pi else { continue }
                // Waves meeting the wall or a bubble wind round its edge too; those aren't spirals.
                let q = SIMD2(Float(x), Float(y))
                if ((q - middle) * (q - middle)).sum() > (middle - 6) * (middle - 6) { continue }
                if obstacles.contains(where: { let d = q - SIMD2($0.x, $0.y); return (d * d).sum() < ($0.r + 5) * ($0.r + 5) }) { continue }
                if !found.contains(where: { (($0 - q) * ($0 - q)).sum() < 144 }) { found.append(q) }
            }
        }
        return found.count
    }

    /// A drone that brightens as more of the dish turns blue, and a soft note each time a wave reaches one of eight
    /// points round the dish, like an electrode logging the reaction's oscillation.
    private func play(_ synth: Synth, _ f: Frame) {
        let p = probes.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let scale: [Float] = [1, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3, 2, 9.0 / 4, 5.0 / 2]
        let root: Float = 98
        noteCool -= f.dt
        for i in 0..<8 {
            let high = p[i].x > 0.5
            if high, !wasHigh[i], noteCool < 0 {
                let angle = Float(i) / 8 * 2 * .pi
                synth.play(i % 3 == 0 ? .bell : .pluck, root * 2 * scale[i], amp: 0.045 * level, pan: cos(angle) * 0.7, length: 2.5)
                noteCool = 0.35
            }
            wasHigh[i] = high
        }
        let blue = min(max(oxidised, 0), 1)
        synth.drone(0, root, 0.07 * level, pan: -0.2)
        synth.drone(1, root * 1.5, 0.045 * level, pan: 0.2)
        synth.drone(2, root * 2, (0.02 + 0.03 * blue) * level, pan: -0.4)
        synth.drone(3, root * 2.5, 0.02 * blue * level, pan: 0.4)
        synth.noise(0.015 * level, cutoff: 500, crackle: 1.5 * Float(bubbles.count) / 6, crackleTone: 5200)
    }

    private static let source = """
    // Barkley's model. a: a, ε, dt, 1/dx²   medium: threshold offset b (or a pacemaker's lower one), wall
    kernel void bzStep(constant U &u [[buffer(0)]], texture2d<float, access::read> s [[texture(0)]],
                       texture2d<float, access::write> o [[texture(1)]], texture2d<float, access::read> med [[texture(2)]],
                       uint2 g [[thread_position_in_grid]]) {
        int n = s.get_width();
        if (int(g.x) >= n || int(g.y) >= n) return;
        float2 m = med.read(g).rg;
        if (m.y > 0.5) { o.write(float4(0), g); return; }
        float2 c = s.read(g).rg;
        // No flux through the dish wall or a bubble: a missing neighbour counts as this cell.
        float sum = 0;
        int2 offs[4] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
        for (int i = 0; i < 4; i++) {
            uint2 q = uint2(clamp(int2(g) + offs[i], int2(0), int2(n - 1)));
            sum += med.read(q).g > 0.5 ? c.x : s.read(q).r;
        }
        float lap = (sum - 4 * c.x) * u.a.w;
        float k = u.a.z / u.a.y;
        float th = (c.y + m.x) / u.a.x;
        // Semi-implicit in the fast reaction, so the stiff kinetics stay stable at a useful time step.
        float un = c.x < th ? c.x / (1 - k * (1 - c.x) * (c.x - th)) : 1 - (1 - c.x) / (1 + k * c.x * (c.x - th));
        un = clamp(un + u.a.z * lap, 0.0, 1.0);
        if (m.x < 0) un = max(un, 0.002);
        float vn = c.y + u.a.z * (un - c.y);
        o.write(float4(un, vn, 0, 0), g);
    }

    // The medium: the dish's round wall, bubbles, and pacemakers. a: features, grid size, b
    kernel void bzMedium(constant U &u [[buffer(0)]], const device float4 *fs [[buffer(1)]],
                         texture2d<float, access::write> med [[texture(0)]], uint2 g [[thread_position_in_grid]]) {
        float n = u.a.y;
        if (float(g.x) >= n || float(g.y) >= n) return;
        float2 p = float2(g) + 0.5;
        // No dish is perfectly even: the excitability varies a little from place to place, so fronts wrinkle.
        float b = u.a.z + 0.012 * (fbm(p * 0.02 + u.a.w, 4) - 0.5), wall = length(p - n * 0.5) > n * 0.5 - 1 ? 1.0 : 0.0;
        for (int i = 0; i < int(u.a.x); i++) {
            float4 f = fs[i];
            if (length(p - f.xy) < f.z) {
                if (f.w > 5) wall = 1; else b = f.w;
            }
        }
        med.write(float4(b, wall, 0, 0), g);
    }

    // Erases the waves in a half-disc (a: centre, radius, angle of its cut edge) back to resting medium, leaving the
    // fronts that crossed it with free ends, which curl into pairs of spirals; with b.x = 1 resets the whole field.
    kernel void bzBreak(constant U &u [[buffer(0)]], texture2d<float, access::read_write> s [[texture(0)]],
                        uint2 g [[thread_position_in_grid]]) {
        if (g.x >= s.get_width() || g.y >= s.get_height()) return;
        if (u.b.x > 0.5) { s.write(float4(0), g); return; }
        float2 d = float2(g) - u.a.xy;
        if (length(d) < u.a.z && dot(d, float2(cos(u.a.w), sin(u.a.w))) > 0) s.write(float4(0), g);
    }

    // The field at eight points round the dish, for the sound.
    kernel void bzProbe(constant U &u [[buffer(0)]], texture2d<float, access::read> s [[texture(0)]],
                        device float2 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        if (i >= 8) return;
        float n = s.get_width();
        float t = float(i) / 8 * TAU;
        float r = (i % 2 == 0 ? 0.55 : 0.3) * n * 0.5;
        out[i] = s.read(uint2(n * 0.5 + r * cos(t), n * 0.5 + r * sin(t))).rg;
    }

    // Light from the overhead softbox falling on the bench at p, in centered coordinates.
    inline float overhead(float2 p) {
        float2 d = p - float2(-0.35, 0.55);
        return 1.0 / pow(1 + dot(d, d) * 0.35, 1.5);
    }

    // Colour that the liquid lets through, travelling down to the paper and back up: Beer–Lambert, with red ferroin
    // absorbing green and blue strongly, and blue ferriin absorbing red.
    inline float3 transmittance(float ox, float path) {
        const float3 ferroin = float3(0.10, 0.95, 1.40);
        const float3 ferriin = float3(0.95, 0.50, 0.16);
        return exp(-mix(ferroin, ferriin, ox) * path);
    }

    // A white card under the dish, on a dark slate bench: the card is lit most under the dish and falls off toward its
    // edges, the bench is nearly black with a faint sheen.
    inline float3 paper(float2 p, float angle, float2 centre) {
        float2 q = rot(angle) * (p - centre);
        float sheet = smoothstep(1.12, 1.10, abs(q.x)) * smoothstep(1.12, 1.10, abs(q.y));
        float fibre = 0.97 + 0.03 * fbm(q * 60, 3) + 0.015 * (noise(q * float2(400, 30)) - 0.5);
        float3 white = float3(0.80, 0.79, 0.76) * fibre * (1.0 - 0.8 * smoothstep(0.45, 1.15, length(q)));
        float3 bench = float3(0.0075, 0.0085, 0.010) * (0.8 + 0.4 * fbm(p * 5, 3)) * (0.6 + 0.4 * smoothstep(-1.2, 0.8, p.y));
        return mix(bench, white, sheet) * overhead(p) * 0.55;
    }

    // a: aspect, fade, framing (dish offset x, y)   b: paper angle, softbox highlight seed
    fragment float4 bzPhoto(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> field [[texture(0)]],
                            texture2d<float> med [[texture(1)]], const device float4 *fs [[buffer(1)]]) {
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float2 p = centered(in.uv, u.res);
        float px = 2 / u.res.y;
        float2 centre = float2(u.a.z, u.a.w);
        const float R = 0.88, squash = 0.985, rimScale = 1.03, glass = 0.016;
        float2 rimShift = float2(0, 0.012);
        float2 d = (p - centre) / float2(R, R * squash);
        float rb = length(d);
        float2 dt = (p - centre - rimShift) / (float2(R, R * squash) * rimScale);
        float rt = length(dt);
        float2 toLight = normalize(float2(-0.35, 0.55) - centre);

        // The paper, darkened in the dish's soft shadow and brightened by the glass wall's caustic.
        float3 c = paper(p, u.b.x, centre);
        float2 ds = (p - centre + toLight * 0.03) / float2(R, R * squash);
        float rs = length(ds);
        c *= 1 - 0.35 * smoothstep(1.16, 1.0, rs) * step(1.0, rb);
        c *= 1 + 0.25 * exp(-pow((rs - 1.035) / 0.012, 2.0)) * step(1.0, rb);

        float outer = 1 + glass / R;
        if (rt < outer + px * 2) {
            float3 inside;
            if (rb < 1) {
                // The liquid: the field mapped onto the dish, lit from beneath by the paper through the glass floor.
                float2 uv = d * 0.5 + 0.5;
                float2 st = field.sample(ls, uv).rg;
                // The catalyst oxidises in the sharp autocatalytic front and is reduced back slowly behind it.
                float ox = max(smoothstep(0.15, 0.75, st.x), pow(saturate(st.y / 0.6), 2.6));
                // The glass floor bows up a little, so the liquid is shallower in the middle.
                float depth = 0.85 + 0.3 * rb * rb;
                // The meniscus climbs the wall: a longer path, a darker band, and a thin bright line of reflection.
                float men = smoothstep(0.965, 1.0, rb);
                float3 under = paper(p, u.b.x, centre) * 0.92;
                inside = under * transmittance(ox, depth * (1 + men * 1.5));
                float glint = exp(-pow((rb - 0.993) / 0.0025, 2.0)) * pow(saturate(dot(normalize(d * float2(1, -1)), toLight * float2(1, -1)) * 0.5 + 0.5), 3);
                inside += float3(0.9, 0.9, 0.88) * glint * 0.35 * overhead(p);
                // Bubbles of CO2 on the floor of the dish: dark rims from total internal reflection, a bright core.
                for (int i = 0; i < int(u.b.z); i++) {
                    float4 f = fs[i];
                    if (f.w < 5) continue;
                    float2 bq = (uv * float(field.get_width()) - f.xy) / f.z;
                    float br = length(bq);
                    if (br < 1.15) {
                        // Little liquid lies over the bubble, so it shows paler; refraction darkens its edge.
                        float rim = smoothstep(0.6, 1.0, br) * smoothstep(1.15, 0.95, br);
                        float3 bubble = mix(under, inside, 0.35 + 0.3 * br * br) * (1 - 0.55 * rim);
                        float spec = exp(-dot(bq - float2(-0.3, -0.35), bq - float2(-0.3, -0.35)) * 60);
                        bubble += spec * 0.25 * overhead(p);
                        inside = mix(inside, bubble, smoothstep(1.15, 1.0, br));
                    }
                }
                // Specks of dust settled on the floor of the dish.
                float2 dq = uv * 90 + u.b.y * 50;
                float2 cell = floor(dq);
                float speck = step(0.93, hash12(cell)) * smoothstep(0.12, 0.04, length(fract(dq) - 0.2 - 0.6 * hash22(cell + 3.1)));
                inside *= 1 - 0.5 * speck * hash12(cell + 7.7);
                // The softbox mirrored in the flat liquid surface: a faint, soft-edged sheen.
                float2 hq = (p - (centre + float2(-0.32, 0.36) + (u.b.y - 0.5) * 0.15)) / float2(0.3, 0.2);
                float box = smoothstep(1.0, 0.55, length(max(abs(hq) - 0.5, 0.0)) + 0.5);
                inside += float3(0.06, 0.06, 0.058) * box;
            } else {
                // The glass wall and rim. The wall guides light like a lens, so it glows with the colour of the liquid
                // at its foot, over the paper seen through it, with a faint green cast toward its outer edge.
                float2 foot = normalize(d) * 0.975;
                float2 fs = field.sample(ls, foot * 0.5 + 0.5).rg;
                float fox = max(smoothstep(0.15, 0.75, fs.x), pow(saturate(fs.y / 0.6), 2.6));
                float3 under = paper(p + (p - centre) * 0.012, u.b.x, centre);
                float x = saturate((rt - 1) / (outer - 1));
                float3 tint = mix(float3(1.0), transmittance(fox, 1.3), 0.75 - 0.4 * x);
                inside = under * tint * mix(float3(0.95, 0.97, 0.96), float3(0.80, 0.90, 0.84), x) * (0.8 + 0.3 * x);
            }
            {
                float facing = saturate(dot(normalize(dt * float2(1, -1)), toLight * float2(1, -1)) * 0.5 + 0.5);
                float lines = 0, glints = 0;
                // The rim's inner and outer edges, and the foot of the wall where the liquid meets it.
                float edges[3] = {1.0, outer, 1.0};
                for (int i = 0; i < 3; i++) {
                    float r = i < 2 ? rt : rb;
                    float w = (r - edges[i]) * R / px;
                    lines += exp(-w * w / 1.5) * (i == 1 ? 0.3 : 0.2);
                    glints += exp(-(w + 1.2) * (w + 1.2) / 1.5) * (i == 0 ? 1.0 : 0.4);
                }
                // The polished rim mirrors the softbox on the side toward it, and more faintly on the far side.
                float arc = pow(facing, 6) + 0.25 * pow(1 - facing, 8);
                inside *= 1 - saturate(lines);
                inside += float3(1.0, 0.99, 0.96) * glints * (0.04 + arc) * 0.8 * overhead(p);
            }
            float cover = saturate((outer - rt) / px * 0.5 + 0.5);
            c = mix(c, inside, cover);
        }
        return float4(c * u.a.y, 1);
    }
    """
}
