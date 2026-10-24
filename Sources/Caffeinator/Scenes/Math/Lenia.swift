import Foundation
import Metal

/// Bert Chan's Lenia: a continuous cellular automaton whose gliders look alive. Each step convolves the field with a
/// smooth ring kernel and grows or shrinks every cell by a bell curve of the result. Orbium creatures glide across a
/// dark sea, collide and dissolve; new ones drift in from a margin hidden past the screen's edges.
@MainActor final class Lenia: Scene {
    private static let radius = 13
    private static let mu: Float = 0.15
    private static let sigma: Float = 0.015
    private static let period: Float = 10
    /// Simulation steps per second; each moves an orbium about 0.6 cells.
    private static let rate: Float = 7
    private static let visibleHeight = 124
    private static let margin = 18
    /// The direction an unrotated orbium swims, in grid cells (x right, y down): 68 degrees.
    private static let heading: Float = 1.194
    /// Mass of one orbium, in cell units.
    private static let orbiumMass: Float = 71

    /// Orbium unicaudatus, from Chan's Lenia notebook (20 × 20 cells, R = 13).
    private static let orbium: [Float] = [
        0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45, 0, 0, 0,
        0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0, 0,
        0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0, 0,
        0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0, 0,
        0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17, 0,
        0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0,
        0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0,
        0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22, 0.07,
        0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11,
        0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1,
        0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21, 0.05,
        0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2, 0.01,
        0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24, 0.13, 0,
        0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29, 0.19, 0.04, 0,
        0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21, 0.08, 0, 0,
        0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0,
    ]

    /// A creature found in the last readback: centroid in cells and mass.
    struct Creature { var x: Float; var y: Float; var mass: Float; var size: Float = 0 }

    private let stepKernel: MTLComputePipelineState
    private let stampKernel: MTLComputePipelineState
    private let prepare: MTLComputePipelineState
    private let render: MTLRenderPipelineState
    private let bloom: Bloom
    private let taps: MTLBuffer
    private let tapCount: Int
    private let pattern: MTLBuffer
    private let stats: [MTLBuffer]
    private var fields: [MTLTexture] = []
    private var look: MTLTexture?
    private var wake: MTLTexture?
    private var hdr: MTLTexture?
    private var readback: MTLBuffer?
    private var gw = 0, gh = 0, vw = 0

    private var current = 0
    private var steps = 0
    private var clock: Float = 0
    private var stamps: [SIMD4<Float>] = []
    private var rng: UInt64
    private var creatures: [Creature] = []
    private var counted = 0
    private var sinceSpawn: Float = 0
    private var sinceRead: Float = 0
    private var pendingRead = false
    private var mass: Float = 0
    private var motion: Float = 0
    private var breath: Float = 0
    /// The school's general heading; each creature swims within half a radian of it, so collisions stay occasional.
    private var school: Float = 0
    /// Regions being dissolved: x, y, radius in cells, seconds left.
    private var dissolving: [SIMD4<Float>] = []

    var scale: Float { 0.6 }

    var facts: String? {
        let n = counted == 1 ? "1 orbium glider on screen" : "\(counted) orbium gliders on screen"
        return "Lenia: \(n), kernel radius 13, growth centre μ 0.15, width σ 0.015, time step one tenth"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        stepKernel = try gpu.kernel(library, "leniaStep")
        stampKernel = try gpu.kernel(library, "leniaStamp")
        prepare = try gpu.kernel(library, "leniaLook")
        render = try gpu.pipeline(library, "leniaRender")
        bloom = try Bloom(gpu: gpu)
        // The kernel: a smooth bump over a ring of radius R, normalised to sum to one.
        var list: [SIMD4<Float>] = []
        let r = Self.radius
        for dy in -r...r {
            for dx in -r...r {
                let d = (Float(dx * dx + dy * dy)).squareRoot() / Float(r)
                guard d > 0, d < 1 else { continue }
                list.append(SIMD4(Float(dx), Float(dy), exp(4 - 1 / (d * (1 - d))), 0))
            }
        }
        let total = list.reduce(0) { $0 + $1.z }
        taps = gpu.buffer(list.map { SIMD4($0.x, $0.y, $0.z / total, 0) })
        tapCount = list.count
        pattern = gpu.buffer(Self.orbium)
        stats = (0..<2).map { _ in gpu.buffer(16) }
        rng = UInt64.random(in: 1...UInt64.max)
        if let v = Snapshot.variant { rng = UInt64(v + 1) &* 0x9E37_79B9_7F4A_7C15 }
    }

    private func random() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(rng >> 40) / Float(1 << 24)
    }

    /// Lays the grid out to the screen's shape, with a hidden margin all round, and scatters the first creatures.
    private func setUp(_ f: Frame) {
        vw = Int((Float(Self.visibleHeight) * f.aspect).rounded())
        gw = vw + 2 * Self.margin
        gh = Self.visibleHeight + 2 * Self.margin
        fields = (0..<3).map { _ in f.gpu.texture(gw, gh, .r32Float) }
        look = f.gpu.texture(gw, gh, .rgba16Float)
        wake = f.gpu.texture(gw, gh, .r32Float)
        readback = f.gpu.buffer(gw * gh * 4)
        let blank = U()
        for t in fields { f.cmd.compute(stampKernel, gw, gh, uniforms: blank, textures: [t], buffers: [pattern]) }
        f.cmd.compute(stampKernel, gw, gh, uniforms: blank, textures: [wake!], buffers: [pattern])
        var placed: [SIMD2<Float>] = []
        let want = 7
        for _ in 0..<200 where placed.count < want {
            let p = SIMD2(Float(Self.margin) + 12 + random() * Float(vw - 24), Float(Self.margin) + 12 + random() * Float(Self.visibleHeight - 24))
            if placed.allSatisfy({ (($0 - p) * ($0 - p)).sum() > 42 * 42 }) { placed.append(p) }
        }
        school = random() * 2 * .pi
        for p in placed { stamps.append(SIMD4(p.x, p.y, swimAngle(), 1)) }
        counted = placed.count
    }

    func draw(_ f: Frame) {
        if gw == 0 || hdr?.width != f.target.width || hdr?.height != f.target.height {
            hdr = f.gpu.texture(f.target.width, f.target.height)
            if gw == 0 { setUp(f) }
        }
        let size = (gw, gh)

        // New creatures are stamped into both the last and current fields, so they don't fade in from nothing.
        for s in stamps {
            for i in [current, (current + 2) % 3] {
                f.cmd.compute(stampKernel, 36, 36, uniforms: f.uniforms(s, [Float(gw), Float(gh), 1, 0]), textures: [fields[i]],
                              buffers: [pattern])
            }
        }
        stamps.removeAll()

        clock += f.dt * Self.rate
        while clock >= 1 {
            clock -= 1
            let next = (current + 1) % 3
            let st = stats[steps % 2]
            st.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 4)
            f.cmd.compute(stepKernel, size.0, size.1,
                          uniforms: f.uniforms([Self.mu, Self.sigma, 1 / Self.period, Float(tapCount)], [Float(gw), Float(gh), 0, 0],
                                               dissolving.first ?? .zero, dissolving.dropFirst().first ?? .zero),
                          textures: [fields[current], fields[next]], buffers: [taps, st])
            current = next
            steps += 1
        }
        dissolving = dissolving.map { SIMD4($0.x, $0.y, $0.z, $0.w - f.dt) }.filter { $0.w > 0 }
        let previous = (current + 2) % 3
        let u = f.uniforms([clock, 1 - exp(-f.dt / 5), Float(Self.margin), Float(vw)], [Float(gw), Float(gh), Float(Self.visibleHeight), 0])
        f.cmd.compute(prepare, size.0, size.1, uniforms: u,
                      textures: [fields[previous], fields[current], look!, wake!, fields[(current + 1) % 3]])
        f.cmd.draw(render, into: hdr!, uniforms: u, textures: [look!, wake!, look!, look!])
        let glow = bloom.glow(f.cmd, hdr!)
        f.cmd.draw(render, into: f.target, uniforms: f.uniforms([clock, 0, Float(Self.margin), Float(vw)],
                                                                 [Float(gw), Float(gh), Float(Self.visibleHeight), 1]),
                   textures: [look!, wake!, hdr!, glow])

        let s = stats[(steps + 1) % 2].contents().assumingMemoryBound(to: UInt32.self)
        if steps > 1 {
            mass += (Float(s[0]) / 1000 - mass) * min(f.dt, 1)
            motion += (Float(s[1]) / 1000 - motion) * min(f.dt, 1)
        }
        census(f)
        if let synth = f.sound { play(synth, f) }
    }

    /// Once a second, copies the field back to count creatures and find where they are; tops the population up
    /// from the hidden margin when collisions have thinned it out.
    private func census(_ f: Frame) {
        sinceRead += f.dt
        sinceSpawn += f.dt
        guard let readback else { return }
        if pendingRead, sinceRead > 0.3 {
            pendingRead = false
            let before = creatures
            creatures = Self.find(readback.contents().assumingMemoryBound(to: Float.self), gw, gh)
            counted = creatures.filter(onScreen).count
            if let synth = f.sound, creatures.count < before.count, let gone = before.first {
                synth.play(.bell, 110 * 3, amp: 0.05, pan: pan(gone.x), length: 4)
            }
            let total = creatures.reduce(0) { $0 + $1.mass }
            if creatures.count < 8, sinceSpawn > 3 || creatures.isEmpty {
                spawn(readback.contents().assumingMemoryBound(to: Float.self), f)
            }
            for c in creatures where c.mass > Self.orbiumMass * 2.3 && dissolving.count < 2 {
                dissolving.append(SIMD4(c.x, c.y, c.size + 8, 2.5))
                f.sound?.play(.thump, 55, amp: 0.06, pan: pan(c.x), length: 2)
            }
            if total > Self.orbiumMass * 30 {
                for i in 0..<3 { f.cmd.compute(stampKernel, gw, gh, uniforms: U(), textures: [fields[i]], buffers: [pattern]) }
                creatures = []
            }
        }
        if !pendingRead, sinceRead > 1 {
            sinceRead = 0
            pendingRead = true
            guard let blit = f.cmd.makeBlitCommandEncoder() else { return }
            blit.copy(from: fields[current], sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                      sourceSize: MTLSize(width: gw, height: gh, depth: 1), to: readback, destinationOffset: 0,
                      destinationBytesPerRow: gw * 4, destinationBytesPerImage: gw * gh * 4)
            blit.endEncoding()
        }
    }

    /// Seeds an orbium somewhere empty in the hidden margin, heading anywhere.
    private func spawn(_ field: UnsafePointer<Float>, _ f: Frame) {
        school += 0.08
        for _ in 0..<60 {
            let along = random()
            // The hidden bands on opposite sides meet across the wrap, so the grid's own edge is their middle;
            // enter across the band the school is swimming through.
            let p: SIMD2<Float> = abs(cos(school)) > abs(sin(school)) ? SIMD2(0, along * Float(gh)) : SIMD2(along * Float(gw), 0)
            var empty = true
            for dy in stride(from: -30, through: 30, by: 3) where empty {
                for dx in stride(from: -30, through: 30, by: 3) {
                    let x = (Int(p.x) + dx + gw) % gw, y = (Int(p.y) + dy + gh) % gh
                    if field[y * gw + x] > 0.02 { empty = false; break }
                }
            }
            if empty {
                stamps.append(SIMD4(p.x, p.y, swimAngle(), 1))
                sinceSpawn = 0
                f.sound?.play(.bell, 110 * 6, amp: 0.035, pan: pan(p.x), length: 3)
                return
            }
        }
    }

    private func onScreen(_ c: Creature) -> Bool {
        c.x > Float(Self.margin) && c.x < Float(gw - Self.margin) && c.y > Float(Self.margin) && c.y < Float(gh - Self.margin)
    }

    /// Stamp rotation for a creature heading near the school's direction.
    private func swimAngle() -> Float { school + (random() - 0.5) * 0.6 - Self.heading }

    private func pan(_ x: Float) -> Float {
        max(-0.9, min(0.9, (x - Float(gw) / 2) / Float(vw) * 1.8))
    }

    /// Connected regions of the field above a faint threshold, wrapping round the torus; those heavy enough to be
    /// creatures, with their centroids.
    static func find(_ a: UnsafePointer<Float>, _ w: Int, _ h: Int) -> [Creature] {
        var seen = [Bool](repeating: false, count: w * h)
        var out: [Creature] = []
        var stack: [Int] = []
        for start in 0..<(w * h) where !seen[start] && a[start] > 0.05 {
            seen[start] = true
            stack.append(start)
            var m: Float = 0, sx: Float = 0, sy: Float = 0, far: Float = 0
            let ox = start % w, oy = start / w
            while let i = stack.popLast() {
                let x = i % w, y = i / w
                let v = a[i]
                m += v
                // Unwrap relative to the first cell, so creatures straddling an edge get a sensible centroid.
                var dx = x - ox, dy = y - oy
                if dx > w / 2 { dx -= w } else if dx < -w / 2 { dx += w }
                if dy > h / 2 { dy -= h } else if dy < -h / 2 { dy += h }
                sx += v * Float(dx)
                sy += v * Float(dy)
                far = max(far, Float(abs(dx) + abs(dy)))
                for (nx, ny) in [((x + 1) % w, y), ((x + w - 1) % w, y), (x, (y + 1) % h), (x, (y + h - 1) % h)] {
                    let j = ny * w + nx
                    if !seen[j], a[j] > 0.05 {
                        seen[j] = true
                        stack.append(j)
                    }
                }
            }
            if m > 25 {
                let cx = (Float(ox) + sx / m).truncatingRemainder(dividingBy: Float(w))
                let cy = (Float(oy) + sy / m).truncatingRemainder(dividingBy: Float(h))
                out.append(Creature(x: cx < 0 ? cx + Float(w) : cx, y: cy < 0 ? cy + Float(h) : cy, mass: m, size: far))
            }
        }
        return out
    }

    /// A breathing pad: its swell follows the total mass, its brightness how fast the field is changing, and each
    /// creature hums a note set by its height on screen, panned to where it swims.
    private func play(_ synth: Synth, _ f: Frame) {
        breath += f.dt * (0.09 + min(motion / 400, 1) * 0.05) * 2 * .pi
        let swell = 0.75 + 0.25 * sin(breath)
        let life = min(mass / (Self.orbiumMass * 6), 1.2)
        let root: Float = 98
        synth.drone(0, root, 0.07 * swell * (0.4 + 0.6 * life), pan: -0.15)
        synth.drone(1, root * 1.5, 0.045 * swell * life, pan: 0.15)
        let scale: [Float] = [2, 9.0 / 4, 5.0 / 2, 3, 10.0 / 3, 4, 9.0 / 2, 5]
        let sorted = creatures.sorted { $0.x < $1.x }
        for i in 0..<6 {
            guard i < sorted.count else { synth.drone(2 + i, root * 2, 0); continue }
            let c = sorted[i]
            let height = 1 - (c.y - Float(Self.margin)) / Float(Self.visibleHeight)
            let note = scale[max(0, min(scale.count - 1, Int(height * Float(scale.count))))]
            synth.drone(2 + i, root * note, (onScreen(c) ? 0.022 : 0.006) * swell, pan: pan(c.x))
        }
        synth.noise(0.012 + min(motion / 300, 1) * 0.02, cutoff: 260 + min(motion / 300, 1) * 500)
    }

    private static let source = """
    // a: mu, sigma, dt, tap count. Taps are (dx, dy, weight). stats: total mass, total change, ×1000.
    kernel void leniaStep(constant U &u [[buffer(0)]], const device float4 *taps [[buffer(1)]],
                          device atomic_uint *stats [[buffer(2)]],
                          texture2d<float, access::read> a [[texture(0)]], texture2d<float, access::write> b [[texture(1)]],
                          uint2 g [[thread_position_in_grid]]) {
        int w = a.get_width(), h = a.get_height();
        bool valid = int(g.x) < w && int(g.y) < h;
        float sum = 0, was = 0, now = 0;
        if (valid) {
            int n = int(u.a.w);
            for (int i = 0; i < n; i++) {
                float4 t = taps[i];
                int x = (int(g.x) + int(t.x) + w) % w, y = (int(g.y) + int(t.y) + h) % h;
                sum += a.read(uint2(x, y)).r * t.z;
            }
            float d = (sum - u.a.x) / u.a.y;
            float grow = 2 * exp(-0.5 * d * d) - 1;
            was = a.read(g).r;
            now = clamp(was + u.a.z * grow, 0.0, 1.0);
            // Collisions occasionally breed a blob that grows without end; it melts away instead.
            for (int k = 0; k < 2; k++) {
                float4 z = k == 0 ? u.c : u.d;
                if (z.w <= 0) continue;
                float2 d = abs(float2(g) - z.xy);
                d = min(d, u.b.xy - d);
                if (length(d) < z.z) now *= 0.82;
            }
            b.write(float4(now), g);
        }
        uint m = simd_sum(uint(now * 1000));
        uint c = simd_sum(uint(abs(now - was) * 1000));
        if (simd_is_first()) {
            atomic_fetch_add_explicit(&stats[0], m, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[1], c, memory_order_relaxed);
        }
    }

    // Stamps an orbium rotated by a.z, centred on a.xy (b.xy is the grid size). With a.w = 0, clears the texture.
    kernel void leniaStamp(constant U &u [[buffer(0)]], const device float *orb [[buffer(1)]],
                           texture2d<float, access::read_write> t [[texture(0)]], uint2 g [[thread_position_in_grid]]) {
        if (u.a.w == 0) {
            if (g.x < t.get_width() && g.y < t.get_height()) t.write(float4(0), g);
            return;
        }
        float2 d = float2(g) - 17.5;
        float c = cos(u.a.z), s = sin(u.a.z);
        float2 q = float2(c * d.x + s * d.y, -s * d.x + c * d.y) + 9.5;
        if (any(q < 0.0) || any(q > 19.0)) return;
        int2 i = int2(floor(q));
        float2 f = q - float2(i);
        int2 j = min(i + 1, 19);
        float v = mix(mix(orb[i.y * 20 + i.x], orb[i.y * 20 + j.x], f.x), mix(orb[j.y * 20 + i.x], orb[j.y * 20 + j.x], f.x), f.y);
        int2 p = int2(floor(u.a.xy + d + 0.5));
        uint2 at = uint2((p.x + int(u.b.x)) % int(u.b.x), (p.y + int(u.b.y)) % int(u.b.y));
        t.write(float4(max(t.read(at).r, v)), at);
    }

    inline float blended(texture2d<float, access::read> prev, texture2d<float, access::read> cur, uint2 g, int dx, int dy, float t) {
        int w = cur.get_width(), h = cur.get_height();
        uint2 p = uint2((int(g.x) + dx + w) % w, (int(g.y) + dy + h) % h);
        return mix(prev.read(p).r, cur.read(p).r, t);
    }

    // Blends the last two steps by a.x, and stores the density, its gradient and its change for lighting. The wake
    // remembers where creatures have been, fading with rate a.y.
    kernel void leniaLook(constant U &u [[buffer(0)]], texture2d<float, access::read> prev [[texture(0)]],
                          texture2d<float, access::read> cur [[texture(1)]], texture2d<float, access::write> look [[texture(2)]],
                          texture2d<float, access::read_write> wake [[texture(3)]], texture2d<float, access::read> older [[texture(4)]],
                          uint2 g [[thread_position_in_grid]]) {
        int w = cur.get_width(), h = cur.get_height();
        if (int(g.x) >= w || int(g.y) >= h) return;
        float t = u.a.x;
        float v = blended(prev, cur, g, 0, 0, t);
        float2 grad = float2(blended(prev, cur, g, 1, 0, t) - blended(prev, cur, g, -1, 0, t),
                             blended(prev, cur, g, 0, 1, t) - blended(prev, cur, g, 0, -1, t)) * 0.5;
        float p = prev.read(g).r;
        float change = mix(p - older.read(g).r, cur.read(g).r - p, t);
        look.write(float4(v, grad, change), g);
        float old = wake.read(g).r;
        wake.write(float4(max(old * (1 - u.a.y), v)), g);
    }

    // Cubic B-spline filtering from four bilinear taps, so the coarse grid scales up without a trace of its cells.
    inline float4 smooth(texture2d<float> t, float2 p) {
        constexpr sampler s(filter::linear, address::repeat);
        float2 size = float2(t.get_width(), t.get_height());
        float2 q = p - 0.5, i = floor(q), f = q - i;
        float2 w0 = (1 - f) * (1 - f) * (1 - f) / 6, w1 = (4 - 6 * f * f + 3 * f * f * f) / 6;
        float2 w2 = (1 + 3 * f + 3 * f * f - 3 * f * f * f) / 6, w3 = f * f * f / 6;
        float2 g0 = w0 + w1, g1 = w2 + w3;
        float2 h0 = (w1 / g0 - 1 + i + 0.5) / size, h1 = (w3 / g1 + 1 + i + 0.5) / size;
        return (t.sample(s, float2(h0.x, h0.y)) * g0.x + t.sample(s, float2(h1.x, h0.y)) * g1.x) * g0.y
             + (t.sample(s, float2(h0.x, h1.y)) * g0.x + t.sample(s, float2(h1.x, h1.y)) * g1.x) * g1.y;
    }

    // Pass 1 (b.w = 0) lights the creatures into HDR; pass 2 (b.w = 1) adds the glow and tone maps.
    // a: blend, wake fade, margin, visible width   b: grid size, visible height, pass
    fragment float4 leniaRender(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> look [[texture(0)]],
                                texture2d<float> wake [[texture(1)]], texture2d<float> hdr [[texture(2)]],
                                texture2d<float> glow [[texture(3)]]) {
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        if (u.b.w > 0.5) {
            float3 c = hdr.sample(ls, in.uv).rgb + glow.sample(ls, in.uv).rgb * 0.5;
            return float4(film(c), 1);
        }
        float2 p = float2(u.a.z) + in.uv * float2(u.a.w, u.b.z);
        float4 f = smooth(look, p);
        float trail = smooth(wake, p).r;
        float a = saturate(f.x);
        float2 grad = f.yz;

        // Deep water: a faint glow from above and slow drifting motes.
        float2 c = centered(in.uv, u.res);
        float3 col = mix(float3(0.001, 0.003, 0.007), float3(0.004, 0.012, 0.022), saturate(0.4 + 0.6 * c.y));
        float n = fbm(float3(c * 1.6, u.time * 0.02), 3);
        col *= 0.7 + 0.6 * n;
        float2 mp = c * 38 + float2(u.time * 0.15, u.time * 0.4);
        float2 cell = floor(mp), off = hash22(cell);
        float mote = smoothstep(0.09, 0.0, length(fract(mp) - off * 0.8 - 0.1)) * step(0.82, hash12(cell + 7.1));
        col += float3(0.05, 0.10, 0.12) * mote * (0.5 + 0.5 * sin(u.time * 0.7 + off.x * 40));

        // The wake: a cool phosphorescent trail where creatures have passed.
        col += float3(0.00, 0.07, 0.12) * pow(saturate(trail - a), 1.5) * 0.5;

        // The body: translucent membrane, luminous interior, a warm core. Growing edges glow cyan, the trailing
        // edge where the creature dissolves glows violet.
        // Two kinds of light, cyan and green-gold, drift across the sea so creatures change colour as they swim.
        float kind = smoothstep(0.35, 0.65, noise(float3(p * 0.012, u.time * 0.008)));
        float3 membrane = mix(float3(0.02, 0.35, 0.55), float3(0.10, 0.42, 0.22), kind);
        float3 body = mix(float3(0.10, 0.65, 1.00), float3(0.40, 0.95, 0.45), kind);
        float3 core = mix(float3(0.85, 1.00, 0.75), float3(1.00, 0.92, 0.60), kind);
        float3 tint = mix(membrane, body, smoothstep(0.08, 0.5, a));
        tint = mix(tint, core, smoothstep(0.6, 1.0, a));
        float change = f.w * 12;
        tint = mix(tint, float3(0.15, 1.0, 0.75), saturate(change) * 0.7);
        tint = mix(tint, float3(0.75, 0.25, 1.0), saturate(-change) * 0.8);
        float3 normal = normalize(float3(-grad * 9, 1));
        float3 light = normalize(float3(-0.5, 0.6, 0.65));
        float diffuse = saturate(dot(normal, light));
        float spec = pow(saturate(dot(normal, normalize(light + float3(0, 0, 1)))), 40);
        float rim = saturate(length(grad) * 7) * smoothstep(0.0, 0.15, a);
        float3 lit = tint * (0.35 + 0.9 * diffuse) * pow(a, 1.1) * 1.3 + tint * rim * 0.6 + float3(0.8, 0.95, 1.0) * spec * a * 0.5;
        col += lit;
        return float4(col, 1);
    }
    """
}
