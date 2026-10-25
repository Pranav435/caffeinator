import Foundation
import AppKit
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
    private let lens: Lens
    private var milky: MTLTexture?
    private var dummy: MTLTexture?
    private var loaded = false
    private var backdropAt: Float = -1000
    private var skyNote = ""
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
    var ready: Bool { loaded }
    var credit: String? { milky == nil ? nil : MilkyWay.credit }

    var facts: String? {
        let r = String(format: "%.2f", order)
        let k = String(format: "%.1f", coupling / Self.critical)
        if gust >= 0 { return "\(Self.count.formatted()) fireflies; a breeze is sweeping the meadow and scattering their rhythm, synchrony \(r)" }
        let sky = skyNote.isEmpty ? "" : "; \(skyNote)"
        let state = order > 0.8 ? "flashing together, the pulse rolling across the meadow" : order > 0.4 ? "falling into step" : "flashing on their own"
        return "\(Self.count.formatted()) fireflies \(state); synchrony \(r), coupling \(k) times the critical strength\(sky)"
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        scenery = try gpu.pipeline(library, "meadow")
        sky = try gpu.pipeline(library, "night")
        pools = try gpu.pipeline(library, "pool", vertex: "poolVertex", blend: .add)
        glows = try gpu.pipeline(library, "bokeh", vertex: "flyVertex", blend: .add)
        grass = try gpu.pipeline(library, "blades", blend: .alpha)
        lens = try Lens(gpu: gpu)
        buffers = (0..<3).map { _ in gpu.buffer(Self.count * MemoryLayout<SIMD4<Float>>.stride * 3) }
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
        Task { @MainActor in
            milky = await MilkyWay.texture(gpu)
            loaded = true
            backdropAt = -1000
        }
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
            backdrop = f.gpu.texture(f.target.width, f.target.height)
            dummy = f.gpu.texture(1, 1)
            backdropAt = -1000
        }
        // The sky, woods and meadow stand still, so they're drawn once, and again every couple of minutes as the real
        // sky turns (the Milky Way moves a quarter of a degree in that time).
        if f.time - backdropAt > 120 {
            backdropAt = f.time
            let (right, forward, up, note) = Self.camera(Date())
            skyNote = milky == nil ? "" : note
            f.cmd.draw(scenery, into: backdrop!, uniforms: f.uniforms([f.aspect, milky == nil ? 0 : 1, 0, f.seed], SIMD4(right, 0),
                                                                       SIMD4(forward, 0), SIMD4(up, 0)),
                       textures: [milky ?? dummy!])
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
            let before = place(fly, f.time - 0.45).0, middle = place(fly, f.time - 0.2).0
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
            buffer[3 * i] = SIMD4(p.x, p.y, fly.d, flash)
            buffer[3 * i + 1] = SIMD4(fly.tint, fly.h, before.x, before.y)
            buffer[3 * i + 2] = SIMD4(middle.x, middle.y, 0, 0)
        }
        shine += (lit / Float(Self.count) - shine) * min(dt * 8, 1)

        let u = f.uniforms([f.aspect, wind, shine, f.seed], [front, Self.horizon, 0, 0])
        let flyBuffer = buffers[Int(f.index % 3)]
        f.cmd.draw(pools, into: light!, uniforms: u, buffers: [flyBuffer], vertices: 6, instances: Self.count)
        f.cmd.draw(sky, into: hdr!, uniforms: u, textures: [backdrop!, light!])
        f.cmd.draw(glows, into: hdr!, clear: false, uniforms: u, buffers: [flyBuffer], vertices: 6, instances: Self.count)
        f.cmd.draw(grass, into: hdr!, clear: false, uniforms: u, textures: [light!])
        lens.develop(f, hdr!, Lens.Look(exposure: 1.5, white: [0.95, 1, 1.08], bloom: 0.07, halation: 0.05, vignette: 0.45,
                                        aberration: 2, grain: 1.1))

        if let synth = f.sound { play(synth, f, onsets) }
    }

    /// A flash's brightness at a phase: a soft rise over about 0.3 s up to the moment the phase wraps, a fade over
    /// about 0.6 s after it, and dark in between, so a shared pulse stands out against the night.
    static func flash(_ phase: Float) -> Float {
        let before = 2 * .pi - phase
        if before < 0.6 {
            let t = 1 - before / 0.6
            return t * t * (3 - 2 * t)
        }
        if phase < 1.1 {
            let t = 1 - phase / 1.1
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
        // Fireflies fly as they flash: slow loops a few tens of centimetres wide, which a long exposure draws as streaks.
        x += 0.34 * sin(t * 0.85 + fly.wy * 3) + 0.12 * sin(t * 2.1 + fly.wz)
        let h = fly.h + sin(t * 0.09 + fly.wz) * 0.2 * min(fly.h, 1) + 0.06 * sin(t * 0.4 + fly.wx)
              + 0.26 * sin(t * 1.15 + fly.wz * 2) + 0.1 * sin(t * 2.3 + fly.wx)
        return (SIMD2(x / d, Self.horizon + (h - 1) / d), SIMD2(x, d))
    }

    /// The camera's basis in J2000 equatorial coordinates for the real sky now over the Mac, as (right, forward, up) with
    /// the horizon a little below the middle of the frame, and a spoken note. It faces the compass direction that
    /// shows the most Milky Way.
    static func camera(_ date: Date) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, String) {
        let place = Location.shared
        let lat = Astro.rad(place.latitude), theta = Astro.rad(Astro.lst(date, longitude: place.longitude))
        let east = SIMD3(-sin(theta), cos(theta), 0)
        let north = SIMD3(-sin(lat) * cos(theta), -sin(lat) * sin(theta), cos(lat))
        let zenith = SIMD3(cos(lat) * cos(theta), cos(lat) * sin(theta), sin(lat))
        let gal: [SIMD3<Double>] = [[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445],
                                    [-0.8676661490, -0.1980763734, 0.4559837762]]
        func bright(_ v: SIMD3<Double>) -> Double {
            let b = asin(max(-1, min(1, (gal[2] * v).sum())))
            let l = atan2((gal[1] * v).sum(), (gal[0] * v).sum())
            return exp(-b * b / (2 * 0.17 * 0.17)) * (1 + 2 * exp(-l * l / (2 * 0.7 * 0.7)))
        }
        var best = (score: -1.0, az: 0.0)
        for step in 0..<36 {
            let az = Double(step) * .pi / 18
            var score = 0.0
            for off in stride(from: -0.6, through: 0.6, by: 0.3) {
                for el in stride(from: 0.15, through: 0.9, by: 0.15) {
                    let a = az + off
                    let v = east * cos(el) * sin(a) + north * cos(el) * cos(a) + zenith * sin(el)
                    score += bright(SIMD3(Double(v.x), Double(v.y), Double(v.z)))
                }
            }
            if score > best.score { best = (score, az) }
        }
        let az = best.az
        let ahead = east * sin(az) + north * cos(az), side = east * cos(az) - north * sin(az)
        let pitch = 0.11
        let forward = ahead * cos(pitch) + zenith * sin(pitch), up = zenith * cos(pitch) - ahead * sin(pitch)
        let names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
        let facing = names[Int((best.az / (.pi / 4)).rounded()) % 8]
        let note = "the Milky Way is drawn as it stands now over \(place.place), looking \(facing)"
        func f(_ v: SIMD3<Double>) -> SIMD3<Float> { SIMD3(Float(v.x), Float(v.y), Float(v.z)) }
        return (f(side), f(forward), f(up), note)
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
        // Broadleaf crowns at a second, larger scale, and twig-and-leaf detail along the whole edge.
        float cw2 = 0.19;
        float id2 = floor(x / cw2);
        for (int k = -1; k <= 1; k++) {
            float c = id2 + float(k);
            float cx = (c + 0.1 + 0.8 * hash11(c * 2.91 + seed)) * cw2;
            float rw = cw2 * (0.55 + 0.7 * hash11(c * 6.7 + seed));
            float u = (x - cx) / rw;
            if (abs(u) < 1) top = max(top, HORIZON + 0.06 + 0.12 * hash11(c * 4.3 + seed) + 0.07 * pow(1 - u * u, 0.7) * (0.55 + 0.9 * fbm(float2(x * 16 + c, 1.3), 3)));
        }
        top += 0.024 * (fbm(float2(x * 70, 2.9), 4) - 0.5) + 0.008 * (noise(float2(x * 380, 5.1)) - 0.5);
        return top;
    }

    inline float3 canopyEdge(float2 p, float seed, float aspect, thread float &leaf) {
        float x = p.x / aspect;
        float ramp = saturate(0.5 * max(p.y, 0.0) + 0.55 * x * x + 0.1 * abs(x));
        float cl = fbm(float3(p * float2(2.4, 2.8) + seed, 1.3), 5) + 0.5 * (fbm(float3(p * 26.0, 4.0), 3) - 0.5);
        float lobe = fbm(float3(p * 7.0, 9.0), 3);
        float edge = cl * 0.8 + ramp * 1.1 - 0.95 + 0.12 * (lobe - 0.5);
        leaf = smoothstep(0.0, 0.03, edge);
        return float3(edge, ramp, lobe);
    }

    fragment float4 meadow(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> milky [[texture(0)]]) {
        float2 p = centered(in.uv, u.res);
        float seed = u.a.w * 0.01;
        float k = 0.9;
        // Sky: deep blue fading to a faint glow at the horizon.
        float3 c = mix(float3(0.010, 0.016, 0.040), float3(0.0012, 0.0018, 0.0065), smoothstep(HORIZON, 1.0, p.y));
        c += float3(0.016, 0.015, 0.022) * exp(-max(p.y - HORIZON, 0.0) * 5);
        if (u.a.y > 0.5) {
            float3 d = normalize(u.c.xyz + p.x * k * u.b.xyz + p.y * k * u.d.xyz);
            float spread = k * 2.0 / u.res.y;
            float3 mw = milkyWay(milky, d, spread * 2.0);
            // The photograph is display-referred; bring it down to what a moonless night puts on a camera sensor, and
            // let the horizon's airglow and haze swallow its lowest part.
            c += pow(mw, float3(1.15)) * 0.16 * smoothstep(HORIZON - 0.02, HORIZON + 0.35, p.y) * float3(1.02, 1.0, 0.95);
        }
        float2 sp = p * 140;
        float2 sc = floor(sp);
        float star = step(0.992, hash12(sc + 3.1)) * smoothstep(0.3, 0.0, length(fract(sp) - hash22(sc) * 0.6 - 0.2));
        c += float3(0.6, 0.65, 0.8) * star * hash12(sc) * 0.25 * smoothstep(0.1, 0.5, p.y);
        float albedo = 0;
        // Far and near tree lines, the far one paler in the mist.
        float topFar = HORIZON + (treeline(p.x * 1.6 + 17.0, seed + 3.3) - HORIZON) * 0.62;
        float far = smoothstep(0.003, -0.002, p.y - topFar);
        float haze = exp(-max(p.y - HORIZON, 0.0) * 9);
        float3 mist = float3(0.011, 0.015, 0.022);
        c = mix(c, mix(float3(0.0016, 0.0024, 0.0036), mist, 0.55 + 0.4 * haze), far);
        float topNear = treeline(p.x, seed);
        float depthIn = topNear - p.y;
        float pin = smoothstep(0.55, 0.75, noise(float2(p.x * 230, p.y * 230))) * smoothstep(0.045, 0.0, depthIn);
        float trees = smoothstep(0.003, -0.003, p.y - topNear) * (1 - 0.7 * pin);
        float bark = 0.7 + 0.6 * fbm(float3(p.x * 60, p.y * 30, 2.0), 3);
        float tcell = floor(p.x / 0.06 + seed * 3.0);
        float tx = (tcell + 0.5 + 0.35 * (hash11(tcell * 1.7) - 0.5)) * 0.06 - seed * 0.18;
        float trunkM = step(0.7, hash11(tcell * 3.3 + 1.0)) * smoothstep(0.005 + 0.007 * hash11(tcell), 0.0025, abs(p.x - tx - 0.02 * (p.y + 0.1)));
        float hz = haze * 0.8 * (1 - 0.88 * trunkM);
        c = mix(c, mix(float3(0.0007, 0.0011, 0.0015) * bark, mist * 0.7, hz), trees);
        if (p.y < HORIZON + 0.01) {
            float fr = smoothstep(-0.7, HORIZON, p.y);
            float3 g = mix(float3(0.0015, 0.0028, 0.0020), float3(0.005, 0.008, 0.009), fr * fr);
            float clumps = fbm(float2(p.x * 3, p.y * 14) / max(HORIZON + 0.02 - p.y, 0.05), 4);
            g *= 0.7 + 0.6 * clumps;
            float m = smoothstep(HORIZON + 0.01, HORIZON - 0.005, p.y);
            c = mix(c, g, m);
            albedo = m * (0.5 + 0.7 * clumps);
        }
        // Mist lying on the meadow and between the trunks.
        float sheet = exp(-pow((p.y - HORIZON + 0.03) / 0.09, 2.0)) * (0.6 + 0.8 * fbm(float3(p.x * 2.3, p.y * 9, 7.0), 4));
        c += float3(0.006, 0.0085, 0.0125) * sheet;
        // Overhanging canopy and the trunks beside the frame: near black, with moonlight on the leaf edges.
        float leaf;
        float3 e = canopyEdge(p, seed, u.a.x, leaf);
        float rim = smoothstep(0.07, 0.0, e.x) * leaf;
        float3 foliage = float3(0.0006, 0.0010, 0.0013) + float3(0.012, 0.017, 0.026) * rim * (0.4 + e.z);
        c = mix(c, foliage, leaf);
        float x = p.x / u.a.x;
        for (int t = 0; t < 2; t++) {
            float side = t == 0 ? -1.0 : 1.0;
            float cx = side * (0.93 - 0.04 * hash11(seed * 3.0 + float(t)) + 0.03 * sin(p.y * 2.0 + float(t)));
            float w = 0.045 + 0.02 * smoothstep(0.0, -1.0, p.y);
            float trunk = smoothstep(w, w - 0.006, abs(x - cx));
            float lit = smoothstep(0.0, 0.03, (x - cx) * -side + w * 0.9);
            float3 tc = float3(0.0008, 0.0010, 0.0012) * (0.6 + 0.6 * fbm(float3(p.y * 3, x * 80, 1.0), 3))
                      + float3(0.008, 0.011, 0.016) * pow(saturate(1 - abs((x - cx) * -side * 1.0 + w * 0.7) / (w * 0.5)), 3);
            c = mix(c, tc, trunk * smoothstep(-0.9, -0.5, p.y + 0.4));
        }
        return float4(c, albedo * (1 - trees) * (1 - leaf));
    }

    // Pools of light each firefly throws on the grass beneath it, foreshortened on the ground, into a quarter-size
    // texture. Fireflies higher up light the grass more faintly.
    struct L { float4 pos [[position]]; float2 q; float3 color; };

    vertex L poolVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                        const device float4 *flies [[buffer(1)]]) {
        float4 f = flies[3 * iid];
        float h = flies[3 * iid + 1].y;
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

    // Each firefly, as a long exposure records it: a faint, slightly curved streak along the path it flew during the
    // flash, brightest mid-way and fading at both ends; close to the lens a wide, soft bokeh disc; far away a tiny point.
    // Every firefly has its own brightness. The path is three points (now, a fifth of a second ago, and further back).
    struct G { float4 pos [[position]]; float2 q; float4 color; float radius; float soft;
               float2 a [[flat]]; float2 b [[flat]]; float2 c [[flat]]; };

    vertex G flyVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]],
                       const device float4 *flies [[buffer(1)]]) {
        float4 f = flies[3 * iid];
        float4 g = flies[3 * iid + 1];
        float2 mid = flies[3 * iid + 2].xy;
        float tint = g.x;
        float d = f.z;
        float px = u.res.y * 0.5;
        float blur = max(1 / d - 1 / 4.5, 0.0);
        float bokeh = blur * 0.3 * px;
        float soft = smoothstep(4.0, 14.0, bokeh);
        float vary = 0.18 + 1.1 * pow(fract(tint * 7.31 + 0.37), 2.0);
        bool lit = f.w > 0.002;
        float radius = max(lit ? max(bokeh, 9.0) : 0.0, 1.0);
        // Bokeh discs don't streak; the path's bend is exaggerated a little, as a wobbling flight is.
        float2 a = f.xy * px, c = mix(g.zw * px, a, soft);
        float2 b = mix(mid * px, a, soft);
        b += (b - (a + c) * 0.5) * 0.3;
        float2 lo = lit ? min(min(a, b), c) - radius : a, hi = lit ? max(max(a, b), c) + radius : a;
        const float2 corners[6] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1)};
        float2 k = corners[vid];
        float2 pos = mix(lo, hi, k) / px;
        G o;
        o.pos = float4(pos.x / u.a.x, pos.y, 0, 1);
        o.q = mix(lo, hi, k);
        float3 warm = mix(float3(0.70, 1.0, 0.20), float3(1.0, 0.78, 0.20), tint * tint);
        o.color = float4(warm * f.w * vary * clamp(9 / d, 0.35, 1.5), 0);
        o.radius = radius;
        o.soft = soft;
        o.a = a; o.b = b; o.c = c;
        return o;
    }

    // Distance from p to the segment a-b, and how far along it the nearest point is (0...1).
    inline float2 segmentDistance(float2 p, float2 a, float2 b) {
        float2 ab = b - a;
        float t = saturate(dot(p - a, ab) / max(dot(ab, ab), 1e-4));
        return float2(length(p - a - ab * t), t);
    }

    fragment float4 bokeh(G in [[stage_in]]) {
        float2 s1 = segmentDistance(in.q, in.a, in.b), s2 = segmentDistance(in.q, in.b, in.c);
        bool second = s2.x < s1.x;
        float r = second ? s2.x : s1.x;
        float along = second ? 0.5 + 0.5 * s2.y : 0.5 * s1.y;
        float taper = sin(3.14159 * (0.08 + 0.84 * along));
        float len = length(in.a - in.b) + length(in.b - in.c);
        float spread = 1 / (1 + len * 0.06);
        float point = (1.4 * exp(-r * r / 2.4) + 0.06 * exp(-r * r / 30)) * mix(0.35, 1.0, taper) * spread * mix(1.0, 3.0, saturate(1 - len * 0.2));
        float edge = r / in.radius;
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
    """
}
