import Foundation
import Metal
import simd

/// A Schwarzschild black hole with a thin accretion disk, ray traced along null geodesics. Every pixel integrates
/// the photon orbit equation u'' + u = 3/2 u² (u = r_s / r) in its own plane, so the far side of the disk arches
/// over the shadow, a secondary image hugs it from below, a photon ring circles it and the stars behind are lensed.
/// The disk has the Page–Thorne temperature profile from the innermost stable orbit outward, Doppler beamed and
/// gravitationally redshifted, and a hot spot orbits in it now and then. Lengths are in Schwarzschild radii.
@MainActor final class BlackHole: Scene {
    static let observer: Float = 15
    static let isco: Float = 3
    static let outer: Float = 9
    /// Light-crossing times of one Schwarzschild radius per second of screen time, so the inner disk turns in about 40 s.
    static let clock: Float = 1.15

    private let trace: MTLRenderPipelineState
    private let present: MTLRenderPipelineState
    private let bloom: Bloom
    private var frames: [MTLTexture] = []
    private let peak: Float
    /// The hot spot: a blob of hotter gas on a Keplerian orbit, like the flares seen circling Sagittarius A*.
    private var spotRadius: Float = 4.5
    private var spotAngle: Float = 0
    private var spotBorn: Float = 0
    private var spotLife: Float = 0
    private var spotNext: Float = 6
    private var spotLevel: Float = 0
    private var spotShift: Float = 1
    private var elevation: Float = 0.2

    var scale: Float { 0.5 }
    var facts: String? {
        let degrees = Int((elevation * 180 / .pi).rounded())
        var s = "Schwarzschild black hole seen from 15 Schwarzschild radii, \(degrees) degrees above its disk. "
            + "The disk's inner edge, at the innermost stable orbit 3 radii out, moves at half the speed of light."
        if spotLevel > 0.3 {
            let r = String(format: "%.1f", spotRadius)
            let percent = Int((abs(spotShift - 1) * 100).rounded())
            s += spotShift >= 1 ? " A hot spot at \(r) radii is swinging toward us, its light \(percent) percent bluer."
                                : " A hot spot at \(r) radii is moving away, its light \(percent) percent redder."
        }
        return s
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        trace = try gpu.pipeline(library, "bhTrace")
        present = try gpu.pipeline(library, "bhPresent")
        bloom = try Bloom(gpu: gpu)
        peak = stride(from: Self.isco, through: Self.outer, by: 0.01).map(Self.flux).max() ?? 1
    }

    /// Page–Thorne flux from a thin disk around a Schwarzschild hole, zero at the innermost stable orbit (r = 3).
    nonisolated static func flux(_ r: Float) -> Float {
        guard r > 3 else { return 0 }
        let s = r.squareRoot(), a = Float(1.5).squareRoot(), b = Float(3).squareRoot()
        let l = log((s + a) * (b - a) / ((s - a) * (b + a)))
        return max(s - b + 0.5 * a * l, 0) / ((r - 1.5) * r * r * s)
    }

    /// Orbital speed in units of c, as a hovering observer at radius r measures it.
    nonisolated static func orbitalSpeed(_ r: Float) -> Float { (0.5 / (r - 1)).squareRoot() }

    func draw(_ f: Frame) {
        let w = f.target.width, h = f.target.height
        if frames.first?.width != w || frames.first?.height != h { frames = [f.gpu.texture(w, h), f.gpu.texture(w, h)] }
        let t = f.time
        let tau = t * Self.clock

        let (azimuth, roll, shift) = camera(t, f.seed)
        updateSpot(azimuth, f)

        let jitter = SIMD2<Float>(FieldKit.halton(Int(f.index) % 64 + 1, 2), FieldKit.halton(Int(f.index) % 64 + 1, 3)) - 0.5
        let blend: Float = f.index == 0 ? 1 : 1 - exp(-f.dt / 0.12)
        let now = frames[Int(f.index & 1)], past = frames[Int((f.index + 1) & 1)]
        let u = f.uniforms([azimuth, elevation, roll, 1.7], [shift.x, shift.y, jitter.x, jitter.y], [tau, 1, blend, 1 / peak],
                           [spotRadius, spotAngle, spotLevel, 0])
        f.cmd.draw(trace, into: now, uniforms: u, textures: [past])
        let glow = bloom.glow(f.cmd, now)
        f.cmd.draw(present, into: f.target, uniforms: f.uniforms([0.9, 0, 0, 0]), textures: [now, glow])

        if let synth = f.sound { play(synth, azimuth) }
    }

    /// A slow orbit around the hole while the view tilts between nearly edge-on and 16 degrees above the disk,
    /// and the framing drifts so nothing bright sits on one spot of the screen. `--variant n` starts at n × 45°.
    private func camera(_ t: Float, _ seed: Float) -> (Float, Float, SIMD2<Float>) {
        let phase = Snapshot.variant.map { Float($0) * .pi / 4 } ?? seed
        elevation = (9.5 + 6.5 * sin(t * 2 * .pi / 260 + phase * 3)) * .pi / 180
        let roll = 0.12 * sin(t * 2 * .pi / 350 + 1.3 * phase)
        let shift = SIMD2<Float>(0.16 * sin(t * 2 * .pi / 430 + phase * 5), 0.06 * sin(t * 2 * .pi / 310 + 2 * phase))
        return (phase + t * 2 * .pi / 640, roll, shift)
    }

    /// Spawns, ages and moves the hot spot, and works out how blue- or redshifted it looks from the camera.
    private func updateSpot(_ azimuth: Float, _ f: Frame) {
        let t = f.time
        if spotLife == 0 || t > spotBorn + spotLife + spotNext {
            spotRadius = .random(in: 3.6...5.8)
            spotAngle = .random(in: 0..<(2 * .pi))
            spotBorn = t
            let period = 2 * Float.pi / (0.5 / pow(spotRadius, 3)).squareRoot() / Self.clock
            spotLife = period * .random(in: 2.2...3.5)
            spotNext = .random(in: 10...40)
        }
        let omega = (0.5 / pow(spotRadius, 3)).squareRoot()
        spotAngle += omega * f.dt * Self.clock
        let age = (t - spotBorn) / spotLife
        spotLevel = age < 1 ? pow(sin(.pi * min(max(age, 0), 1)), 2) : 0

        // The Doppler factor of light sent from the spot toward the camera, with the gravitational shifts at both ends.
        let eye = Self.observer * SIMD3<Float>(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth))
        let q = spotRadius * SIMD3<Float>(sin(spotAngle), 0, cos(spotAngle))
        let toward = simd_normalize(eye - q)
        let v = Self.orbitalSpeed(spotRadius)
        let along = simd_dot(SIMD3<Float>(cos(spotAngle), 0, -sin(spotAngle)), toward)
        spotShift = ((1 - 1 / spotRadius) / (1 - 1 / Self.observer)).squareRoot() * (1 - v * v).squareRoot() / (1 - v * along)
    }

    /// A deep rumble and a drone on A. The hot spot sings a quiet tone whose pitch follows its Doppler shift, and the
    /// chord swells while it swings toward us.
    private func play(_ synth: Synth, _ azimuth: Float) {
        let swell = spotLevel * min(max((spotShift - 0.95) / 0.5, 0), 1)
        let edge = 1 - min(max((elevation * 180 / .pi - 3) / 17, 0), 1)
        synth.noise(0.09 + 0.05 * swell, cutoff: 95 + 70 * swell)
        synth.drone(0, 55, 0.075, pan: -0.15)
        synth.drone(1, 82.5, 0.045 + 0.01 * edge, pan: 0.15)
        synth.drone(2, 110.3, 0.02 + 0.03 * swell, pan: -0.3)
        synth.drone(3, 164.6, 0.01 + 0.025 * swell, pan: 0.3)
        synth.drone(4, 220 * min(max(spotShift, 0.5), 2), 0.022 * spotLevel, pan: sin(spotAngle - azimuth) * 0.6)
        synth.drone(5, 277.2, 0.012 * swell, pan: 0.1)
    }

    private static let source = FieldKit.stars + """
    constant float R0 = 15.0;
    constant float RIN = 3.0;
    constant float ROUT = 9.0;

    inline float kepler(float r) { return sqrt(0.5 / (r * r * r)); }

    // Page–Thorne flux, zero at the innermost stable orbit; lengths in Schwarzschild radii.
    inline float pageThorne(float r) {
        float s = sqrt(r), a = sqrt(1.5), b = sqrt(3.0);
        float l = log((s + a) * (b - a) / max((s - a) * (b + a), 1e-6));
        return max(s - b + 0.5 * a * l, 0.0) / ((r - 1.5) * r * r * s);
    }

    // One RK4 step of the photon orbit equation u'' = 3/2 u^2 - u, with the orbit angle as the parameter.
    inline float2 orbitStep(float2 s, float h) {
        float2 k1 = float2(s.y, 1.5 * s.x * s.x - s.x);
        float2 q = s + 0.5 * h * k1;
        float2 k2 = float2(q.y, 1.5 * q.x * q.x - q.x);
        q = s + 0.5 * h * k2;
        float2 k3 = float2(q.y, 1.5 * q.x * q.x - q.x);
        q = s + h * k3;
        float2 k4 = float2(q.y, 1.5 * q.x * q.x - q.x);
        return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    // Gas in the disk: the Page–Thorne glow, broken into streaks that shear with Keplerian rotation. Two copies of the
    // pattern take turns, each fading out before the shear winds it too tight. A real disk around a stellar black hole
    // peaks in X-rays; this one is scaled so its hottest ring is 5,900 K and its colours stay visible. The redshift g
    // scales that temperature, and the observed brightness goes as g^4. Returns emission and opacity.
    inline float4 diskLight(float r, float th, float g, constant U &u) {
        float tau = u.c.x;
        float om = kepler(r);
        float lr = log(r);
        float n = 0;
        for (int k = 0; k < 2; k++) {
            float cycle = tau / 70 + 0.5 * k;
            float ph = fract(cycle);
            float a = th - om * ph * 70;
            float3 q = float3(cos(a) * 2.2, sin(a) * 2.2, lr * 7.5) + hash11(floor(cycle) * 1.37 + k * 5.1) * 40;
            float s = sin(PI * ph);
            n += s * s * fbm(q, 4);
        }
        float streak = smoothstep(0.22, 0.78, n);
        float flux = pageThorne(r) * u.c.w;
        float dth = atan2(sin(th - u.d.y), cos(th - u.d.y));
        float spot = u.d.z * exp(-(r - u.d.x) * (r - u.d.x) / 0.16 - (dth * r) * (dth * r) / 1.1);
        float edge = smoothstep(ROUT, ROUT - 3.5, r);
        float temp = 5900 * pow(flux * (1 + 2.5 * spot), 0.25) * g;
        float bright = flux * (0.12 + 1.5 * streak * streak + 6 * spot) * pow(g, 4.0) * edge;
        float alpha = clamp((0.55 + 0.45 * streak) * mix(1.0, 0.45, smoothstep(4.0, ROUT, r)) * edge, 0.0, 1.0);
        float3 tint = blackbody(temp);
        tint = mix(float3(dot(tint, float3(0.3, 0.55, 0.15))), tint, 1.8);
        return float4(max(tint, 0.0) * bright * 0.5, alpha);
    }

    // A distant spiral galaxy, a few degrees across, that the hole lenses into arcs when it passes behind.
    inline float3 galaxy(float3 d) {
        float3 G = normalize(float3(0.62, -0.17, -0.77));
        if (dot(d, G) < 0.985) return 0;
        float3 t1 = normalize(cross(G, float3(0, 1, 0))), t2 = cross(t1, G);
        float2 q = rot(0.7) * (float2(dot(d, t1), dot(d, t2)) / 0.045);
        q.y /= 0.5;
        float r = length(q), ang = atan2(q.y, q.x);
        float arms = pow(0.5 + 0.5 * cos(2 * (ang - 2.6 * log(r + 0.08))), 3.0);
        float dust = 1 - 0.6 * smoothstep(0.55, 0.9, noise(q * 9 + 3.1)) * smoothstep(0.1, 0.4, r);
        float disk = exp(-r * 3.2) * (0.2 + arms * (0.6 + 0.6 * noise(q * 14))) * dust;
        float bulge = exp(-r * r * 30);
        return float3(0.55, 0.65, 1.0) * disk * 0.7 + float3(1.0, 0.78, 0.5) * bulge * 1.2;
    }

    // The sky in direction d: a faint Milky Way band with dust lanes, a galaxy and two layers of stars.
    inline float3 sky(float3 d, float fp) {
        float3 N = normalize(float3(0.25, 0.92, 0.3));
        float lat = dot(d, N);
        float band = exp(-lat * lat / 0.035);
        float3 core = normalize(float3(-0.8, 0.1, 0.55));
        float bulge = pow(max(dot(d, core), 0.0), 6.0);
        float cloud = fbm(d * 5 + 2.3, 4);
        float dust = smoothstep(0.45, 0.7, fbm(d * 11 + 7.1, 4)) * exp(-lat * lat / 0.004);
        float3 mw = (float3(0.45, 0.55, 0.8) * band * (0.3 + cloud * cloud * 1.6) + float3(0.9, 0.75, 0.6) * bulge * band * 1.5) * (1 - 0.7 * dust);
        float3 c = mw * 0.03 + float3(0.0010, 0.0012, 0.0020);
        c += galaxy(d);
        c += starLayer(d, 300, 0.07, fp, 0.8, 1.7, 0.0021) * (1 + 2 * band);
        c += starLayer(d, 90, 0.045, fp, 9.0, 4.2, 0.0021);
        return c;
    }

    fragment float4 bhTrace(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> past [[texture(0)]]) {
        float2 p = centered(in.uv + u.b.zw / u.res, u.res) - u.b.xy;
        p = rot(u.a.z) * p;
        float az = u.a.x, el = u.a.y;
        float3 eye = R0 * float3(cos(el) * sin(az), sin(el), cos(el) * cos(az));
        float3 d = ray(p, eye, float3(0), u.a.w);

        // The photon's orbit lies in the plane of e1 (out from the hole to the camera) and e2.
        float3 e1 = eye / R0;
        float ca = dot(d, e1);
        float3 w = d - ca * e1;
        float sa = max(length(w), 1e-5);
        float3 e2 = w / sa;
        // The camera hovers at R0, so its local angle to the hole gives the impact parameter b and the starting slope
        // du/dphi through the lapse factor sqrt(1 - r_s / R0).
        float u0 = 1 / R0, lapse = sqrt(1 - u0);
        float b = R0 * sa / lapse;
        float2 s = float2(u0, -u0 * lapse * ca / sa);
        float na = e1.y, nb = e2.y;
        float spin = cross(e1, e2).y;
        float phi = 0, side = na;
        float3 col = 0;
        float trans = 1;
        bool escaped = false;
        for (int i = 0; i < 420; i++) {
            float h = min(0.11 / (1 + 7 * s.x), 0.12 * s.x / (abs(s.y) + 1e-4));
            float2 n = orbitStep(s, h);
            float phi1 = phi + h;
            float side1 = na * cos(phi1) + nb * sin(phi1);
            if (side * side1 < 0) {
                // Crossed the disk plane: find the radius there by Hermite interpolation of u.
                float t = side / (side - side1), t2 = t * t, t3 = t2 * t;
                float uc = (2 * t3 - 3 * t2 + 1) * s.x + (t3 - 2 * t2 + t) * h * s.y + (3 * t2 - 2 * t3) * n.x + (t3 - t2) * h * n.y;
                float r = 1 / uc;
                if (r > RIN && r < ROUT) {
                    float pc = phi + t * h;
                    float3 P = cos(pc) * e1 + sin(pc) * e2;
                    // Observed over emitted frequency for gas on a circular orbit: time dilation of the orbit, the
                    // camera's own gravitational blueshift, and the Doppler term from the photon's angular momentum.
                    float g = sqrt(1 - 1.5 / r) / (lapse * (1 + kepler(r) * b * spin));
                    float4 e = diskLight(r, atan2(P.x, P.z), g, u);
                    col += trans * e.rgb;
                    trans *= 1 - e.a;
                    if (trans < 0.02) break;
                }
            }
            s = n; phi = phi1; side = side1;
            if (s.x > 1) break;
            if (s.x < 0.03 && s.y < 0) { escaped = true; break; }
        }
        // Where the light came from: the asymptote of the outgoing ray.
        float out = phi + atan2(s.x, -s.y);
        float3 D = cos(out) * e1 + sin(out) * e2;
        float fp = sqrt(length(cross(dfdx(D), dfdy(D))));
        if (escaped) col += trans * sky(D, min(fp, 0.05));
        float3 prev = past.read(uint2(in.pos.xy)).rgb;
        return float4(mix(prev, col, u.c.z), 1);
    }

    fragment float4 bhPresent(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> hdr [[texture(0)]],
                              texture2d<float> glow [[texture(1)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float3 c = hdr.sample(s, in.uv).rgb + glow.sample(s, in.uv).rgb * 0.45;
        float2 v = in.uv - 0.5;
        c *= 1 - 0.35 * dot(v, v);
        return float4(film(c * u.a.x), 1);
    }
    """
}
