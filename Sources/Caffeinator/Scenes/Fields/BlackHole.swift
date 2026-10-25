import Foundation
import Metal
import simd

/// A Schwarzschild black hole with a thin accretion disk, ray traced along null geodesics in front of the real sky.
/// Every pixel integrates the photon orbit equation u'' + u = 3/2 u² (u = r_s / r) in its own plane, so the far side
/// of the disk arches over the shadow, a secondary image hugs it from below, a thin photon ring circles it, and ESO's
/// photograph of the Milky Way behind is lensed into arcs and an Einstein ring. The disk follows the Page–Thorne
/// thin-disk profile (T ∝ r^-3/4 away from the inner edge, falling to nothing at the innermost stable orbit), its
/// gas churned into sheared filaments; each ray sees a black body at the temperature its redshift g makes of the
/// gas, so the approaching side is brighter and bluer and the receding side dim and orange. Lengths in Schwarzschild radii.
@MainActor final class BlackHole: Scene {
    static let observer: Float = 30
    static let isco: Float = 3
    /// Light-crossing times of one Schwarzschild radius per second of screen time, so the inner disk turns in about a minute.
    static let clock: Float = 1.15
    /// The hottest ring of the disk, kelvin.
    static let hottest: Float = 6000
    /// The hole's own axes in galactic coordinates: its disk is tipped 30 degrees to the plane of the Milky Way, so the
    /// camera's orbit keeps the band in view, and the camera starts out looking toward the galactic centre.
    static let axes: [SIMD3<Double>] = [[0, -0.866, 0.5], [0, 0.5, 0.866], [-1, 0, 0]]

    private let trace: MTLRenderPipelineState
    private let lens: Lens
    private var frames: [MTLTexture] = []
    private let placeholder: MTLTexture
    private var sky: MTLTexture?
    private var loaded = false
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
    private var azimuth: Float = 0

    var scale: Float { 0.5 }
    var ready: Bool { loaded }
    var credit: String? { sky == nil ? nil : MilkyWay.credit }
    var facts: String? {
        let degrees = Int((elevation * 180 / .pi).rounded())
        var s = "Schwarzschild black hole seen from 30 Schwarzschild radii, \(degrees) degrees above its disk. "
            + "The disk's inner edge, at the innermost stable orbit 3 radii out, moves at half the speed of light; "
            + "its hottest ring glows at 6,000 kelvin."
        if spotLevel > 0.3 {
            let r = String(format: "%.1f", spotRadius)
            let percent = Int((abs(spotShift - 1) * 100).rounded())
            s += spotShift >= 1 ? " A hot spot at \(r) radii is swinging toward us, its light \(percent) percent bluer."
                                : " A hot spot at \(r) radii is moving away, its light \(percent) percent redder."
        }
        if sky != nil { s += " Behind it, lensed: " + behind() + "." }
        else if loaded { s += " The Milky Way photograph is unavailable offline, so the stars behind are drawn." }
        return s
    }

    init(gpu: GPU) throws {
        let library = try gpu.library(Self.source)
        trace = try gpu.pipeline(library, "bhTrace")
        lens = try Lens(gpu: gpu)
        var black: UInt32 = 0
        placeholder = gpu.texture(1, 1, .rgba8Unorm, bytes: &black, bytesPerRow: 4)
        peak = stride(from: Self.isco, through: 12, by: 0.01).map(Self.flux).max() ?? 1
        Task { @MainActor in
            sky = await MilkyWay.texture(gpu)
            loaded = true
        }
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

        let (roll, shift) = camera(t, f.seed)
        updateSpot(f)

        let jitter = SIMD2<Float>(FieldKit.halton(Int(f.index) % 64 + 1, 2), FieldKit.halton(Int(f.index) % 64 + 1, 3)) - 0.5
        let blend: Float = f.index == 0 ? 1 : 1 - exp(-f.dt / 0.12)
        let now = frames[Int(f.index & 1)], past = frames[Int((f.index + 1) & 1)]
        let u = f.uniforms([azimuth, elevation, roll, 2.6], [shift.x, shift.y, jitter.x, jitter.y],
                           [tau, Self.hottest, blend, 1 / peak], [spotRadius, spotAngle, spotLevel, sky == nil ? 0 : 1])
        f.cmd.draw(trace, into: now, uniforms: u, textures: [past, sky ?? placeholder])
        lens.develop(f, now, Lens.Look(exposure: 0.85, bloom: 0.07, halation: 0.04, vignette: 0.32, aberration: 1.0, grain: 0.35))

        if let synth = f.sound { play(synth) }
    }

    /// A slow orbit around the hole while the view tilts between nearly edge-on and 30 degrees above the disk,
    /// and the framing drifts so nothing bright sits on one spot of the screen. `--variant n` starts at n × 45°.
    private func camera(_ t: Float, _ seed: Float) -> (Float, SIMD2<Float>) {
        let phase = Snapshot.variant.map { Float($0) * .pi / 4 } ?? seed
        elevation = (16 - 12.5 * cos(t * 2 * .pi / 300 + phase * 3)) * .pi / 180
        azimuth = phase + t * 2 * .pi / 720
        let roll = 0.1 * sin(t * 2 * .pi / 350 + 1.3 * phase)
        let shift = SIMD2<Float>(0.14 * sin(t * 2 * .pi / 430 + phase * 5), 0.05 * sin(t * 2 * .pi / 310 + 2 * phase))
        return (roll, shift)
    }

    /// The constellation behind the hole, from the camera's line of sight.
    private func behind() -> String {
        let el = Double(elevation), az = Double(azimuth)
        let v = -SIMD3(cos(el) * sin(az), sin(el), cos(el) * cos(az))
        let g = Self.axes[0] * v.x + Self.axes[1] * v.y + Self.axes[2] * v.z
        let rows = SkyCatalog.galactic
        let eq = rows[0] * g.x + rows[1] * g.y + rows[2] * g.z
        let latitude = asin(min(max(g.z, -1), 1)) * 180 / .pi
        if g.x > cos(8 * Double.pi / 180) { return "the star clouds of Sagittarius, toward the centre of the Milky Way" }
        let name = SkyCatalog.constellations.max { a, b in
            simd_dot(SkyCatalog.unit(ra: Double(a.ra), dec: Double(a.dec)), eq) < simd_dot(SkyCatalog.unit(ra: Double(b.ra), dec: Double(b.dec)), eq)
        }?.name ?? "deep space"
        return abs(latitude) < 12 ? "the Milky Way in \(name)" : "the sky in \(name), far from the Milky Way's band"
    }

    /// Spawns, ages and moves the hot spot, and works out how blue- or redshifted it looks from the camera.
    private func updateSpot(_ f: Frame) {
        let t = f.time
        if spotLife == 0 || t > spotBorn + spotLife + spotNext {
            spotRadius = .random(in: 3.8...6)
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
    private func play(_ synth: Synth) {
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
    constant float R0 = 30.0;
    constant float RIN = 3.0;
    constant float ROUT = 15.0;
    // The hole's axes in galactic coordinates (columns), as in BlackHole.axes.
    constant float3x3 BH_TO_GAL = float3x3(float3(0, -0.866, 0.5), float3(0, 0.5, 0.866), float3(-1, 0, 0));

    inline float kepler(float r) { return sqrt(0.5 / (r * r * r)); }

    // Page–Thorne flux, zero at the innermost stable orbit; lengths in Schwarzschild radii.
    inline float pageThorne(float r) {
        float s = sqrt(r), a = sqrt(1.5), b = sqrt(3.0);
        float l = log((s + a) * (b - a) / max((s - a) * (b + a), 1e-6));
        return max(s - b + 0.5 * a * l, 0.0) / ((r - 1.5) * r * r * s);
    }

    // Black-body radiance relative to a 6,504 K body, which is white, at effective wavelengths (605, 510 and 415 nm)
    // fitted so that the colours match the CIE colour-matching functions in sRGB to within a few percent.
    inline float3 planck(float T) {
        const float3 lam = float3(0.605, 0.510, 0.415);
        float3 x0 = 14388.0 / (lam * 6504.0);
        float3 x = min(14388.0 / (lam * max(T, 400.0)), 80.0);
        return (exp(x0) - 1) / (exp(x) - 1);
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

    // Magnetised turbulence in the gas: clumps and finer filaments that the disk's differential rotation shears into
    // tight trailing spirals. Each scale has two copies of its pattern that take turns, each fading out before the
    // shear winds it too tight.
    inline float sheared(float r, float th, float tau, float life, float around, float out, float salt) {
        float om = kepler(r);
        float lr = log(r);
        float n = 0;
        for (int k = 0; k < 2; k++) {
            float cycle = tau / life + 0.5 * k;
            float ph = fract(cycle);
            float a = th - om * ph * life;
            float3 q = float3(cos(a) * around, sin(a) * around, lr * out) + hash11(floor(cycle) * 1.37 + k * 5.1 + salt) * 40;
            float s = sin(PI * ph);
            n += s * s * (fbm(q, 4) - 0.5);
        }
        return n;
    }
    inline float churn(float r, float th, float tau) {
        return 0.6 * sheared(r, th, tau, 26, 1.6, 9.0, 0) + 0.8 * sheared(r, th, tau, 45, 3.2, 34.0, 3.3);
    }

    // Light from the disk where the ray crosses it at radius r and angle th, redshifted by g: a black body at g times
    // the gas's temperature, which carries both the colour shift and the beaming. Returns radiance and opacity.
    inline float4 diskLight(float r, float th, float g, constant U &u) {
        float flux = pageThorne(r) * u.c.w;
        float dth = atan2(sin(th - u.d.y), cos(th - u.d.y));
        float spot = u.d.z * exp(-(r - u.d.x) * (r - u.d.x) / 0.2 - (dth * r) * (dth * r) / 1.4);
        float n = churn(r, th, u.c.x);
        flux *= exp(2.2 * n) * (1 + 2.5 * spot);
        float temp = u.c.y * pow(max(flux, 0.0), 0.25);
        float edge = smoothstep(ROUT, 6.0, r);
        float3 light = planck(g * temp) * 0.8 * edge;
        float alpha = smoothstep(ROUT, ROUT - 6.0, r);
        return float4(light, alpha);
    }

    // The fallback sky while the photograph loads, or offline: two layers of stars and a faint band.
    inline float3 drawnSky(float3 d, float fp) {
        float3 N = normalize(float3(0.25, 0.92, 0.3));
        float lat = dot(d, N);
        float band = exp(-lat * lat / 0.035);
        float3 c = float3(0.45, 0.5, 0.6) * band * (0.3 + fbm(d * 5 + 2.3, 4)) * 0.02;
        c += starLayer(d, 300, 0.07, fp, 0.8, 1.7, 0.0021) * (1 + 2 * band);
        c += starLayer(d, 90, 0.045, fp, 9.0, 4.2, 0.0021);
        return c;
    }

    fragment float4 bhTrace(V in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> past [[texture(0)]],
                            texture2d<float> photo [[texture(1)]]) {
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
        for (int i = 0; i < 480; i++) {
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
            if (s.x < 0.5 * u0 && s.y < 0) { escaped = true; break; }
        }
        // Where the light came from: the asymptote of the outgoing ray.
        float out = phi + atan2(s.x, -s.y);
        float3 D = cos(out) * e1 + sin(out) * e2;
        float fp = sqrt(length(cross(dfdx(D), dfdy(D))));
        if (escaped) {
            float3 bg = u.d.w > 0.5 ? milkyWay(photo, transpose(EQ_TO_GAL) * (BH_TO_GAL * D), clamp(fp, 1e-5, 0.5)) * 0.45
                                    : drawnSky(D, min(fp, 0.05));
            col += trans * bg;
        }
        float3 prev = past.read(uint2(in.pos.xy)).rgb;
        return float4(mix(prev, col, u.c.z), 1);
    }
    """
}
