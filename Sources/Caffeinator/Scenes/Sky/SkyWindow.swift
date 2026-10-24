import AppKit
import Metal
import simd

/// The sky over the Mac right now, seen through a window: physically based sky colour for the real Sun, clouds,
/// rain, snow and mist from the current weather, drifting with the real wind, and at night the real stars, planets
/// and Moon. Rooftops, trees and hills along the horizon.
@MainActor final class SkyWindow: Scene {
    /// The current weather from Open-Meteo.
    struct Weather: Sendable {
        var temperature = 15.0
        var cloud = 0.0, low = 0.0, mid = 0.0, high = 0.0
        var precipitation = 0.0, rain = 0.0, snowfall = 0.0
        var code = 0
        var wind = 0.0, windFrom = 270.0
        var visibility = 24000.0
        var humidity = 60.0

        var rainIntensity: Double {
            let base: Double = switch code {
            case 51, 56: 0.2
            case 53, 61, 80: 0.4
            case 55, 57, 63, 66, 81: 0.65
            case 65, 67, 82, 95, 96, 99: 0.9
            default: 0
            }
            return min(1, max(base, rain > 0 ? min(rain / 3, 1) * 0.9 + 0.1 : 0))
        }
        var snowIntensity: Double {
            let base: Double = switch code {
            case 77: 0.3
            case 71, 85: 0.4
            case 73: 0.65
            case 75, 86: 0.9
            default: 0
            }
            return min(1, max(base, snowfall > 0 ? min(snowfall / 1.5, 1) * 0.9 + 0.1 : 0))
        }
        var fog: Double { code == 45 || code == 48 ? 1 : max(0, min(1, 1 - visibility / 6000)) * 0.8 }
        var thunder: Bool { code >= 95 }

        var summary: String {
            switch code {
            case 0: "clear"
            case 1: "mostly clear"
            case 2: "partly cloudy"
            case 3: "overcast"
            case 45: "foggy"
            case 48: "freezing fog"
            case 51, 53, 55: "drizzle"
            case 56, 57: "freezing drizzle"
            case 61: "light rain"
            case 63: "rain"
            case 65: "heavy rain"
            case 66, 67: "freezing rain"
            case 71: "light snow"
            case 73: "snow"
            case 75: "heavy snow"
            case 77: "snow grains"
            case 80: "rain showers"
            case 81: "heavy showers"
            case 82: "violent showers"
            case 85, 86: "snow showers"
            case 95: "a thunderstorm"
            case 96, 99: "a thunderstorm with hail"
            default: cloud > 0.7 ? "cloudy" : "clear"
            }
        }

        init() {}

        init?(json: Any?) {
            guard let root = json as? [String: Any], let c = root["current"] as? [String: Any] else { return nil }
            func num(_ key: String) -> Double? { (c[key] as? NSNumber)?.doubleValue }
            guard let t = num("temperature_2m") else { return nil }
            temperature = t
            cloud = (num("cloud_cover") ?? 0) / 100
            low = (num("cloud_cover_low") ?? cloud * 100) / 100
            mid = (num("cloud_cover_mid") ?? cloud * 60) / 100
            high = (num("cloud_cover_high") ?? cloud * 40) / 100
            precipitation = num("precipitation") ?? 0
            rain = num("rain") ?? 0
            snowfall = num("snowfall") ?? 0
            code = Int(num("weather_code") ?? 0)
            wind = num("wind_speed_10m") ?? 0
            windFrom = num("wind_direction_10m") ?? 270
            visibility = num("visibility") ?? 24000
            humidity = num("relative_humidity_2m") ?? 60
        }
    }

    /// Fixed looks for `--variant n`, to preview weather and light the real sky may not have today.
    private struct Preset {
        let name: String
        /// Find the moment when the Sun is at this altitude, rising or setting; nil means the Sun's highest point.
        let sun: Double?
        let rising: Bool
        /// True to wait for the Moon to be up, false for it to be down, nil to not mind.
        let moonUp: Bool?
        let weather: Weather
    }

    private static func weather(_ code: Int, _ t: Double, low: Double = 0, mid: Double = 0, high: Double = 0, wind: Double = 10,
                                rain: Double = 0, snow: Double = 0, visibility: Double = 24000) -> Weather {
        var w = Weather()
        w.code = code; w.temperature = t; w.low = low; w.mid = mid; w.high = high; w.cloud = max(low, mid, high)
        w.wind = wind; w.windFrom = 250; w.rain = rain; w.snowfall = snow; w.precipitation = rain + snow; w.visibility = visibility
        return w
    }

    private static let presets: [Preset] = [
        Preset(name: "golden hour", sun: 4, rising: false, moonUp: nil, weather: weather(2, 24, low: 0.25, mid: 0.2, high: 0.35, wind: 12)),
        Preset(name: "midday cumulus", sun: nil, rising: false, moonUp: nil, weather: weather(2, 29, low: 0.45, mid: 0.1, high: 0.2, wind: 16)),
        Preset(name: "blue hour", sun: -5, rising: false, moonUp: nil, weather: weather(1, 21, low: 0, mid: 0.05, high: 0.3, wind: 6)),
        Preset(name: "moonlit night", sun: -30, rising: false, moonUp: true, weather: weather(1, 22, low: 0.12, high: 0.1, wind: 5)),
        Preset(name: "rain", sun: 25, rising: true, moonUp: nil, weather: weather(63, 17, low: 1, mid: 0.8, wind: 22, rain: 2.4, visibility: 6000)),
        Preset(name: "snow", sun: 14, rising: true, moonUp: nil, weather: weather(73, -3, low: 0.95, mid: 0.6, wind: 9, snow: 1.1, visibility: 3000)),
        Preset(name: "fog", sun: 7, rising: true, moonUp: nil, weather: weather(45, 9, low: 1, wind: 3, visibility: 300)),
        Preset(name: "thunderstorm", sun: 9, rising: false, moonUp: nil, weather: weather(95, 26, low: 0.9, mid: 0.7, high: 0.5, wind: 34, rain: 5)),
        Preset(name: "dark night", sun: -35, rising: false, moonUp: false, weather: weather(0, 16, wind: 2)),
        Preset(name: "late dusk", sun: -8, rising: false, moonUp: nil, weather: weather(3, 28, high: 0.97, wind: 7, visibility: 9000)),
        Preset(name: "late dusk clear", sun: -8, rising: false, moonUp: nil, weather: weather(0, 28, wind: 7)),
        Preset(name: "cloudy night", sun: -25, rising: false, moonUp: nil, weather: weather(3, 14, low: 0.95, mid: 0.4, wind: 14)),
    ]

    /// Positions in the sky for one moment.
    private struct Sky {
        var sun = (alt: 0.0, az: 0.0)
        var moon = (alt: 0.0, az: 0.0)
        var lit = 0.0
        var waxing = true
        var lst = 0.0
        var planets: [(name: String, alt: Double, az: Double, mag: Float, color: SIMD3<Float>)] = []
    }

    private let gpu: GPU
    private let lutPass: MTLRenderPipelineState
    private let skyPass: MTLRenderPipelineState
    private let starPass: MTLRenderPipelineState
    private let frontPass: MTLRenderPipelineState
    private let finishPass: MTLRenderPipelineState
    private let meterPass: MTLComputePipelineState
    private let meter: MTLBuffer
    private var exposureNow: Double = 0
    /// How far the wind has carried the clouds, in kilometres east and north.
    private var drift = SIMD2<Float>(0, 0)
    private let bloom: Bloom
    private var lut: MTLTexture?
    private var hdr: MTLTexture?
    private var weather: Weather?
    private var fetched = Date.distantPast
    private var attempted = false
    private let preset: Preset?
    private var offset: TimeInterval = 0
    private var sky = Sky()
    private var skyAt: Float = -100
    private var stars: MTLBuffer?
    private var starCount = 0
    private var camera: Double?
    private var flash: Float = 0
    private var nextFlash: Float = 8
    private var thunderAt: Float = -1
    private var crickets: [(next: Float, pulses: Int, pulseAt: Float)] = [(0.4, 0, 0), (0.9, 0, 0)]

    var scale: Float { 0.6 }
    var ready: Bool { attempted }

    init(gpu: GPU) throws {
        self.gpu = gpu
        let library = try gpu.library(SkyKit.metal + Self.source)
        lutPass = try gpu.pipeline(library, "skyLUT")
        skyPass = try gpu.pipeline(library, "skyBase")
        starPass = try gpu.pipeline(library, "sprite", vertex: "skyStar", blend: .add)
        frontPass = try gpu.pipeline(library, "skyFront", blend: .alpha)
        finishPass = try gpu.pipeline(library, "skyFinish")
        meterPass = try gpu.kernel(library, "skyMeter")
        meter = gpu.buffer(16)
        bloom = try Bloom(gpu: gpu)
        Location.shared.request()
        if let v = Snapshot.variant, v > 0 {
            preset = Self.presets[(v - 1) % Self.presets.count]
        } else {
            preset = nil
        }
    }

    // MARK: Data

    private func refresh(_ date: Date) {
        fetched = date
        if let preset {
            if !attempted { offset = Self.moment(for: preset) }
            weather = preset.weather
            attempted = true
            return
        }
        // Rounded to a tenth of a degree, about 11 km: plenty for weather, and it keeps the exact location private.
        let lat = (Location.shared.latitude * 10).rounded() / 10, lon = (Location.shared.longitude * 10).rounded() / 10
        let fields = "temperature_2m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,precipitation,rain,snowfall,"
            + "weather_code,wind_speed_10m,wind_direction_10m,is_day,visibility,relative_humidity_2m"
        let url = URL(string: "https://api.open-meteo.com/v1/forecast?latitude=\(lat)&longitude=\(lon)&current=\(fields)")!
        Task { @MainActor [weak self] in
            let json = await Feed.object(url, maxAge: 900)
            guard let self else { return }
            if let w = Weather(json: json) { weather = w }
            attempted = true
        }
    }

    /// How far from now a preset's moment is: when the Sun reaches its altitude (and the Moon is up, if asked).
    private static func moment(for p: Preset) -> TimeInterval {
        let lat = Location.shared.latitude, lon = Location.shared.longitude
        func sunAlt(_ d: Date) -> Double { Astro.horizontal(Astro.sun(d), d, latitude: lat, longitude: lon).alt }
        func moonAlt(_ d: Date) -> Double { Astro.horizontal(Astro.moon(d), d, latitude: lat, longitude: lon).alt }
        let start = Date()
        var best: (TimeInterval, Double)?
        for step in 0..<(30 * 24 * 12) {
            let t = TimeInterval(step) * 300, d = start + t
            let a = sunAlt(d), b = sunAlt(d + 300)
            if let up = p.moonUp, up ? moonAlt(d) < 25 : moonAlt(d) > -3 { continue }
            if let target = p.sun {
                guard (a - target) * (b - target) <= 0, (b > a) == p.rising else { continue }
                return t
            } else if best == nil || a > best!.1 {
                best = (t, a)
                if step > 24 * 12 { break }
            }
        }
        return best?.0 ?? 0
    }

    private func updateSky(_ date: Date) {
        let lat = Location.shared.latitude, lon = Location.shared.longitude
        var s = Sky()
        let sun = Astro.horizontal(Astro.sun(date), date, latitude: lat, longitude: lon)
        let moon = Astro.horizontal(Astro.moon(date), date, latitude: lat, longitude: lon)
        let phase = Astro.moonPhase(date)
        s.sun = sun
        s.moon = moon
        s.lit = phase.lit
        s.waxing = phase.waxing
        s.lst = Astro.lst(date, longitude: lon)
        let looks: [(Astro.Body, Float, SIMD3<Float>)] = [(.venus, -4.2, [1, 0.97, 0.9]), (.jupiter, -2.3, [1, 0.94, 0.84]),
                                                          (.mars, 0.6, [1, 0.62, 0.42]), (.saturn, 0.7, [1, 0.9, 0.72]),
                                                          (.mercury, 0.2, [0.95, 0.88, 0.8])]
        for (body, mag, color) in looks {
            let h = Astro.horizontal(Astro.planet(body, date), date, latitude: lat, longitude: lon)
            s.planets.append((body.name, h.alt, h.az, mag, color))
        }
        sky = s

        // Stars and planets as directions (x east, y up, z north), magnitude, colour index.
        var points: [SIMD4<Float>] = []
        for star in SkyStars.all {
            let h = Astro.horizontal(Astro.Equatorial(ra: star.ra * 15, dec: star.dec, distance: 1), date, latitude: lat, longitude: lon)
            points.append(SIMD4(Self.direction(h.alt, h.az), Float(star.mag)))
            points.append(SIMD4(Float(star.bv), 0, 0, 0))
        }
        for p in s.planets {
            points.append(SIMD4(Self.direction(p.alt, p.az), p.mag))
            points.append(SIMD4(p.color, 1))
        }
        stars = gpu.buffer(points)
        starCount = points.count / 2
    }

    /// A unit vector toward an altitude and azimuth: x east, y up, z north.
    static func direction(_ alt: Double, _ az: Double) -> SIMD3<Float> {
        let a = alt * .pi / 180, z = az * .pi / 180
        return SIMD3(Float(cos(a) * sin(z)), Float(sin(a)), Float(cos(a) * cos(z)))
    }

    /// Light left after crossing the atmosphere from `height` km up toward a body at `elevation` degrees.
    nonisolated static func transmittance(height: Double, elevation: Double, haze: Double = 1) -> SIMD3<Double> {
        let re = 6360.0, ra = 6420.0, r0 = re + height
        let e = elevation * .pi / 180
        let d = SIMD2(cos(e), sin(e)), p = SIMD2(0.0, r0)
        // The ray's closest approach to the Earth's centre: below the surface, the body is hidden by the planet.
        let shadow = e < 0 ? max(0, min(1, r0 * cos(e) - re + 0.5)) : 1
        guard shadow > 0 else { return .zero }
        let b = simd_dot(p, d), top = -b + (b * b - simd_dot(p, p) + ra * ra).squareRoot()
        let rayleigh = SIMD3<Double>(5.802e-3, 13.558e-3, 33.1e-3), ozone = SIMD3<Double>(1.3e-3, 1.881e-3, 0.085e-3)
        let mie = 3.996e-3 * 1.11 * haze
        var depth = SIMD3<Double>.zero
        let n = 40, dt = top / Double(n)
        for i in 0..<n {
            let q = p + d * ((Double(i) + 0.5) * dt)
            let h = max(simd_length(q) - re, 0)
            let r: Double = exp(-h / 8), m: Double = mie * exp(-h / 1.2), o: Double = max(0, 1 - abs(h - 25) / 15)
            let local: SIMD3<Double> = rayleigh * r + ozone * o + SIMD3<Double>(repeating: m)
            depth += local * dt
        }
        return SIMD3(exp(-depth.x), exp(-depth.y), exp(-depth.z)) * shadow
    }

    nonisolated static func smooth(_ a: Double, _ b: Double, _ x: Double) -> Double {
        let t = max(0, min(1, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    }

    /// A first guess at exposure before the sky has been measured: bright day, rich twilight, deep night.
    nonisolated static func exposure(sun alt: Double, overcast: Double) -> Double {
        let knots: [(Double, Double)] = [(-90, 2400), (-16, 2400), (-11, 600), (-6, 60), (-2, 12), (2, 4), (8, 2), (90, 1.4)]
        var e = knots.last!.1
        for i in 1..<knots.count where alt <= knots[i].0 {
            let (a0, e0) = knots[i - 1], (a1, e1) = knots[i]
            let t = (alt - a0) / (a1 - a0)
            e = exp(log(e0) + (log(e1) - log(e0)) * t)
            break
        }
        return e * (1 + 0.8 * overcast * (alt > -6 ? 1 : 0.2))
    }

    // MARK: Facts

    var facts: String? {
        guard attempted else { return "Looking up the weather over \(Location.shared.place)." }
        let s = sky
        var parts: [String] = []
        let place = Location.shared.place.isEmpty ? "here" : Location.shared.place
        if let w = weather {
            var line = "Over \(place): \(w.summary), \(Int(w.temperature.rounded())) °C"
            if w.wind >= 3 { line += ", wind \(Int(w.wind.rounded())) km/h from the \(SkyKit.compass(w.windFrom))" }
            parts.append(line)
        } else {
            parts.append("Over \(place): the weather is unavailable offline, so the sky is shown clear")
        }
        let sunAlt = Int(s.sun.alt.rounded())
        if sunAlt > 0 {
            parts.append("the Sun is \(sunAlt)° up in the \(SkyKit.compass(s.sun.az))")
        } else if sunAlt == 0 {
            parts.append("the Sun is on the horizon in the \(SkyKit.compass(s.sun.az))")
        } else {
            parts.append("the Sun is \(-sunAlt)° below the horizon")
        }
        let lit = Int((s.lit * 100).rounded())
        if s.moon.alt > 0 {
            parts.append("the Moon is \(lit)% lit, \(Int(s.moon.alt.rounded()))° up in the \(SkyKit.compass(s.moon.az))")
        } else {
            parts.append("the Moon, \(lit)% lit, is below the horizon")
        }
        if s.sun.alt < -6, let p = s.planets.filter({ $0.alt > 5 && $0.name != "Mercury" }).min(by: { $0.mag < $1.mag }) {
            parts.append("\(p.name) shines in the \(SkyKit.compass(p.az))")
        }
        var text = parts.joined(separator: "; ") + "."
        if let rate = chirpRate { text += " Crickets chirp \(Int(rate.rounded())) times a minute, by Dolbear's law." }
        if let preset { text = "Preview of \(preset.name). " + text }
        return text
    }

    /// Dolbear's law: chirps a minute = 4 × °F − 160, for crickets at night above about 13 °C, when it's dry.
    private var chirpRate: Double? {
        guard let w = weather, sky.sun.alt < -4, w.temperature >= 13, w.rainIntensity < 0.3, w.snowIntensity == 0 else { return nil }
        return Self.dolbear(celsius: w.temperature)
    }

    nonisolated static func dolbear(celsius: Double) -> Double { max(0, 4 * (celsius * 9 / 5 + 32) - 160) }

    // MARK: Drawing

    func draw(_ f: Frame) {
        if f.now.timeIntervalSince(fetched) > 900 { refresh(f.now) }
        if f.index == 0 { drift = SIMD2(f.seed * 3.7, f.seed * 1.3) }
        let date = f.now + offset
        if f.time - skyAt > 2 || stars == nil {
            skyAt = f.time
            updateSky(date)
        }
        let w = weather ?? Weather()
        let s = sky
        let size = (f.target.width, f.target.height)
        if hdr?.width != size.0 || hdr?.height != size.1 { hdr = f.gpu.texture(size.0, size.1) }
        if lut == nil { lut = f.gpu.texture(200, 125) }

        // Look toward the Sun by day and twilight, toward the Moon by night, otherwise toward the equator.
        let target = s.sun.alt > -12 ? s.sun.az : s.moon.alt > 4 ? s.moon.az : (Location.shared.latitude >= 0 ? 180 : 0)
        if let c = camera {
            var d = (target - c).truncatingRemainder(dividingBy: 360)
            if d > 180 { d -= 360 } else if d < -180 { d += 360 }
            camera = c + d * (1 - exp(-Double(f.dt) / 180))
        } else {
            camera = target + 8
        }
        let az = Float(camera! * .pi / 180)
        // Tilt up a little to keep a high Moon in the window.
        let lookUp = s.sun.alt < -12 && s.moon.alt > 4 ? min(max(s.moon.alt - 36, 0), 7) : 0
        let pitch = Float((20 + lookUp + 1.2 * sin(Double(f.time) / 300)) * .pi / 180)
        let focal = Float(1 / tan(27.5 * .pi / 180) - 0.035 * lookUp)

        let overcast = max(w.low, w.mid * 0.7)
        // Exposure adapts like an eye to the sky's measured brightness, aiming lower at night so night stays night.
        let moonUp = s.moon.alt > 0 ? s.lit : 0
        let key = 0.06 + 0.05 * moonUp + (0.32 - 0.06 - 0.05 * moonUp) * Self.smooth(-9, 6, s.sun.alt)
        let measured = Double(meter.contents().load(as: Float.self))
        let wanted = measured > 0 && f.index > 1 ? key / measured : Self.exposure(sun: s.sun.alt, overcast: overcast)
        exposureNow = exposureNow == 0 ? wanted : exp(log(exposureNow) + (log(wanted) - log(exposureNow)) * (1 - exp(-Double(f.dt) / 0.7)))
        exposureNow = min(max(exposureNow, 0.2), 1e6)
        let exposure = Float(exposureNow)
        let haze = min(1 + 2 * w.fog + max(0, (w.humidity - 70) / 30), 2.5)
        // Moonlight is about 2.5 millionths of sunlight at full Moon; the sky gets a perceptual boost so moonlit nights read as brighter.
        let moonIllum = 2.5e-6 * pow(s.lit, 1.6)
        // The light that falls on each cloud layer: sunlight, or moonlight once the Sun is well down.
        let sunUp = s.sun.alt > -8
        let lightAlt = sunUp ? s.sun.alt : s.moon.alt
        let lightScale = sunUp ? 1.0 : moonIllum
        // Each layer's light is taken just before its shadow line; the shader decides per cloud whether the Sun has set there.
        func light(_ h: Double) -> SIMD4<Float> {
            let dip = (2 * h / 6360).squareRoot() * 180 / .pi
            let t = Self.transmittance(height: h, elevation: max(lightAlt, -dip + 0.3), haze: haze) * lightScale
            return SIMD4(Float(t.x), Float(t.y), Float(t.z), 0)
        }
        let sunDir = Self.direction(s.sun.alt, s.sun.az), moonDir = Self.direction(s.moon.alt, s.moon.az)
        let lightDir = sunUp ? sunDir : moonDir
        let ground = Self.transmittance(height: 0.05, elevation: s.sun.alt, haze: haze)

        // Wind: clouds drift toward where the wind blows, faster aloft than at ground level.
        let toward = (w.windFrom + 180) * .pi / 180
        let speed = w.wind / 3.6 / 1000 * 1.6
        drift += SIMD2(Float(sin(toward) * speed), Float(cos(toward) * speed)) * f.dt
        let rain = Float(w.rainIntensity), snow = Float(w.snowIntensity), fog = Float(w.fog)

        // Lightning in a thunderstorm: a soft glow inside the clouds every half minute or so, never a strobe.
        if w.thunder {
            if f.time > nextFlash {
                nextFlash = f.time + Float.random(in: 18...45)
                flash = 1
                thunderAt = f.time + Float.random(in: 2.5...7)
            }
            flash = max(0, flash - f.dt / 1.4)
        }
        let night = Float(max(0, min(1, (-s.sun.alt - 4) / 10)))
        let hour = Float(Calendar.current.component(.hour, from: date)) + Float(Calendar.current.component(.minute, from: date)) / 60
        let params: [SIMD4<Float>] = [
            [az, pitch, focal, exposure],
            SIMD4(sunDir, 20),
            SIMD4(moonDir, s.sun.alt > -4 || s.moon.alt < -5 ? 0 : Float(moonIllum * 20 * 6)),
            [Float(s.lit), Float(0.8 * .pi / 180), s.waxing ? 1 : -1, Float(Location.shared.latitude * .pi / 180)],
            SIMD4(Float(ground.x), Float(ground.y), Float(ground.z), 0),
            light(1.5), light(4), light(9),
            SIMD4(lightDir, sunUp ? 1 : 0),
            [Float(w.low), Float(w.mid), Float(w.high), w.thunder ? 0.8 : rain > 0.3 ? 0.6 : snow > 0.3 ? 0.35 : 0],
            [drift.x, drift.y, Float(toward), Float(w.wind)],
            [rain, snow, fog, flash],
            [night * (1 - Float(overcast) * 0.85), Float(s.lst * .pi / 180), Float(haze), hour],
            [Float(Location.shared.latitude * 7.3 + Location.shared.longitude * 3.1).truncatingRemainder(dividingBy: 97), 0, 0, 0],
        ]
        let p = f.gpu.buffer(params)
        let u = f.uniforms()
        f.cmd.draw(lutPass, into: lut!, uniforms: f.uniforms(res: SIMD2(Float(f.target.width), Float(f.target.height))), buffers: [p])
        f.cmd.compute(meterPass, 1, uniforms: u, textures: [lut!], buffers: [p, meter])
        f.cmd.draw(skyPass, into: hdr!, uniforms: u, textures: [lut!], buffers: [p])
        if let stars, starCount > 0 {
            f.cmd.draw(starPass, into: hdr!, clear: false, uniforms: u, buffers: [p, stars], vertices: starCount, primitive: .point)
        }
        f.cmd.draw(frontPass, into: hdr!, clear: false, uniforms: u, textures: [lut!], buffers: [p])
        let glow = bloom.glow(f.cmd, hdr!)
        f.cmd.draw(finishPass, into: f.target, uniforms: u, textures: [hdr!, glow], buffers: [p])

        if let synth = f.sound { play(synth, f, w) }
    }

    // MARK: Sound

    /// Rain hiss and drops when it rains, wind that rises with the real wind speed, distant thunder after lightning,
    /// and at night crickets chirping at the rate the temperature sets.
    private func play(_ synth: Synth, _ f: Frame, _ w: Weather) {
        let rain = Float(w.rainIntensity), snow = Float(w.snowIntensity)
        let windy = Float(min(w.wind / 45, 1))
        let gust = 0.75 + 0.25 * sin(f.time * 0.13) * sin(f.time * 0.051 + 1)
        let windLevel = 0.022 + 0.065 * windy * windy * gust // a soft air bed even on still nights
        let level = windLevel + 0.05 * rain + 0.01 * snow
        let cutoff = (250 + 700 * windy * gust) * (1 - rain) + 3200 * rain
        synth.noise(level, cutoff: cutoff, crackle: rain > 0 ? 12 + 110 * rain : 0, crackleTone: 3600)
        // Strong wind starts to whistle a little.
        let whistle = max(0, windy - 0.5) * 2
        synth.drone(0, 310 + 40 * sin(f.time * 0.07), 0.012 * whistle * gust, pan: -0.4)
        synth.drone(1, 465 + 50 * sin(f.time * 0.05 + 2), 0.008 * whistle * gust, pan: 0.4)
        if thunderAt > 0, f.time > thunderAt {
            thunderAt = -1
            synth.play(.thump, 42, amp: 0.13, pan: Float.random(in: -0.5...0.5), length: 3.5)
            synth.play(.thump, 63, amp: 0.06, pan: 0, length: 2.5)
        }
        guard let rate = chirpRate, rate > 0 else { return }
        let interval = Float(60 / rate)
        let voices: [(pitch: Float, pan: Float, tempo: Float)] = [(4400, -0.5, 1), (4750, 0.45, 1.035)]
        for i in crickets.indices {
            var c = crickets[i]
            if c.pulses == 0, f.time > c.next {
                c.pulses = 3
                c.pulseAt = f.time
                c.next = f.time + interval * voices[i].tempo * Float.random(in: 0.97...1.03)
            }
            if c.pulses > 0, f.time >= c.pulseAt {
                synth.play(.click, voices[i].pitch, amp: 0.026, pan: voices[i].pan, length: 0.05)
                c.pulses -= 1
                c.pulseAt += 0.038
            }
            crickets[i] = c
        }
    }
}
