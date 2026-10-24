import AppKit
import Metal

/// A typographic instrument: many clocks at once, set like a type specimen over a slow astrolabe of turning dials.
/// Local time, UTC, Unix time, Julian days, sidereal time, atomic and GPS time, Mars time at Perseverance, the ISO
/// week, and how long Voyager 1's signals take to reach us.
@MainActor final class TimeScene: Scene {
    private let dial: MTLRenderPipelineState
    private var lastText = ""
    private var lastDraw: Float = -1
    private var lastSecond = -1
    private var lastMinute = -1

    var scale: Float { 1 }

    var facts: String? {
        let now = Date()
        let unix = Int(floor(Chronology.unix(now)))
        let jd = Chronology.grouped(Chronology.julianDay(now), decimals: 2)
        let sol = Chronology.perseveranceSol(now)
        let lmst = Chronology.clock(Chronology.marsLocalTime(now, eastLongitude: Chronology.jezero.longitude))
        let light = Chronology.voyagerLightTime(now)
        var s = "Unix time \(SkyKit.grouped(unix)); Julian Day \(jd); Perseverance sol \(SkyKit.grouped(sol)) at \(lmst.prefix(5)) local mean solar time in Jezero crater. "
        s += "Voyager 1's signals take \(Chronology.duration(light)) to reach Earth"
        let days = Int((Chronology.oneLightDay.timeIntervalSince(now) / 86400).rounded(.up))
        s += days > 0 ? "; it will be a full light-day away in \(days) day\(days == 1 ? "" : "s"), around \(Self.day(Chronology.oneLightDay))."
                       : "; it has been more than a light-day away since \(Self.day(Chronology.oneLightDay))."
        return s
    }

    init(gpu: GPU) throws {
        dial = try gpu.pipeline(gpu.library(Self.source), "timeDials")
        Location.shared.request()
    }

    private static let dayFormat: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_GB")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "d MMMM yyyy"
        return f
    }()

    private static let dateFormat: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_GB")
        f.dateFormat = "EEEE d MMMM yyyy"
        return f
    }()

    private static func day(_ date: Date) -> String { dayFormat.string(from: date) }

    func draw(_ f: Frame) {
        let now = f.now
        let t = now.timeIntervalSince1970
        let day = t.truncatingRemainder(dividingBy: 86400) / 86400
        let lst = Astro.lst(now, longitude: Location.shared.longitude) / 360
        let sol = Chronology.marsCoordinatedTime(now) / 24
        let second = t.truncatingRemainder(dividingBy: 60) / 60
        let drift = SkyKit.drift(Double(f.time) + Double(f.seed) * 31, amount: 16, seed: Double(f.seed))
        f.cmd.draw(dial, into: f.target, uniforms: f.uniforms([Float(second), Float(day), Float(lst), Float(sol)],
                                                             [Float(drift.x), Float(drift.y), f.overlay.size.width > 0 ? Float(f.overlay.size.width) : 1, 0]))
        drawOverlay(f, now, drift)
        if let synth = f.sound { play(synth, now) }
    }

    /// A soft tick every second, alternating a little in pitch and side like a clock's escapement, and a quiet
    /// two-note chime at the top of each minute (lower on the hour).
    private func play(_ synth: Synth, _ now: Date) {
        let t = now.timeIntervalSince1970
        let second = Int(floor(t))
        if second != lastSecond {
            if lastSecond >= 0 {
                let tock = second % 2 == 0
                synth.play(.click, tock ? 1750 : 1980, amp: 0.028, pan: tock ? -0.15 : 0.15, length: 0.012)
            }
            lastSecond = second
        }
        let minute = second / 60
        if minute != lastMinute {
            if lastMinute >= 0 {
                let hour = minute % 60 == 0
                synth.play(.bell, hour ? 392 : 784, amp: 0.05, pan: -0.1, length: 3.5)
                synth.play(.bell, hour ? 587.3 : 1174.7, amp: 0.025, pan: 0.1, length: 3)
            }
            lastMinute = minute
        }
    }

    // MARK: Type

    private struct Cell { let label: String; let value: String; let caption: String }

    private func cells(_ now: Date) -> (hero: [String], voyager: [String], grid: [Cell]) {
        let zone = TimeZone.current
        let place = Location.shared.place.isEmpty ? zone.identifier : Location.shared.place
        let lat = Location.shared.latitude, lon = Location.shared.longitude
        let where_ = String(format: "%.2f° %@  %.2f° %@", abs(lat), lat >= 0 ? "N" : "S", abs(lon), lon >= 0 ? "E" : "W")
        let date = Self.dateFormat
        date.timeZone = zone
        let offset = zone.secondsFromGMT(for: now)
        let gmt = String(format: "UTC%@%d%@", offset >= 0 ? "+" : "−", abs(offset) / 3600, abs(offset) % 3600 == 0 ? "" : String(format: ":%02d", abs(offset) % 3600 / 60))
        let name = zone.localizedName(for: zone.isDaylightSavingTime(for: now) ? .daylightSaving : .standard, locale: Locale(identifier: "en_GB")) ?? zone.identifier
        let hero = ["LOCAL TIME  ·  \(place.uppercased())", Chronology.clock(now, in: zone), "\(date.string(from: now))  ·  \(name)  ·  \(gmt)", where_]

        let light = Chronology.voyagerLightTime(now)
        let au = Chronology.voyagerDistance(now)
        let km = au * Chronology.kilometresPerAU / 1e9
        let days = Int((Chronology.oneLightDay.timeIntervalSince(now) / 86400).rounded(.up))
        let voyager = ["VOYAGER 1  ·  ONE-WAY LIGHT TIME", Chronology.clock(light / 3600),
                       String(format: "%.3f AU  ·  %.3f billion km from Earth", au, km),
                       days > 0 ? "a full light-day away around \(Self.day(Chronology.oneLightDay)), in \(days) days"
                                : "more than a light-day away since \(Self.day(Chronology.oneLightDay))"]

        let utc = TimeZone(identifier: "UTC")!
        let jd = Chronology.julianDay(now)
        let week = Chronology.isoWeek(now)
        let msd = Chronology.marsSolDate(now)
        let grid = [
            Cell(label: "UTC", value: Chronology.clock(now, in: utc), caption: "Coordinated Universal Time"),
            Cell(label: "UNIX TIME", value: SkyKit.grouped(Int(floor(Chronology.unix(now)))), caption: "seconds since 1 January 1970"),
            Cell(label: "JULIAN DAY", value: Chronology.grouped(jd, decimals: 5), caption: "days since noon, 1 January 4713 BC"),
            Cell(label: "MODIFIED JULIAN DATE", value: Chronology.grouped(Chronology.modifiedJulianDate(now), decimals: 5),
                 caption: "Julian Day − 2,400,000.5"),
            Cell(label: "LOCAL SIDEREAL TIME", value: Chronology.clock(Astro.lst(now, longitude: lon) / 15),
                 caption: String(format: "the stars' clock at %.2f° %@", abs(lon), lon >= 0 ? "E" : "W")),
            Cell(label: "ATOMIC TIME  ·  TAI", value: Chronology.clock(now + Chronology.leapSeconds, in: utc), caption: "UTC + 37 leap seconds"),
            Cell(label: "GPS TIME", value: Chronology.clock(now + Chronology.gpsOffset, in: utc), caption: "UTC + 18 s, no leap seconds since 1980"),
            Cell(label: "DAY  ·  ISO WEEK", value: String(format: "%03d · W%02d", Chronology.dayOfYear(now), week.week),
                 caption: String(format: "ISO %d-W%02d-%d", week.year, week.week, week.day)),
            Cell(label: "MARS COORDINATED TIME", value: Chronology.clock(Chronology.marsCoordinatedTime(now)), caption: "mean solar time at Airy-0"),
            Cell(label: "MARS SOL DATE", value: Chronology.grouped(msd, decimals: 4), caption: "sols since 29 December 1873"),
            Cell(label: "PERSEVERANCE  ·  SOL", value: SkyKit.grouped(Chronology.perseveranceSol(now)), caption: "since landing on 18 February 2021"),
            Cell(label: "JEZERO CRATER  ·  LMST", value: Chronology.clock(Chronology.marsLocalTime(now, eastLongitude: Chronology.jezero.longitude)),
                 caption: "local mean solar time, 18.44° N 77.45° E"),
        ]
        return (hero, voyager, grid)
    }

    private func drawOverlay(_ f: Frame, _ now: Date, _ drift: CGPoint) {
        // Redraw when any figure changes (a few times a second) or the slow drift has moved.
        let (hero, voyager, grid) = cells(now)
        let text = (hero + voyager + grid.map(\.value)).joined() + "\(Int(drift.x * 2)),\(Int(drift.y * 2))"
        guard text != lastText, f.time - lastDraw >= 0.1 || lastDraw < 0 else { return }
        lastText = text
        lastDraw = f.time
        f.overlay.draw { ctx, size in
            let W = size.width, H = size.height
            let mx = W * 0.07 + drift.x, my = H * 0.085 + drift.y
            let right = W - W * 0.07 + drift.x
            let hair = NSColor(white: 1, alpha: 0.16)
            @MainActor func rule(_ y: CGFloat, from x0: CGFloat = mx, to x1: CGFloat? = nil) {
                ctx.setStrokeColor(hair.cgColor)
                ctx.setLineWidth(0.5)
                ctx.move(to: CGPoint(x: x0, y: y))
                ctx.addLine(to: CGPoint(x: x1 ?? right, y: y))
                ctx.strokePath()
            }
            @MainActor func label(_ s: String, _ x: CGFloat, _ y: CGFloat, anchor: CGFloat = 0) {
                ctx.text(s, at: CGPoint(x: x, y: y), size: 10.5, weight: .medium, color: NSColor(white: 1, alpha: 0.46),
                         anchor: CGPoint(x: anchor, y: 0), tracking: 2.2)
            }
            @MainActor func draw(_ s: String, _ font: NSFont, _ alpha: CGFloat, _ x: CGFloat, _ y: CGFloat, tracking: CGFloat = 0) {
                NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: NSColor(white: 1, alpha: alpha), .kern: tracking])
                    .draw(at: CGPoint(x: x, y: y))
            }

            // Header.
            label("TIME, ALL OF IT", mx, my)
            label(hero[3], right, my, anchor: 1)
            rule(my + 24)

            // The local time, large and light, with Voyager's light time beside it.
            let heroTop = my + 44
            label(hero[0], mx, heroTop)
            let big = min(H * 0.17, W * 0.1)
            draw(hero[1], NSFont.monospacedDigitSystemFont(ofSize: big, weight: .ultraLight), 0.93, mx - big * 0.06, heroTop + 12, tracking: -big * 0.01)
            let below = heroTop + 18 + big * 1.2
            ctx.text(hero[2], at: CGPoint(x: mx, y: below), size: 15, weight: .light, color: NSColor(white: 1, alpha: 0.62))

            let vx = mx + (right - mx) * 0.62
            label(voyager[0], vx, heroTop)
            draw(voyager[1], NSFont.monospacedSystemFont(ofSize: big * 0.36, weight: .light), 0.9, vx - 2, heroTop + 18)
            ctx.text(voyager[2], at: CGPoint(x: vx, y: heroTop + 26 + big * 0.48), size: 13, weight: .regular, color: NSColor(white: 1, alpha: 0.6))
            ctx.text(voyager[3], at: CGPoint(x: vx, y: heroTop + 46 + big * 0.48), size: 13, weight: .regular, color: NSColor(white: 1, alpha: 0.45))

            // Three rows of four instruments, each a label, a figure in SF Mono, and a line saying what it counts.
            let gridTop = below + 52
            let rows = 3, cols = 4
            let rowH = (H - H * 0.07 + drift.y - gridTop) / CGFloat(rows)
            let colW = (right - mx) / CGFloat(cols)
            let valueSize = min(rowH * 0.3, colW * 0.092)
            for r in 0..<rows {
                let y = gridTop + CGFloat(r) * rowH
                rule(y)
                for c in 0..<cols {
                    let cell = grid[r * cols + c]
                    let x = mx + CGFloat(c) * colW
                    label(cell.label, x, y + 16)
                    draw(cell.value, NSFont.monospacedSystemFont(ofSize: valueSize, weight: .light), 0.9, x - 1, y + 34)
                    ctx.text(cell.caption, at: CGPoint(x: x, y: y + 42 + valueSize * 1.2), size: 11.5, color: NSColor(white: 1, alpha: 0.42))
                }
            }
            rule(gridTop + CGFloat(rows) * rowH)
        }
    }

    private static let source = """
    // An astrolabe of hairline dials turning with real clocks: the minute, the UTC day, the sidereal day, the Mars sol.
    // a: fractions of the minute, UTC day, sidereal day and Mars sol   b: drift in points, overlay width in points
    float ticks(float angle, float count, float width) {
        float x = fract(angle / TAU * count);
        return smoothstep(width, 0.0, min(x, 1 - x));
    }

    fragment float4 timeDials(V in [[stage_in]], constant U &u [[buffer(0)]]) {
        float2 p = centered(in.uv, u.res);
        float pt = 2.0 / u.res.y * (u.res.x / max(u.b.z, 1.0));
        float2 c = float2(u.res.x / u.res.y * 0.42, -0.08) + float2(u.b.x, -u.b.y) * pt;
        float2 d = p - c;
        float r = length(d);
        float a = atan2(d.y, d.x);
        float px = 2.0 / u.res.y;
        float3 col = mix(float3(0.006, 0.008, 0.014), float3(0.016, 0.018, 0.03), exp(-r * 1.4));
        // Rings: radius, angle, tick count, tick length, brightness.
        float4 rings[4] = { float4(1.18, u.a.z, 360, 0.022), float4(1.0, u.a.y, 96, 0.03), float4(0.82, u.a.w, 24, 0.045), float4(0.66, u.a.x, 60, 0.02) };
        float3 tints[4] = { float3(0.55, 0.65, 1.0), float3(1.0, 0.92, 0.8), float3(1.0, 0.6, 0.45), float3(0.8, 0.85, 1.0) };
        for (int i = 0; i < 4; i++) {
            float R = rings[i].x;
            float ang = a + rings[i].y * TAU;
            float circle = smoothstep(px * 1.2, 0.0, abs(r - R));
            float band = step(R - rings[i].w, r) * step(r, R);
            float major = ticks(ang, rings[i].z / (i == 1 ? 4 : 5), px / max(r, 0.1) * 0.9);
            float minor = ticks(ang, rings[i].z, px / max(r, 0.1) * 0.6);
            float inner = step(R - rings[i].w * 0.5, r);
            float marks = band * max(major, minor * inner * 0.7);
            // One brighter index mark per ring shows the hand.
            float hand = smoothstep(px * 1.5 / max(r, 0.1), 0.0, abs(fmod(ang + PI * 0.5 + TAU * 4, TAU) - PI)) * smoothstep(R + 0.05, R, r) * step(R - 0.12, r);
            col += tints[i] * (circle * 0.07 + marks * 0.09 + hand * 0.16);
        }
        // Faint grain so the dark gradient never bands.
        col += (hash12(in.uv * u.res + fract(u.time) * 100) - 0.5) * 0.004;
        float2 v = in.uv - 0.5;
        col *= 1 - 0.4 * dot(v, v);
        return float4(col, 1);
    }
    """
}
