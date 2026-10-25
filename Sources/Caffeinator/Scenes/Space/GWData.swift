import Accelerate
import Foundation

/// The LIGO Hanford and Livingston recordings of one merger, from the Gravitational Wave Open Science Center, processed
/// the way the discovery papers show them: whitened by the detector's own noise spectrum, kept to the 30–400 Hz band
/// where the instruments hear best, Livingston shifted by the light-travel delay between the sites (and flipped when
/// the arms point the other way), plus a Q-transform-like map of energy over time and frequency.
struct GWSignal: Sendable {
    /// Samples per second of `hanford` and `livingston`.
    static let rate = 2048.0
    /// Seconds before the merger that the arrays start; they run to `after` seconds past it.
    static let before = 3.0
    static let after = 1.5
    /// Rows of the time–frequency map, log-spaced from `low` to `high` hertz; columns every millisecond.
    static let rows = 64
    static let low = 24.0, high = 480.0
    static let columnRate = 1000.0

    /// Whitened strain in standard deviations of the noise, merger at `before` seconds.
    var hanford: [Float]
    var livingston: [Float]
    /// Energy over noise, row-major (`rows` × columns), for each detector.
    var energyH: [Float]
    var energyL: [Float]
    var columns: Int
    /// Livingston's shift onto Hanford in milliseconds, and whether it was flipped.
    var lag: Double
    var flipped: Bool
    /// The merger's GPS time, from the peak of the signal.
    var gps: Double

    static func frequency(row: Double) -> Double { low * pow(high / low, row / Double(rows - 1)) }

    /// Energy at `seconds` from the merger in a row, summed over both detectors, or 0 outside the recording.
    func energy(row: Int, at seconds: Double, detector: Int? = nil) -> Float {
        let c = Int(((seconds + Self.before) * Self.columnRate).rounded())
        guard row >= 0, row < Self.rows, c >= 0, c < columns else { return 0 }
        switch detector {
        case 0: return energyH[row * columns + c]
        case 1: return energyL[row * columns + c]
        default: return 0.5 * (energyH[row * columns + c] + energyL[row * columns + c])
        }
    }
}

enum GWData {
    /// A merger with public 32-second, 4 kHz strain from both LIGO sites: the GWOSC event folder, the GPS second the
    /// files start, and the catalogue's GPS time.
    struct Source: Sendable {
        var name: String
        var folder: String
        var start: Int
        var gps: Double

        func url(_ site: String) -> URL {
            URL(string: "https://gwosc.org/eventapi/json/\(folder)/\(site)-\(site)1_GWOSC_4KHZ_R1-\(start)-32.txt.gz")!
        }
    }

    /// Loud binary black hole mergers whose chirp stands out of the noise by eye, as in the papers' figures.
    static let sources = [
        Source(name: "GW150914", folder: "GWTC-1-confident/GW150914/v3", start: 1_126_259_447, gps: 1_126_259_462.4),
        Source(name: "GW200129", folder: "GWTC-3-confident/GW200129_065458/v1", start: 1_264_316_101, gps: 1_264_316_116.4),
        Source(name: "GW170814", folder: "GWTC-1-confident/GW170814/v3", start: 1_186_741_846, gps: 1_186_741_861.5),
        Source(name: "GW190521", folder: "GWTC-2.1-confident/GW190521/v4", start: 1_242_442_952, gps: 1_242_442_967.4),
        Source(name: "GW190412", folder: "GWTC-2.1-confident/GW190412/v4", start: 1_239_082_247, gps: 1_239_082_262.1),
        Source(name: "GW190521_074359", folder: "GWTC-2.1-confident/GW190521_074359/v2", start: 1_242_459_842, gps: 1_242_459_857.4),
        Source(name: "GW170104", folder: "GWTC-1-confident/GW170104/v2", start: 1_167_559_921, gps: 1_167_559_936.6),
        Source(name: "GW200224", folder: "GWTC-3-confident/GW200224_222234/v1", start: 1_266_618_157, gps: 1_266_618_172.4),
        Source(name: "GW191109", folder: "GWTC-3-confident/GW191109_010717/v1", start: 1_257_296_840, gps: 1_257_296_855.2),
        Source(name: "GW190828_063405", folder: "GWTC-2.1-confident/GW190828_063405/v2", start: 1_251_009_248, gps: 1_251_009_263.7),
        Source(name: "GW200311", folder: "GWTC-3-confident/GW200311_115853/v1", start: 1_267_963_136, gps: 1_267_963_151.3),
    ]

    /// Downloads (or reads from the cache) both detectors' strain and processes it off the main thread.
    static func load(_ s: Source) async -> GWSignal? {
        async let h = Feed.data(s.url("H"), maxAge: 3650 * 86400)
        async let l = Feed.data(s.url("L"), maxAge: 3650 * 86400)
        guard let hd = await h, let ld = await l else { return nil }
        let start = Double(s.start), gps = s.gps
        return await Task.detached(priority: .utility) {
            guard let hs = strain(gunzip(hd)), let ls = strain(gunzip(ld)) else { return nil }
            return process(hs, ls, start: start, gps: gps)
        }.value
    }

    // MARK: Files

    /// The body of a gzip file: skips the header and its optional fields, then inflates the raw deflate stream.
    static func gunzip(_ data: Data) -> Data? {
        let b = [UInt8](data)
        guard b.count > 18, b[0] == 0x1f, b[1] == 0x8b, b[2] == 8 else { return nil }
        let flags = b[3]
        var i = 10
        if flags & 4 != 0 { guard i + 2 <= b.count else { return nil }; i += 2 + Int(b[i]) + Int(b[i + 1]) << 8 }
        if flags & 8 != 0 { while i < b.count, b[i] != 0 { i += 1 }; i += 1 }
        if flags & 16 != 0 { while i < b.count, b[i] != 0 { i += 1 }; i += 1 }
        if flags & 2 != 0 { i += 2 }
        guard i < b.count - 8 else { return nil }
        let body = Data(b[i..<(b.count - 8)])
        return try? (body as NSData).decompressed(using: .zlib) as Data
    }

    /// One number per line after `#` comments, as GWOSC's text files have them. Gaps (NaN) become zero.
    static func strain(_ data: Data?) -> [Double]? {
        guard let data, !data.isEmpty else { return nil }
        var out: [Double] = []
        out.reserveCapacity(131_072)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: CChar.self) else { return }
            let n = raw.count
            var line = [CChar](repeating: 0, count: 64)
            var len = 0
            func flush() {
                if len > 0, line[0] != 35 {   // "#"
                    line[min(len, 63)] = 0
                    let v = strtod(line, nil)
                    out.append(v.isFinite ? v : 0)
                }
                len = 0
            }
            for k in 0..<n {
                let c = base[k]
                if c == 10 || c == 13 { flush() } else if len < 63 { line[len] = c; len += 1 }
            }
            flush()
        }
        return out.count >= 65536 ? out : nil
    }

    // MARK: Processing

    static let fileRate = 4096.0

    /// Whitens both recordings, finds the merger near the catalogue time, lines Livingston up with Hanford and maps
    /// the energy over time and frequency.
    static func process(_ h: [Double], _ l: [Double], start: Double, gps catalogued: Double) -> GWSignal? {
        let n = 1 << Int(log2(Double(min(h.count, l.count))))
        guard n >= 65536 else { return nil }
        guard var wh = whiten(Array(h.prefix(n))), var wl = whiten(Array(l.prefix(n))) else { return nil }
        let fs = fileRate
        let center = Int(((catalogued - start) * fs).rounded())
        // Unit noise: the median absolute deviation well away from the event.
        for k in 0..<2 {
            var quiet: [Double] = []
            let w = k == 0 ? wh : wl
            for i in stride(from: 0, to: n, by: 3) where abs(i - center) > Int(2 * fs) && i > Int(fs) && i < n - Int(fs) {
                quiet.append(abs(w[i]))
            }
            quiet.sort()
            let sigma = max(1.4826 * (quiet.isEmpty ? 1 : quiet[quiet.count / 2]), 1e-12)
            if k == 0 { wh = wh.map { $0 / sigma } } else { wl = wl.map { $0 / sigma } }
        }
        // The merger: the loudest 10 ms of both detectors together within 0.15 s of the catalogue time.
        let box = Int(0.01 * fs), reach = Int(0.15 * fs)
        var best = center, bestValue = -1.0
        var running = 0.0
        let lo = max(center - reach - box, 0), hi = min(center + reach + box, n - 1)
        var window: [Double] = []
        for i in lo...hi {
            let e = wh[i] * wh[i] + wl[i] * wl[i]
            window.append(e)
            running += e
            if window.count > box { running -= window[window.count - box - 1] }
            let mid = i - box / 2
            if abs(mid - center) <= reach, running > bestValue { bestValue = running; best = mid }
        }
        // Livingston's delay and sign: the lag within ±11 ms that best matches Hanford over the last quarter second.
        let a = max(best - Int(0.25 * fs), 0), b = min(best + Int(0.05 * fs), n - 1)
        var lag = 0, corr = 0.0
        for d in -45...45 {
            var s = 0.0, sh = 0.0, sl = 0.0
            for i in a...b where i + d >= 0 && i + d < n {
                s += wh[i] * wl[i + d]
                sh += wh[i] * wh[i]
                sl += wl[i + d] * wl[i + d]
            }
            let c = s / max((sh * sl).squareRoot(), 1e-12)
            if abs(c) > abs(corr) { corr = c; lag = d }
        }
        let sign = corr < 0 ? -1.0 : 1.0

        // Strain around the merger at 2048 Hz (pairs averaged; the band stops at 400 Hz).
        let first = best - Int(GWSignal.before * fs), last = best + Int(GWSignal.after * fs)
        var hs: [Float] = [], ls: [Float] = []
        for i in stride(from: first, to: last, by: 2) {
            func at(_ w: [Double], _ j: Int) -> Double { j >= 0 && j + 1 < n ? 0.5 * (w[j] + w[j + 1]) : 0 }
            hs.append(Float(at(wh, i)))
            ls.append(Float(sign * at(wl, i + lag)))
        }
        let eh = scalogram(wh, around: best)
        let el = scalogram(wl, around: best + lag)
        guard let eh, let el else { return nil }
        return GWSignal(hanford: hs, livingston: ls, energyH: eh.energy, energyL: el.energy, columns: eh.columns,
                        lag: Double(lag) / fs * 1000, flipped: sign < 0, gps: start + Double(best) / fs)
    }

    /// Whitens a recording: divides its spectrum by the noise amplitude (Welch's median of 4 s Hann segments) and keeps
    /// 30–400 Hz with soft edges. Tukey-tapered at both ends so they don't ring.
    static func whiten(_ x: [Double]) -> [Double]? {
        let n = x.count, fs = fileRate
        let seg = Int(4 * fs)
        guard n >= 2 * seg else { return nil }
        // Welch: median power in each bin across half-overlapping segments.
        let hann = (0..<seg).map { 0.5 - 0.5 * cos(2 * .pi * Double($0) / Double(seg)) }
        var segments: [[Double]] = []
        for s in stride(from: 0, through: n - seg, by: seg / 2) {
            let part = (0..<seg).map { x[s + $0] * hann[$0] }
            segments.append(power(part))
        }
        let bins = seg / 2 + 1
        var psd = [Double](repeating: 0, count: bins)
        for k in 0..<bins {
            var v = segments.map { $0[k] }
            v.sort()
            psd[k] = max(v[v.count / 2], 1e-60)
        }
        // Taper, transform, divide by the noise amplitude, band-limit, transform back.
        let taper = Int(0.5 * fs)
        var y = x
        for i in 0..<taper {
            let w = 0.5 - 0.5 * cos(.pi * Double(i) / Double(taper))
            y[i] *= w
            y[n - 1 - i] *= w
        }
        guard var spectrum = FFT.real(y) else { return nil }
        let df = fs / Double(n), dSeg = fs / Double(seg)
        for k in 0..<spectrum.count {
            let f = Double(k) * df
            let j = f / dSeg
            let j0 = min(Int(j), bins - 2)
            let p = psd[j0] + (psd[j0 + 1] - psd[j0]) * (j - Double(j0))
            let up = min(max((f - 30) / 10, 0), 1), down = min(max((400 - f) / 60, 0), 1)
            let band = 0.5 - 0.5 * cos(.pi * up * down)
            let g = k == 0 || k == spectrum.count - 1 ? 0 : band / p.squareRoot()
            spectrum[k] = (spectrum[k].0 * g, spectrum[k].1 * g)
        }
        return FFT.inverseReal(spectrum, count: n)
    }

    /// Power in each bin of a real signal (length a power of two).
    private static func power(_ x: [Double]) -> [Double] {
        (FFT.real(x) ?? []).map { $0.0 * $0.0 + $0.1 * $0.1 }
    }

    /// Energy over noise in `GWSignal.rows` Gaussian bands (quality factor 6) from `before` to `after` seconds around
    /// sample `center`, one column per millisecond: the analytic signal of each band, squared, over its median.
    static func scalogram(_ w: [Double], around center: Int) -> (energy: [Float], columns: Int)? {
        let fs = fileRate
        let m = 32768
        let first = center - m / 2
        var re = [Double](repeating: 0, count: m), im = [Double](repeating: 0, count: m)
        for i in 0..<m where first + i >= 0 && first + i < w.count { re[i] = w[first + i] }
        guard FFT.complex(&re, &im, inverse: false) else { return nil }
        let columns = Int((GWSignal.before + GWSignal.after) * GWSignal.columnRate)
        let step = fs / GWSignal.columnRate
        var energy = [Float](repeating: 0, count: GWSignal.rows * columns)
        var br = [Double](repeating: 0, count: m), bi = [Double](repeating: 0, count: m)
        for row in 0..<GWSignal.rows {
            let f0 = GWSignal.frequency(row: Double(row))
            let sigma = f0 / 6
            for k in 0..<m {
                let f = Double(k) * fs / Double(m)
                let g = k < m / 2 ? 2 * exp(-0.5 * pow((f - f0) / sigma, 2)) : 0
                br[k] = re[k] * g
                bi[k] = im[k] * g
            }
            guard FFT.complex(&br, &bi, inverse: true) else { return nil }
            var e = (0..<m).map { br[$0] * br[$0] + bi[$0] * bi[$0] }
            var middle = Array(e[(m / 8)..<(m - m / 8)])
            middle.sort()
            let median = max(middle[middle.count / 2], 1e-30)
            // Mean of |a|² for Gaussian noise is median / ln 2.
            let unit = median / log(2.0)
            for k in 0..<m { e[k] /= unit }
            for c in 0..<columns {
                let t = Double(c) / GWSignal.columnRate - GWSignal.before
                let s = Double(m / 2) + t * fs
                let i0 = Int(s), i1 = min(Int(s + step), m)
                guard i0 >= 0, i1 > i0 else { continue }
                var sum = 0.0
                for i in i0..<i1 { sum += e[i] }
                energy[row * columns + c] = Float(sum / Double(i1 - i0))
            }
        }
        return (energy, columns)
    }

    /// Schwarzschild light bending: the exact deflection angle (radians) for impact parameters b = √27 M (1 + s),
    /// s from 10⁻⁶ to 10², log-spaced, by integrating the photon orbit. Past the table the weak-field series takes over;
    /// below it, the strong-deflection limit.
    static func deflectionTable(_ count: Int = 256) -> [Float] {
        let critical = 27.0.squareRoot()
        return (0..<count).map { k in
            let s = pow(10, -6 + 8 * Double(k) / Double(count - 1))
            return Float(deflection(critical * (1 + s)))
        }
    }

    /// α(b) = 2∫₀^{u₀} du / √(1/b² − u² + 2u³) − π, with M = 1 and u₀ the inverse closest approach, written so the
    /// root at u₀ cancels, then integrated on a logarithmic grid that resolves the near-critical orbits.
    static func deflection(_ b: Double) -> Double {
        let r0 = 2 * b / 3.0.squareRoot() * cos(acos(-3 * 3.0.squareRoot() / b) / 3)
        let u0 = 1 / r0
        let steps = 4000
        let y0 = log(1e-9)
        var sum = 0.0, last = 0.0
        for i in 0...steps {
            let y = y0 * (1 - Double(i) / Double(steps))
            let t = exp(y)
            let u = u0 * (1 - t * t)
            let hh = max((u0 + u) - 2 * (u0 * u0 + u0 * u + u * u), 1e-30)
            let f = 2 * u0.squareRoot() / hh.squareRoot() * t
            if i > 0 { sum += 0.5 * (f + last) * (-y0 / Double(steps)) }
            last = f
        }
        let head = 2 * u0.squareRoot() / max(2 * u0 * (1 - 3 * u0), 1e-30).squareRoot() * 1e-9
        return 2 * (sum + head) - .pi
    }
}

/// Real fast Fourier transforms through Accelerate, for power-of-two lengths.
enum FFT {
    /// Bins 0...n/2 of a real signal's spectrum as (re, im).
    static func real(_ x: [Double]) -> [(Double, Double)]? {
        let n = x.count
        let log2n = vDSP_Length(log2(Double(n)))
        guard 1 << Int(log2n) == n, let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        defer { vDSP_destroy_fftsetupD(setup) }
        var re = [Double](repeating: 0, count: n / 2), im = [Double](repeating: 0, count: n / 2)
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                x.withUnsafeBufferPointer { xp in
                    xp.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: n / 2) {
                        vDSP_ctozD($0, 2, &split, 1, vDSP_Length(n / 2))
                    }
                }
                vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
            }
        }
        // zrip packs the Nyquist bin into im[0] and doubles everything.
        var out = [(Double, Double)](repeating: (0, 0), count: n / 2 + 1)
        out[0] = (re[0] / 2, 0)
        out[n / 2] = (im[0] / 2, 0)
        for k in 1..<(n / 2) { out[k] = (re[k] / 2, im[k] / 2) }
        return out
    }

    /// The real signal of length `count` with this spectrum (bins 0...count/2).
    static func inverseReal(_ spectrum: [(Double, Double)], count n: Int) -> [Double]? {
        let log2n = vDSP_Length(log2(Double(n)))
        guard 1 << Int(log2n) == n, spectrum.count == n / 2 + 1,
              let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        defer { vDSP_destroy_fftsetupD(setup) }
        var re = [Double](repeating: 0, count: n / 2), im = [Double](repeating: 0, count: n / 2)
        re[0] = spectrum[0].0
        im[0] = spectrum[n / 2].0
        for k in 1..<(n / 2) { re[k] = spectrum[k].0; im[k] = spectrum[k].1 }
        var out = [Double](repeating: 0, count: n)
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))
                out.withUnsafeMutableBufferPointer { op in
                    op.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: n / 2) {
                        vDSP_ztocD(&split, 1, $0, 2, vDSP_Length(n / 2))
                    }
                }
            }
        }
        let scale = 1 / Double(n)
        return out.map { $0 * scale }
    }

    /// In-place complex transform, unscaled forward and scaled by 1/n inverse.
    static func complex(_ re: inout [Double], _ im: inout [Double], inverse: Bool) -> Bool {
        let n = re.count
        let log2n = vDSP_Length(log2(Double(n)))
        guard 1 << Int(log2n) == n, im.count == n, let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { return false }
        defer { vDSP_destroy_fftsetupD(setup) }
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zipD(setup, &split, 1, log2n, FFTDirection(inverse ? kFFTDirection_Inverse : kFFTDirection_Forward))
            }
        }
        if inverse {
            let scale = 1 / Double(n)
            for k in 0..<n { re[k] *= scale; im[k] *= scale }
        }
        return true
    }
}
