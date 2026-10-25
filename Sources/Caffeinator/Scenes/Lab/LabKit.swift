import Foundation

/// SplitMix64: a fast random source for the lab scenes, which scatter tens of thousands of droplets and grains a second.
struct LabRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64 = UInt64.random(in: 0...UInt64.max)) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func uniform(_ r: ClosedRange<Double> = 0...1) -> Double { Double.random(in: r, using: &self) }

    mutating func gaussian() -> Double {
        let u = max(uniform(), 1e-12), v = uniform()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}

/// Metal helpers the lab scenes share: they are filmed, so they need the same optics.
enum LabShaders {
    /// Henyey–Greenstein scattering and a few small helpers.
    static let common = """
    inline float sq(float x) { return x * x; }
    // Henyey–Greenstein phase function for the cosine of the scattering angle, normalised to 1 at 90 degrees.
    inline float henyey(float c, float g) {
        return pow(1 + g * g, 1.5) / pow(max(1 + g * g - 2 * g * c, 1e-4), 1.5);
    }
    // Pixels per world unit at depth z for the header's camera, on a target `height` pixels tall.
    inline float pixelsPer(float z, float zoom, float height) { return 0.5 * height * zoom / max(z, 1e-3); }
    """
}
