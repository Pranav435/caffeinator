import Foundation
import Testing
@testable import Caffeinator

struct MandelbrotMathTests {
    @Test func doubleDoubleKeepsDigitsPastDouble() {
        let x = MandelbrotDD("-0.743643887037158704752191506114774")
        #expect(x.hi == -0.743643887037158704752191506114774)
        #expect(x.lo != 0 && abs(x.lo) < 1e-16)
        let third = MandelbrotDD(1) / MandelbrotDD(3)
        let back = third * MandelbrotDD(3) - MandelbrotDD(1)
        #expect(abs(back.value) < 1e-30)
        let tenth = MandelbrotDD("0.1") * MandelbrotDD(10) - MandelbrotDD(1)
        #expect(abs(tenth.value) < 1e-30)
    }

    @Test func referenceOrbits() {
        // c = -1 cycles 0, -1, 0, -1 forever; c = 1 runs 0, 1, 2, 5, 26, 677, 458330 and escapes.
        #expect(MandelbrotRefiner.orbit(MandelbrotDD(-1), MandelbrotDD(0), limit: 100).count == 101)
        let escaping = MandelbrotRefiner.orbit(MandelbrotDD(1), MandelbrotDD(0), limit: 100)
        #expect(escaping.count == 7)
        #expect(escaping[5].x == 677)
    }

    @Test func perturbationProbe() {
        let ref = MandelbrotRefiner.orbit(MandelbrotDD(-1), MandelbrotDD(0), limit: 1000)
        ref.withUnsafeBufferPointer { r in
            #expect(MandelbrotRefiner.probe(r, 0, 0, cap: 1000) < 0)
            // Offset by 2 from -1 is c = 1, which escapes on its sixth step; the smooth count is 7 - log2(ln 458330).
            let n = MandelbrotRefiner.probe(r, 2, 0, cap: 1000)
            #expect(abs(n - (7 - log2(log(458_330.0)))) < 1e-9)
        }
    }

    @MainActor @Test func spokenNumbers() {
        #expect(Mandelbrot.words(3.2e9) == "3.2 billion")
        #expect(Mandelbrot.words(16e12) == "16 trillion")
        #expect(Mandelbrot.words(12) == "12")
        #expect(Mandelbrot.complex(-0.7436438, 0.1318259) == "minus 0.743644 plus 0.131826i")
    }

    @MainActor @Test func zoomEasing() {
        #expect(Mandelbrot.ease(0, 0.05) == 0)
        #expect(abs(Mandelbrot.ease(1, 0.05) - 1) < 1e-12)
        #expect(abs(Mandelbrot.ease(0.5, 0.05) - 0.5) < 1e-12)
        #expect(Mandelbrot.ease(0.3, 0.05) < Mandelbrot.ease(0.31, 0.05))
    }
}

struct LeniaTests {
    @MainActor @Test func findsCreaturesAcrossTheWrap() {
        let w = 40, h = 30
        var field = [Float](repeating: 0, count: w * h)
        for y in 10..<16 {
            for x in [37, 38, 39, 0, 1, 2] { field[y * w + x] = 1 }
        }
        // A speck too small to count as a creature.
        field[5 * w + 20] = 0.5
        let found = field.withUnsafeBufferPointer { Lenia.find($0.baseAddress!, w, h) }
        #expect(found.count == 1)
        #expect(abs(found[0].mass - 36) < 1e-4)
        #expect(abs(found[0].x - 39.5) < 1e-3)
        #expect(abs(found[0].y - 12.5) < 1e-3)
    }
}

struct PendulumTests {
    @MainActor @Test func anglesReadAsDegrees() {
        #expect(Pendulums.degrees(-2.0) == "115 degrees left")
        #expect(Pendulums.degrees(.pi / 2) == "90 degrees right")
    }
}
