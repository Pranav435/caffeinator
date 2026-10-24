import Foundation
import Metal
import Testing
@testable import Caffeinator

struct OrbitalTests {
    @Test(arguments: [(1, 0), (2, 0), (2, 1), (3, 2), (4, 3), (6, 5), (5, 2)])
    func radialPartIsNormalised(n: Int, l: Int) {
        let o = Orbital(n: n, l: l, m: 0)
        let dr = 0.01
        let total = stride(from: dr / 2, to: Double(n * n) * 6 + 40, by: dr).reduce(0.0) { sum, r in
            let v = o.radial(r)
            return sum + v * v * r * r * dr
        }
        #expect(abs(total - 1) < 1e-3)
    }

    @Test(arguments: [(0, 0), (1, 0), (1, 1), (2, 1), (3, 3), (5, 1)])
    func angularPartIsNormalised(l: Int, m: Int) {
        let o = Orbital(n: l + 1, l: l, m: m)
        let dx = 1e-4
        let total = stride(from: -1 + dx / 2, to: 1, by: dx).reduce(0.0) { sum, x in
            let y = o.angular(x)
            return sum + 2 * .pi * y * y * dx
        }
        #expect(abs(total - 1) < 1e-3)
    }

    @Test func energiesFollowRydberg() {
        #expect(abs(Orbital(n: 1, l: 0, m: 0).energy + 13.5984) < 1e-3)
        #expect(abs(Orbital(n: 4, l: 3, m: 0).energy + 0.8499) < 1e-3)
        #expect(Hydrogen.eV(Orbital(n: 4, l: 3, m: 0).energy) == "−0.85 eV")
    }

    @Test func spectralLinesKeepTheirRatios() {
        #expect(Hydrogen.line(2, 3) == "Balmer alpha")
        #expect(Hydrogen.line(1, 2) == "Lyman alpha")
        #expect(abs(Hydrogen.lineTone(2, 3) - 392) < 0.01)
        // Hβ / Hα = (1/4 − 1/16) / (1/4 − 1/9) = 27/20, close to a perfect fourth.
        #expect(abs(Hydrogen.lineTone(2, 4) / Hydrogen.lineTone(2, 3) - 27.0 / 20) < 1e-4)
        let lymanAlpha = Hydrogen.frequency(Orbital(n: 1, l: 0, m: 0), Orbital(n: 2, l: 1, m: 0))
        #expect(abs(lymanAlpha / 1e12 - 2466) < 1)
    }
}

struct PlateTests {
    @Test func besselMatchesKnownValues() {
        #expect(abs(PlateMode.bessel(0, 0)) - 1 < 1e-9)
        #expect(abs(PlateMode.bessel(0, 2.404826)) < 1e-5)
        #expect(abs(PlateMode.bessel(1, 3.831706)) < 1e-5)
    }

    @Test(arguments: [(0, 1, 3.831706), (1, 1, 1.841184), (2, 1, 3.054237), (2, 2, 6.706133), (3, 2, 8.015237)])
    func turningPointsAreZerosOfTheDerivative(m: Int, n: Int, expected: Double) {
        #expect(abs(PlateMode.turningPoint(m, n) - expected) < 1e-3)
    }

    @Test func squareModeFrequencyScalesWithWavenumber() {
        #expect(abs(PlateMode.square(3, 5, false).hertz - 36 * 34) < 1e-9)
        #expect(PlateMode.square(1, 2, true).hertz < PlateMode.square(3, 5, true).hertz)
    }
}

struct SlitTests {
    @Test func narrowSlitPassesItsShareOfTheGaussian() {
        let sigma: Float = 50
        let one = DoubleSlit.Setup(slits: 1, spacing: 0, width: 2, k: 0.6).opening(sigma)
        #expect(abs(one - 2 / (sigma * (2 * Float.pi).squareRoot())) < 1e-4)
        let wide = DoubleSlit.Setup(slits: 1, spacing: 0, width: 10_000, k: 0.6).opening(sigma)
        #expect(abs(wide - 1) < 1e-4)
        let two = DoubleSlit.Setup(slits: 2, spacing: 50, width: 8, k: 0.6).opening(sigma)
        #expect(two > 0.1 && two < 0.13)
    }
}

/// Runs every lab scene for a few seconds with a (silent) synth attached, so the sound and facts paths run too.
@MainActor struct LabSceneTests {
    @Test func scenesRunWithSound() throws {
        guard let gpu = GPU.shared else { return }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 640, height: 400, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        let output = try #require(gpu.device.makeTexture(descriptor: d))
        for kind in SceneKind.labScenes {
            let stage = try Stage(gpu: gpu, pointScale: 1)
            stage.sound = Synth()
            stage.show(kind)
            for _ in 0..<150 {
                let cmd = try #require(gpu.queue.makeCommandBuffer())
                stage.render(cmd, into: output, dt: 1.0 / 30)
                cmd.commit()
                cmd.waitUntilCompleted()
                #expect(cmd.error == nil)
            }
            #expect(stage.scene != nil, "\(kind.id) did not start")
            #expect(stage.scene?.facts?.isEmpty == false, "\(kind.id) has no facts")
        }
    }
}
