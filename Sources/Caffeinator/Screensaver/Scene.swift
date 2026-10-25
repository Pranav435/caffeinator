import Foundation
import Metal

/// One screensaver scene. Each frame it draws into a texture at its own resolution;
/// the stage scales that to the display and lays the text overlay on top.
@MainActor protocol Scene: AnyObject {
    /// Render size as a fraction of the display's pixels. Heavy ray-marched scenes use less.
    var scale: Float { get }
    /// False while live data is still on its way. The snapshot tool waits for it.
    var ready: Bool { get }
    /// Live facts for the caption and VoiceOver, such as "237 earthquakes today, largest M4.9 near Vanuatu".
    var facts: String? { get }
    /// Whose photograph or map is on screen, such as "Milky Way: ESO/S. Brunier". Shown in the caption, not read aloud.
    var credit: String? { get }
    func draw(_ f: Frame)
}

extension Scene {
    var scale: Float { 0.5 }
    var ready: Bool { true }
    var facts: String? { nil }
    var credit: String? { nil }
}

@MainActor struct Frame {
    let gpu: GPU
    let cmd: MTLCommandBuffer
    let target: MTLTexture
    /// Seconds since the scene began, and since the last frame.
    let time: Float
    let dt: Float
    let index: UInt32
    let overlay: Overlay
    /// Nil unless sound is on and this is the main display's scene.
    let sound: Synth?
    /// Wall-clock time, for scenes that show the world as it is.
    let now: Date
    /// Different on each display and each run, so two screens never show the same thing.
    let seed: Float

    var size: SIMD2<Float> { SIMD2(Float(target.width), Float(target.height)) }
    var aspect: Float { size.x / size.y }

    func uniforms(_ a: SIMD4<Float> = .zero, _ b: SIMD4<Float> = .zero, _ c: SIMD4<Float> = .zero, _ d: SIMD4<Float> = .zero,
                  res: SIMD2<Float>? = nil) -> U {
        U(res: res ?? size, time: time, dt: dt, frame: index, seed: seed, a: a, b: b, c: c, d: d)
    }
}

enum Family: String, CaseIterable, Identifiable {
    case math = "Math and Chaos"
    case physics = "Physics"
    case space = "Space"
    case earth = "Live Earth"
    var id: String { rawValue }
}

struct SceneKind: Identifiable {
    let id: String
    let name: String
    let family: Family
    /// What's on screen, in a sentence or two, for VoiceOver and Settings.
    let about: String
    let make: @MainActor (GPU) throws -> Scene
    /// False for scenes that are clocks themselves, so the screensaver's clock stays out of the way.
    var clock = true
    /// Scales the soundscape so every scene plays at about the same loudness (about −27 dBFS on average).
    var gain: Float = 1
}

@MainActor extension SceneKind {
    /// Every scene, grouped in Settings by family. Each folder under Scenes/ contributes its own list.
    static let all: [SceneKind] = mathScenes + fieldScenes + labScenes + spaceScenes + worldScenes + earthScenes + skyScenes

    static func named(_ id: String) -> SceneKind? { all.first { $0.id == id } }
}
