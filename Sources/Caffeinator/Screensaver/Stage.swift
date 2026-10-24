import AppKit
import Metal
import MetalKit

/// Plays scenes on one display. Renders the current scene at its own resolution, then composites it with the
/// overlay onto the display-sized output. Changing scenes fades through black.
@MainActor final class Stage {
    let gpu: GPU
    let overlay: Overlay
    /// Set on the main display's stage while sound is on.
    var sound: Synth? {
        didSet {
            if oldValue !== sound { oldValue?.reset() }
            if let kind { sound?.gain = kind.gain }
        }
    }
    private(set) var kind: SceneKind?
    private(set) var scene: Scene?
    /// Called when a scene begins.
    var onScene: (SceneKind) -> Void = { _ in }

    private let composite: MTLRenderPipelineState
    private var target: MTLTexture?
    private var pending: SceneKind?
    private var fade: Float = 0
    private var time: Float = 0
    private var index: UInt32 = 0
    private let seed = Float.random(in: 0..<1000)

    init(gpu: GPU, pointScale: CGFloat) throws {
        self.gpu = gpu
        overlay = Overlay(gpu: gpu, scale: pointScale)
        composite = try gpu.pipeline(gpu.library(Shaders.compositor), "composite", format: .bgra8Unorm)
    }

    /// Fades out the current scene, if any, and starts `kind`.
    func show(_ kind: SceneKind) { pending = kind }

    func render(_ cmd: MTLCommandBuffer, into output: MTLTexture, dt: Float, now: Date = .now) {
        if let next = pending {
            fade = scene == nil ? 0 : max(fade - dt / 0.8, 0)
            if fade == 0 { begin(next) }
        } else {
            fade = min(fade + dt / 1.6, 1)
        }
        overlay.resize(output.width, output.height)

        if let scene {
            let w = max(Int((Float(output.width) * scene.scale).rounded()), 1)
            let h = max(Int((Float(output.height) * scene.scale).rounded()), 1)
            if target?.width != w || target?.height != h { target = gpu.texture(w, h) }
            time += dt
            scene.draw(Frame(gpu: gpu, cmd: cmd, target: target!, time: time, dt: dt, index: index, overlay: overlay,
                             sound: sound, now: now, seed: seed))
            index &+= 1
        }
        overlay.upload()
        let eased = fade * fade * (3 - 2 * fade)
        let black = target ?? overlay.texture
        cmd.draw(composite, into: output, uniforms: U(res: SIMD2(Float(output.width), Float(output.height)), time: time,
                                                       a: SIMD4(scene == nil ? 0 : eased, 0, 0, 0)),
                 textures: [black, overlay.texture])
    }

    private func begin(_ next: SceneKind) {
        pending = nil
        scene = nil
        target = nil
        overlay.clear()
        sound?.reset()
        sound?.gain = next.gain
        time = 0
        index = 0
        kind = next
        do {
            scene = try next.make(gpu)
        } catch {
            FileHandle.standardError.write(Data("Scene \(next.id) failed: \(error)\n".utf8))
        }
        onScene(next)
    }
}

/// A CoreGraphics layer over the scene, for text and hairlines. Scenes draw in points, origin at the top left,
/// and only when something changes.
@MainActor final class Overlay {
    private let gpu: GPU
    let scale: CGFloat
    private(set) var texture: MTLTexture
    private var context: CGContext
    private var dirty = false

    init(gpu: GPU, scale: CGFloat) {
        self.gpu = gpu
        self.scale = scale
        context = Self.context(1, 1)
        texture = Self.texture(gpu, 1, 1)
    }

    /// The overlay's size in points.
    var size: CGSize { CGSize(width: CGFloat(context.width) / scale, height: CGFloat(context.height) / scale) }

    func resize(_ width: Int, _ height: Int) {
        guard width != context.width || height != context.height else { return }
        context = Self.context(width, height)
        texture = Self.texture(gpu, width, height)
        dirty = true
    }

    /// Clears the overlay and redraws it with `body`. NSString and NSAttributedString drawing works inside.
    func draw(_ body: (CGContext, CGSize) -> Void) {
        clear()
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(context.height))
        context.scaleBy(x: scale, y: -scale)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: true)
        body(context, size)
        NSGraphicsContext.restoreGraphicsState()
        context.restoreGState()
    }

    func clear() {
        context.clear(CGRect(x: 0, y: 0, width: context.width, height: context.height))
        dirty = true
    }

    func upload() {
        guard dirty, let data = context.data else { return }
        dirty = false
        texture.replace(region: MTLRegionMake2D(0, 0, context.width, context.height), mipmapLevel: 0, withBytes: data,
                        bytesPerRow: context.bytesPerRow)
    }

    private static func context(_ w: Int, _ h: Int) -> CGContext {
        CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    }

    private static func texture(_ gpu: GPU, _ w: Int, _ h: Int) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        d.usage = .shaderRead
        d.storageMode = gpu.device.hasUnifiedMemory ? .shared : .managed
        return gpu.device.makeTexture(descriptor: d)!
    }
}

extension CGContext {
    /// Draws a line of text. `anchor` places the point: (0, 0) is the text's top left, (0.5, 0.5) its center.
    @MainActor func text(_ string: String, at point: CGPoint, size: CGFloat, weight: NSFont.Weight = .regular, mono: Bool = false,
                         color: NSColor = .white, anchor: CGPoint = .zero, tracking: CGFloat = 0) {
        let font = mono ? NSFont.monospacedSystemFont(ofSize: size, weight: weight) : NSFont.systemFont(ofSize: size, weight: weight)
        let s = NSAttributedString(string: string, attributes: [.font: font, .foregroundColor: color, .kern: tracking])
        let b = s.size()
        s.draw(at: CGPoint(x: point.x - b.width * anchor.x, y: point.y - b.height * anchor.y))
    }
}

/// A full-screen Metal view that runs a stage on one display.
@MainActor final class SceneView: MTKView, MTKViewDelegate {
    /// Development only (`--preview`): keeps drawables readable so a frame can be captured.
    static var readable = false
    let stage: Stage
    /// Development only: receives a copy of the next frame.
    var capture: ((MTLTexture) -> Void)?
    private(set) var frames = 0
    private var last = CACurrentMediaTime()

    init(frame: NSRect, gpu: GPU, pointScale: CGFloat) throws {
        stage = try Stage(gpu: gpu, pointScale: pointScale)
        super.init(frame: frame, device: gpu.device)
        colorPixelFormat = .bgra8Unorm
        colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        framebufferOnly = !Self.readable
        delegate = self
        setAccessibilityElement(true)
        setAccessibilityRole(.image)
    }

    required init(coder: NSCoder) { fatalError("init(coder:) is not used") }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = Float(min(max(now - last, 0.001), 1.0 / 15))
        last = now
        guard let drawable = currentDrawable, let cmd = stage.gpu.queue.makeCommandBuffer() else { return }
        stage.render(cmd, into: drawable.texture, dt: dt)
        frames += 1
        var copy: MTLTexture?
        if capture != nil, Self.readable {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: drawable.texture.width,
                                                             height: drawable.texture.height, mipmapped: false)
            d.storageMode = .shared
            copy = stage.gpu.device.makeTexture(descriptor: d)
            if let copy, let blit = cmd.makeBlitCommandEncoder() {
                blit.copy(from: drawable.texture, to: copy)
                blit.endEncoding()
            }
        }
        cmd.present(drawable)
        cmd.commit()
        if let copy, let capture {
            self.capture = nil
            cmd.waitUntilCompleted()
            capture(copy)
        }
    }
}
