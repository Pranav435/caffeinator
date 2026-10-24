import Metal

/// The Metal device, plus the few helpers every scene draws with. Shaders compile from source at runtime,
/// so scenes need no build step and the package stays a plain `swift build`.
@MainActor final class GPU {
    static let shared = MTLCreateSystemDefaultDevice().flatMap { GPU($0) }

    let device: MTLDevice
    let queue: MTLCommandQueue
    private var libraries: [String: MTLLibrary] = [:]

    private init?(_ device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
    }

    /// Compiles a scene's Metal source with the shared header in front. `precise` turns off fast math,
    /// for shaders that rely on exact rounding such as double-float arithmetic.
    func library(_ source: String, precise: Bool = false) throws -> MTLLibrary {
        let key = (precise ? "p" : "f") + source
        if let library = libraries[key] { return library }
        let options = MTLCompileOptions()
        if #available(macOS 15, *) {
            options.mathMode = precise ? .safe : .fast
        } else {
            options.fastMathEnabled = !precise
        }
        let library = try device.makeLibrary(source: Shaders.header + source, options: options)
        libraries[key] = library
        return library
    }

    /// `fade` multiplies what's already in the target by the fragment's alpha, for trails (see `fadeOut` in the header).
    enum Blend { case none, add, alpha, fade }

    func pipeline(_ library: MTLLibrary, _ fragment: String, vertex: String = "fullscreen",
                  format: MTLPixelFormat = .rgba16Float, blend: Blend = .none) throws -> MTLRenderPipelineState {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = try function(library, vertex)
        d.fragmentFunction = try function(library, fragment)
        let a = d.colorAttachments[0]!
        a.pixelFormat = format
        if blend != .none {
            a.isBlendingEnabled = true
            a.sourceRGBBlendFactor = blend == .fade ? .zero : .one
            a.sourceAlphaBlendFactor = blend == .fade ? .zero : .one
            let destination: MTLBlendFactor = switch blend {
            case .add: .one
            case .fade: .sourceAlpha
            default: .oneMinusSourceAlpha
            }
            a.destinationRGBBlendFactor = destination
            a.destinationAlphaBlendFactor = destination
        }
        return try device.makeRenderPipelineState(descriptor: d)
    }

    func kernel(_ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: function(library, name))
    }

    private func function(_ library: MTLLibrary, _ name: String) throws -> MTLFunction {
        guard let f = library.makeFunction(name: name) else { throw SceneError("No Metal function named \(name)") }
        return f
    }

    /// A GPU-only texture that shaders can sample, write and render into.
    func texture(_ width: Int, _ height: Int, _ format: MTLPixelFormat = .rgba16Float) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(width, 1), height: max(height, 1), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite, .renderTarget]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)!
    }

    /// A texture the CPU fills, for images and data grids.
    func texture(_ width: Int, _ height: Int, _ format: MTLPixelFormat, bytes: UnsafeRawPointer, bytesPerRow: Int) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = device.hasUnifiedMemory ? .shared : .managed
        let t = device.makeTexture(descriptor: d)!
        t.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: bytesPerRow)
        return t
    }

    func buffer(_ length: Int) -> MTLBuffer {
        device.makeBuffer(length: max(length, 16), options: .storageModeShared)!
    }

    func buffer<T>(_ array: [T]) -> MTLBuffer {
        array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
    }
}

struct SceneError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// Per-pass constants, bound at buffer 0 of every shader stage. Matches `struct U` in `Shaders.header`.
struct U {
    var res: SIMD2<Float> = .zero
    var time: Float = 0
    var dt: Float = 0
    var frame: UInt32 = 0
    var seed: Float = 0
    var pad: SIMD2<Float> = .zero
    var a: SIMD4<Float> = .zero
    var b: SIMD4<Float> = .zero
    var c: SIMD4<Float> = .zero
    var d: SIMD4<Float> = .zero
}

extension MTLCommandBuffer {
    /// Runs a render pass into `target`. Without a vertex count it draws one full-screen triangle.
    /// Uniforms go to buffer 0, `buffers` to 1 and up, `textures` to 0 and up, in both stages.
    func draw(_ pipeline: MTLRenderPipelineState, into target: MTLTexture, clear: Bool = true, uniforms: U,
              textures: [MTLTexture] = [], buffers: [MTLBuffer] = [], vertices: Int = 3, instances: Int = 1,
              primitive: MTLPrimitiveType = .triangle) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = clear ? .clear : .load
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        guard let e = makeRenderCommandEncoder(descriptor: pass) else { return }
        e.setRenderPipelineState(pipeline)
        var u = uniforms
        e.setVertexBytes(&u, length: MemoryLayout<U>.stride, index: 0)
        e.setFragmentBytes(&u, length: MemoryLayout<U>.stride, index: 0)
        for (i, b) in buffers.enumerated() {
            e.setVertexBuffer(b, offset: 0, index: i + 1)
            e.setFragmentBuffer(b, offset: 0, index: i + 1)
        }
        for (i, t) in textures.enumerated() {
            e.setVertexTexture(t, index: i)
            e.setFragmentTexture(t, index: i)
        }
        if vertices > 0, instances > 0 {
            e.drawPrimitives(type: primitive, vertexStart: 0, vertexCount: vertices, instanceCount: instances)
        }
        e.endEncoding()
    }

    /// Dispatches a compute kernel over a width × height grid, with the same binding layout as `draw`.
    func compute(_ pipeline: MTLComputePipelineState, _ width: Int, _ height: Int = 1, uniforms: U,
                 textures: [MTLTexture] = [], buffers: [MTLBuffer] = []) {
        guard width > 0, height > 0, let e = makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipeline)
        var u = uniforms
        e.setBytes(&u, length: MemoryLayout<U>.stride, index: 0)
        for (i, b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i + 1) }
        for (i, t) in textures.enumerated() { e.setTexture(t, index: i) }
        let w = pipeline.threadExecutionWidth
        let group = height == 1 ? MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
                                : MTLSize(width: w, height: max(pipeline.maxTotalThreadsPerThreadgroup / w / 2, 1), depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
        e.endEncoding()
    }
}
