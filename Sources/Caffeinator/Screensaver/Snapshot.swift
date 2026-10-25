import AppKit
import AVFoundation
import Metal

/// Renders scenes offscreen to PNG files, to check how they look without sitting in front of the screen:
///
///     Caffeinator --snapshot attractors,black-hole ~/Desktop/shots --at 2,30 --size 1600x1000 --variant 2
///
/// `all` renders every scene. Times are scene seconds, simulated at 30 frames a second. `--variant` asks scenes
/// that cycle through several looks to start at that one. `--audio` also records each soundscape to <scene>.m4a
/// and prints its peak and average level.
@MainActor enum Snapshot {
    private(set) static var variant: Int?

    static func run(_ args: [String]) -> Int32 {
        guard args.count >= 2, let gpu = GPU.shared else {
            print("Usage: Caffeinator --snapshot <scene,scene|all> <folder> [--at 2,10] [--size 1600x1000] [--variant n] [--audio]")
            return 1
        }
        var times: [Float] = [3]
        var width = 1600, height = 1000
        var audio = false
        var rest = args.dropFirst(2).makeIterator()
        while let flag = rest.next() {
            if flag == "--audio" { audio = true; continue }
            let value = rest.next() ?? ""
            switch flag {
            case "--at": times = value.split(separator: ",").compactMap { Float($0) }.sorted()
            case "--size":
                let wh = value.split(separator: "x").compactMap { Int($0) }
                if wh.count == 2 { (width, height) = (wh[0], wh[1]) }
            case "--variant": variant = Int(value)
            default: print("Unknown option \(flag)")
            }
        }
        Location.shared.holdPrompt = true
        let kinds = args[0] == "all" ? SceneKind.all : args[0].split(separator: ",").compactMap { SceneKind.named(String($0)) }
        let folder = URL(fileURLWithPath: (args[1] as NSString).expandingTildeInPath)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        guard let output = gpu.device.makeTexture(descriptor: d) else { return 1 }

        for kind in kinds {
            let started = Date()
            guard let stage = try? Stage(gpu: gpu, pointScale: CGFloat(width) / 1470) else { return 1 }
            stage.show(kind)
            let dt: Float = 1.0 / 30
            let recorder = audio ? Recorder(folder.appendingPathComponent("\(kind.id).m4a")) : nil
            stage.sound = recorder?.synth
            frame(stage, output, dt: dt)
            // The first run downloads photographs, some several megabytes.
            let deadline = Date() + 180
            while stage.scene?.ready == false, Date() < deadline { RunLoop.main.run(until: .now + 0.05) }
            var t: Float = dt
            for at in times {
                var gpu: [Double] = []
                while t < at {
                    gpu.append(frame(stage, output, dt: dt))
                    recorder?.record(Double(dt))
                    t += dt
                    if Int(t / dt) % 30 == 0 { RunLoop.main.run(until: .now + 0.001) }
                }
                let file = folder.appendingPathComponent("\(kind.id)-\(Int(at)).png")
                save(output, to: file)
                // GPU time per frame over the last second, at this snapshot size.
                let recent = gpu.suffix(30).sorted()
                let ms = recent.isEmpty ? 0 : recent[recent.count / 2] * 1000
                print(String(format: "%@ at %.0fs (%.1fs, GPU %.1f ms/frame) %@", kind.id, at, Date().timeIntervalSince(started), ms,
                             stage.scene?.facts ?? ""))
            }
            if let recorder { print(recorder.summary) }
        }
        return 0
    }

    /// Shows the real screensaver windows for a few seconds, saves what the main window shows (its Metal frame with
    /// the caption and clock over it), then quits. Screen capture permission isn't needed.
    ///     Caffeinator --preview <scene> <seconds> <window.png>
    static func preview(_ args: [String]) -> Never {
        Prefs.register()
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        SceneView.readable = true
        let seconds = Double(args.count > 1 ? args[1] : "") ?? 6
        Task { @MainActor in
            let saver = Screensaver.shared
            saver.start(scene: args.first)
            try? await Task.sleep(for: .seconds(seconds / 2))
            if args.count > 2, let window = saver.windows.first {
                window.sceneView.capture = { texture in
                    capture(window, texture, to: URL(fileURLWithPath: args[2]))
                    print("Frames drawn: \(window.sceneView.frames), scene: \(window.stage.kind?.id ?? "none"), key window: \(window.isKeyWindow)")
                }
            }
            try? await Task.sleep(for: .seconds(seconds / 2))
            // Press → then Escape through the normal event path, and report what happened.
            @MainActor func press(_ code: UInt16, _ character: String) {
                guard let window = saver.windows.first else { return }
                NSApp.postEvent(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                                 windowNumber: window.windowNumber, context: nil, characters: character,
                                                 charactersIgnoringModifiers: character, isARepeat: false, keyCode: code)!, atStart: false)
            }
            let before = saver.windows.first?.stage.kind?.id
            press(124, "\u{F703}")
            try? await Task.sleep(for: .seconds(1.5))
            print("Right arrow: \(before ?? "none") → \(saver.windows.first?.stage.kind?.id ?? "none")")
            press(53, "\u{1B}")
            try? await Task.sleep(for: .seconds(0.5))
            print("After Escape, showing: \(saver.isShowing)")
            saver.stop(byUser: false)
            exit(0)
        }
        app.run()
        exit(0)
    }

    private static func capture(_ window: SaverWindow, _ texture: MTLTexture, to url: URL) {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        texture.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        // The AppKit labels, drawn without the Metal layer, on top.
        if let view = window.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
            window.sceneView.isHidden = true
            view.cacheDisplay(in: view.bounds, to: rep)
            window.sceneView.isHidden = false
            if let labels = rep.cgImage { ctx.draw(labels, in: CGRect(x: 0, y: 0, width: w, height: h)) }
        }
        let png = NSBitmapImageRep(cgImage: ctx.makeImage()!).representation(using: .png, properties: [:])
        try? png?.write(to: url)
    }

    /// Records a scene's soundscape offline and measures it.
    private final class Recorder {
        let synth = Synth(offline: true)
        private let file: AVAudioFile?
        private var peak: Float = 0
        private var sum: Double = 0
        private var count = 0

        init(_ url: URL) {
            synth.start()
            file = try? AVAudioFile(forWriting: url, settings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48000,
                                                                 AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 160_000])
        }

        func record(_ seconds: Double) {
            guard let buffer = synth.render(seconds: seconds) else { return }
            for c in 0..<Int(buffer.format.channelCount) {
                let samples = UnsafeBufferPointer(start: buffer.floatChannelData![c], count: Int(buffer.frameLength))
                for x in samples {
                    peak = max(peak, abs(x))
                    sum += Double(x * x)
                }
                count += samples.count
            }
            try? file?.write(from: buffer)
        }

        var summary: String {
            let rms = count > 0 ? (sum / Double(count)).squareRoot() : 0
            let db = { (x: Double) in x > 0 ? String(format: "%.1f dBFS", 20 * log10(x)) : "silent" }
            return "  sound: peak \(db(Double(peak))), average \(db(rms))"
        }
    }

    @discardableResult
    private static func frame(_ stage: Stage, _ output: MTLTexture, dt: Float) -> Double {
        guard let cmd = stage.gpu.queue.makeCommandBuffer() else { return 0 }
        stage.render(cmd, into: output, dt: dt)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let error = cmd.error { print("GPU error: \(error)") }
        return cmd.gpuEndTime - cmd.gpuStartTime
    }

    private static func save(_ texture: MTLTexture, to url: URL) {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        texture.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        let png = NSBitmapImageRep(cgImage: ctx.makeImage()!).representation(using: .png, properties: [:])
        try? png?.write(to: url)
    }
}
