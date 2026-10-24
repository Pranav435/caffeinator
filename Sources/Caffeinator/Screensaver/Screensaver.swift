import AppKit
import Carbon.HIToolbox
import IOKit.ps
import IOKit.pwr_mgt

/// Covers every display with a scene while keeping the Mac awake. Only Escape or the shortcut ends it, so stray keys,
/// VoiceOver commands and a bumped trackpad don't. ← and → change the scene, Space speaks what's showing, M toggles sound.
@MainActor final class Screensaver {
    static let shared = Screensaver()

    weak var engine: Engine?
    private(set) var isShowing = false
    private(set) var windows: [SaverWindow] = []
    private var synth: Synth?
    private var shuffleTimer: Timer?
    private var clockTimer: Timer?
    private var recent: [String] = []
    private var usingSystem = false
    private var endedAt = Date.distantPast
    private var hintedAt = Date.distantPast
    /// The app that was in front, to hand focus back to afterwards.
    private var previous: NSRunningApplication?

    private init() {
        let workspace = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.willSleepNotification, NSWorkspace.screensDidSleepNotification, NSWorkspace.sessionDidResignActiveNotification] {
            workspace.addObserver(forName: name, object: nil, queue: .main) { _ in
                MainActor.assumeIsolated { Screensaver.shared.stop(byUser: false) }
            }
        }
        DistributedNotificationCenter.default().addObserver(forName: .init("com.apple.screenIsLocked"), object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { Screensaver.shared.stop(byUser: false) }
        }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { Screensaver.shared.displaysChanged() }
        }
        // While it's up, the screensaver keeps the keyboard, so Escape and the arrow keys always reach it.
        NotificationCenter.default.addObserver(forName: NSWindow.didResignKeyNotification, object: nil, queue: .main) { note in
            let window = note.object as? NSWindow
            MainActor.assumeIsolated {
                let saver = Screensaver.shared
                guard saver.isShowing, let main = saver.windows.first, window === main else { return }
                Task { main.makeKeyAndOrderFront(nil) }
            }
        }
        workspace.addObserver(forName: NSWorkspace.didTerminateApplicationNotification, object: nil, queue: .main) { note in
            let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            guard app?.bundleIdentifier == "com.apple.ScreenSaver.Engine" else { return }
            MainActor.assumeIsolated { Screensaver.shared.systemSaverEnded() }
        }
    }

    /// The scenes ticked in Settings, or all of them if none are.
    var enabled: [SceneKind] {
        let off = Set(Prefs.saverOff)
        let on = SceneKind.all.filter { !off.contains($0.id) }
        return on.isEmpty ? SceneKind.all : on
    }

    func toggle() {
        if isShowing { return stop() }
        // The key press that dismissed the macOS screen saver also arrives here; don't start again.
        guard Date().timeIntervalSince(endedAt) > 1.5 else { return }
        start()
    }

    /// Starts on every display, opening with `scene` on the main one if given. `auto` means it started itself after idle.
    func start(scene: String? = nil, auto: Bool = false) {
        if isShowing {
            if let scene, let kind = SceneKind.named(scene) { windows.first?.show(kind) }
            return
        }
        if Prefs.saverSystem, scene == nil { return startSystem(auto: auto) }
        guard let gpu = GPU.shared else { return announce("This Mac can't run the screensaver: Metal isn't available") }

        Location.shared.holdPrompt = true
        let screens = NSScreen.screens
        var kinds = pick(screens.count)
        if let scene, let kind = SceneKind.named(scene) { kinds[0] = kind }
        windows = zip(screens, kinds).compactMap { screen, kind in
            guard let window = try? SaverWindow(screen: screen, gpu: gpu) else { return nil }
            window.onKey = { [weak self, weak window] event in if let window { self?.key(event, in: window) } }
            window.onPointer = { [weak self] in self?.hint() }
            window.stage.onScene = { [weak self, weak window] kind in if let window { self?.began(kind, in: window) } }
            window.show(kind)
            return window
        }
        guard !windows.isEmpty else { Location.shared.holdPrompt = false; return }
        isShowing = true
        engine?.setScreensaver(true, pinned: !auto)
        updateFrameRate()

        for window in windows { window.orderFrontRegardless() }
        previous = NSWorkspace.shared.frontmostApplication
        NSApp.activate()
        windows[0].makeKeyAndOrderFront(nil)
        NSCursor.hide()
        if Prefs.saverSound { soundOn() }
        let minutes = Prefs.saverShuffle
        if minutes > 0 { shuffleTimer = .every(Double(minutes * 60)) { [weak self] in self?.shuffle() } }
        clockTimer = .every(10) { [weak self] in self?.windows.forEach { $0.tickClock() } }
        windows.forEach { $0.tickClock() }
        announce(auto ? "Screensaver started. Escape returns." : "Screensaver. Escape to exit. Arrow keys change the scene, Space describes it.")
    }

    /// Ends the screensaver. `byUser` is false when sleep, a lock or a display change ended it.
    func stop(byUser: Bool = true) {
        guard isShowing else { return }
        isShowing = false
        endedAt = Date()
        shuffleTimer?.invalidate()
        clockTimer?.invalidate()
        shuffleTimer = nil
        clockTimer = nil
        synth?.stop()
        synth = nil
        if !usingSystem {
            NSCursor.unhide()
            for window in windows {
                window.sceneView.isPaused = true
                window.orderOut(nil)
            }
            windows = []
            if let previous, previous != .current { previous.activate() }
            previous = nil
        }
        usingSystem = false
        engine?.setScreensaver(false)
        Location.shared.holdPrompt = false
        if byUser {
            announce("Screensaver ended")
            // A sky scene wanted your location; now macOS can ask without fighting the screensaver for the keyboard.
            if Prefs.saverLock { Power.perform(.lock) } else { Location.shared.askIfWaiting() }
        }
    }

    /// Starts itself when the user has been idle long enough while Caffeinator keeps the Mac awake.
    /// Called on the engine's 30-second tick.
    func idleCheck() {
        let minutes = Prefs.saverIdle
        guard minutes > 0, !isShowing, engine?.isAwake == true, idleSeconds() >= Double(minutes * 60),
              !Self.screenLocked(), !Self.someoneElseHoldsDisplay() else { return }
        start(auto: true)
    }

    // MARK: Scenes

    /// Different scenes for `count` displays, avoiding the ones shown recently.
    private func pick(_ count: Int, avoiding showing: [String] = []) -> [SceneKind] {
        let pool = enabled
        var fresh = pool.filter { !recent.contains($0.id) && !showing.contains($0.id) }.shuffled()
        var picked: [SceneKind] = []
        for _ in 0..<count {
            if fresh.isEmpty { fresh = pool.filter { !picked.map(\.id).contains($0.id) && !showing.contains($0.id) }.shuffled() }
            if fresh.isEmpty { fresh = pool.shuffled() }
            picked.append(fresh.removeFirst())
        }
        return picked
    }

    private func shuffle() {
        let showing = windows.compactMap { $0.stage.kind?.id }
        let next = pick(windows.count, avoiding: showing)
        // Stagger multiple displays so they don't all fade at once.
        for (i, (window, kind)) in zip(windows, next).enumerated() {
            if i == 0 { window.show(kind); continue }
            Task { [weak window] in
                try? await Task.sleep(for: .seconds(20 * i))
                window?.show(kind)
            }
        }
    }

    private func step(_ window: SaverWindow, by delta: Int) {
        let pool = enabled
        let current = pool.firstIndex { $0.id == window.stage.kind?.id } ?? 0
        window.show(pool[(current + delta + pool.count) % pool.count])
        // A manual change restarts the shuffle clock.
        let minutes = Prefs.saverShuffle
        shuffleTimer?.invalidate()
        if minutes > 0 { shuffleTimer = .every(Double(minutes * 60)) { [weak self] in self?.shuffle() } }
    }

    private func began(_ kind: SceneKind, in window: SaverWindow) {
        recent = Array(([kind.id] + recent.filter { $0 != kind.id }).prefix(max(enabled.count / 2, 1)))
        window.caption(kind.name, kind.clock)
        let main = window === windows.first
        if window.stage.scene == nil {
            if main { announce("\(kind.name) couldn't start") }
            return step(window, by: 1)
        }
        if main {
            synth?.reset()
            announce("\(kind.name). \(kind.about)")
        }
        // Facts usually arrive a moment later, once live data has loaded.
        Task { [weak window] in
            try? await Task.sleep(for: .seconds(4))
            guard let window, window.stage.kind?.id == kind.id, let facts = window.stage.scene?.facts else { return }
            window.caption(kind.name + "  ·  " + facts, kind.clock)
        }
    }

    // MARK: Input

    private func key(_ event: NSEvent, in window: SaverWindow) {
        switch Int(event.keyCode) {
        case kVK_Escape: stop()
        case kVK_RightArrow: step(window, by: 1)
        case kVK_LeftArrow: step(window, by: -1)
        case kVK_Space:
            guard let kind = window.stage.kind else { return }
            let facts = window.stage.scene?.facts
            announce(kind.name + ". " + (facts ?? kind.about))
            window.caption(kind.name + (facts.map { "  ·  " + $0 } ?? ""), kind.clock)
        case kVK_ANSI_M:
            Prefs.d.set(!Prefs.saverSound, forKey: Key.saverSound)
            if Prefs.saverSound { soundOn() } else { soundOff() }
            announce(Prefs.saverSound ? "Sound on" : "Sound off")
        default: hint()
        }
    }

    /// A gentle reminder of how to leave, at most every ten seconds.
    private func hint() {
        guard Date().timeIntervalSince(hintedAt) > 10 else { return }
        hintedAt = Date()
        windows.forEach { $0.hint("Press Escape to exit") }
        announce("Press Escape to exit")
    }

    // MARK: Sound

    private func soundOn() {
        guard synth == nil else { return }
        let s = Synth()
        s.volume = Prefs.saverVolume
        s.start()
        synth = s
        windows.first?.stage.sound = s
    }

    private func soundOff() {
        windows.first?.stage.sound = nil
        synth?.stop()
        synth = nil
    }

    /// Settings changed while showing.
    func prefsChanged() {
        guard isShowing, !usingSystem else { return }
        synth?.volume = Prefs.saverVolume
        if Prefs.saverSound { soundOn() } else { soundOff() }
        windows.forEach { $0.tickClock() }
    }

    // MARK: System events

    private func displaysChanged() {
        guard isShowing, !usingSystem, windows.count != NSScreen.screens.count || zip(windows, NSScreen.screens).contains(where: { $0.frame != $1.frame }) else { return }
        let first = windows.first?.stage.kind?.id
        stop(byUser: false)
        start(scene: first)
    }

    /// 60 frames a second on power, 30 on battery or in Low Power Mode.
    private func updateFrameRate() {
        let battery = IOPSCopyPowerSourcesInfo().map { info in
            (IOPSGetProvidingPowerSourceType(info.takeRetainedValue())?.takeUnretainedValue() as String?) == kIOPSBatteryPowerValue
        } ?? false
        let fps = battery || ProcessInfo.processInfo.isLowPowerModeEnabled ? 30 : 60
        windows.forEach { $0.sceneView.preferredFramesPerSecond = fps }
    }

    private func startSystem(auto: Bool) {
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.ScreenSaver.Engine") else { return }
        isShowing = true
        usingSystem = true
        engine?.setScreensaver(true, pinned: !auto)
        NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
    }

    private func systemSaverEnded() {
        guard usingSystem else { return }
        stop(byUser: false)
    }

    private static func screenLocked() -> Bool {
        (CGSessionCopyCurrentDictionary() as? [String: Any])?["CGSSessionScreenIsLocked"] as? Bool ?? false
    }

    /// True while another app keeps the display on, such as a video player. macOS holds back its own screen saver then too.
    private static func someoneElseHoldsDisplay() -> Bool {
        var byProcess: Unmanaged<CFDictionary>?
        guard IOPMCopyAssertionsByProcess(&byProcess) == kIOReturnSuccess,
              let all = byProcess?.takeRetainedValue() as? [NSNumber: [[String: Any]]] else { return false }
        let me = ProcessInfo.processInfo.processIdentifier
        return all.contains { pid, assertions in
            pid.int32Value != me && assertions.contains {
                ($0[kIOPMAssertionTypeKey] as? String).map { [kIOPMAssertPreventUserIdleDisplaySleep, "NoDisplaySleepAssertion"].contains($0) } ?? false
            }
        }
    }
}

/// A borderless window over one display: the scene, a fading caption, the clock and a hint line.
/// It's a non-activating panel, so it takes the keyboard even when macOS won't let Caffeinator become the active app.
@MainActor final class SaverWindow: NSPanel {
    let sceneView: SceneView
    var stage: Stage { sceneView.stage }
    var onKey: (NSEvent) -> Void = { _ in }
    var onPointer: () -> Void = {}

    private let captionLabel = SaverWindow.label(size: 15, weight: .medium)
    private let clockLabel = SaverWindow.label(size: 64, weight: .ultraLight)
    private let hintLabel = SaverWindow.label(size: 13, weight: .regular)
    private var captionHide: Task<Void, Never>?
    private var clockShown = true
    private var lastPointer = NSEvent.mouseLocation

    init(screen: NSScreen, gpu: GPU) throws {
        sceneView = try SceneView(frame: NSRect(origin: .zero, size: screen.frame.size), gpu: gpu, pointScale: screen.backingScaleFactor)
        super.init(contentRect: screen.frame, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        setFrame(screen.frame, display: false)
        becomesKeyOnlyIfNeeded = false
        hidesOnDeactivate = false
        level = .screenSaver
        collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenAuxiliary]
        backgroundColor = .black
        isOpaque = true
        hasShadow = false
        isReleasedWhenClosed = false
        acceptsMouseMovedEvents = true
        animationBehavior = .none

        let content = NSView(frame: NSRect(origin: .zero, size: screen.frame.size))
        sceneView.autoresizingMask = [.width, .height]
        content.addSubview(sceneView)
        for label in [captionLabel, clockLabel, hintLabel] {
            label.alphaValue = 0
            content.addSubview(label)
        }
        clockLabel.textColor = NSColor(white: 1, alpha: 0.55)
        hintLabel.textColor = NSColor(white: 1, alpha: 0.7)
        contentView = content
        setAccessibilityLabel("Screensaver")
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    override func keyDown(with event: NSEvent) { onKey(event) }
    // Swallow ⌘Q, ⌘W and the like so they can't quit or close anything behind the screensaver.
    override func performKeyEquivalent(with event: NSEvent) -> Bool { onKey(event); return true }
    override func mouseDown(with event: NSEvent) { onPointer() }

    override func mouseMoved(with event: NSEvent) {
        // Ignore tiny jitters; a deliberate move gets the hint.
        let p = NSEvent.mouseLocation
        if hypot(p.x - lastPointer.x, p.y - lastPointer.y) > 40 { onPointer() }
        lastPointer = p
    }

    func show(_ kind: SceneKind) {
        stage.show(kind)
        sceneView.setAccessibilityLabel("Screensaver: \(kind.name)")
        sceneView.setAccessibilityHelp(kind.about)
    }

    /// Shows the caption for ten seconds, if captions are on.
    func caption(_ text: String, _ clockAllowed: Bool) {
        clockShown = clockAllowed
        tickClock()
        guard Prefs.saverCaptions else { return }
        captionLabel.stringValue = text
        place(captionLabel, x: 40, y: 36)
        captionLabel.setFrameSize(NSSize(width: min(captionLabel.frame.width, frame.width - 80), height: captionLabel.frame.height))
        fade(captionLabel, to: 1)
        captionHide?.cancel()
        captionHide = Task { [weak self] in
            try? await Task.sleep(for: .seconds(10))
            guard !Task.isCancelled, let self else { return }
            self.fade(self.captionLabel, to: 0)
        }
    }

    func hint(_ text: String) {
        hintLabel.stringValue = text
        let size = frame.size
        hintLabel.sizeToFit()
        hintLabel.setFrameOrigin(NSPoint(x: (size.width - hintLabel.frame.width) / 2, y: size.height / 2 - 160))
        fade(hintLabel, to: 1)
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(3))
            guard let self else { return }
            self.fade(self.hintLabel, to: 0)
        }
    }

    /// Updates the clock and lets it wander a little each minute so it never burns in.
    func tickClock() {
        guard Prefs.saverClock, clockShown else { return fade(clockLabel, to: 0) }
        let text = Date().formatted(date: .omitted, time: .shortened)
        guard text != clockLabel.stringValue || clockLabel.alphaValue == 0 else { return }
        clockLabel.stringValue = text
        let size = frame.size
        let minute = Double(Calendar.current.component(.minute, from: Date()))
        clockLabel.sizeToFit()
        place(clockLabel, x: size.width - clockLabel.frame.width - 60 - 40 * (1 + sin(minute * 0.7)),
              y: size.height - clockLabel.frame.height - 60 - 40 * (1 + cos(minute * 0.5)))
        fade(clockLabel, to: 1)
    }

    private func place(_ label: NSTextField, x: CGFloat, y: CGFloat) {
        label.sizeToFit()
        label.setFrameOrigin(NSPoint(x: x, y: y))
    }

    private func fade(_ view: NSView, to alpha: CGFloat) {
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 1.2
            view.animator().alphaValue = alpha
        }
    }

    private static func label(size: CGFloat, weight: NSFont.Weight) -> NSTextField {
        let label = NSTextField(labelWithString: "")
        label.lineBreakMode = .byTruncatingTail
        label.font = .systemFont(ofSize: size, weight: weight)
        label.textColor = NSColor(white: 1, alpha: 0.85)
        let shadow = NSShadow()
        shadow.shadowColor = NSColor(white: 0, alpha: 0.8)
        shadow.shadowBlurRadius = 6
        label.shadow = shadow
        return label
    }
}
