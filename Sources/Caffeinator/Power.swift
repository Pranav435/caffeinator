import AppKit
import AudioToolbox
import IOKit.pwr_mgt

enum PowerAction: String, CaseIterable, Identifiable {
    case sleep, displayOff, lock, logOut, restart, shutDown

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sleep: "Sleep"
        case .displayOff: "Turn Off Display"
        case .lock: "Lock Screen"
        case .logOut: "Log Out"
        case .restart: "Restart"
        case .shutDown: "Shut Down"
        }
    }

    var symbol: String {
        switch self {
        case .sleep: "moon"
        case .displayOff: "display"
        case .lock: "lock"
        case .logOut: "rectangle.portrait.and.arrow.right"
        case .restart: "arrow.clockwise"
        case .shutDown: "power"
        }
    }

    /// Actions that interrupt work get the warning countdown and the audio fade.
    var disruptive: Bool { ![.displayOff, .lock].contains(self) }

    init?(urlName: String) {
        switch urlName.lowercased().replacingOccurrences(of: "-", with: "") {
        case "sleep": self = .sleep
        case "displayoff", "display": self = .displayOff
        case "lock": self = .lock
        case "logout": self = .logOut
        case "restart": self = .restart
        case "shutdown": self = .shutDown
        default: return nil
        }
    }
}

enum Power {
    @MainActor static func perform(_ action: PowerAction) {
        switch action {
        case .sleep:
            let port = IOPMFindPowerManagement(mach_port_t(0))
            IOPMSleepSystem(port)
            IOServiceClose(port)
        case .displayOff: run("/usr/bin/pmset", "displaysleepnow")
        case .lock: lock()
        // The same Apple events the Apple menu sends to loginwindow, minus its confirmation dialog.
        // Apps with unsaved work can still cancel them.
        case .logOut: send("rlgo", action)
        case .restart: send("rest", action)
        case .shutDown: send("shut", action)
        }
    }

    @MainActor private static func lock() {
        typealias Lock = @convention(c) () -> Int32
        if let handle = dlopen("/System/Library/PrivateFrameworks/login.framework/Versions/Current/login", RTLD_LAZY),
           let symbol = dlsym(handle, "SACLockScreenImmediate") {
            _ = unsafeBitCast(symbol, to: Lock.self)()
        } else {
            run("/usr/bin/pmset", "displaysleepnow")
        }
    }

    private static func send(_ code: String, _ action: PowerAction) {
        let id = code.utf8.reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        Task.detached {
            let target = NSAppleEventDescriptor(bundleIdentifier: "com.apple.loginwindow")
            let event = NSAppleEventDescriptor(eventClass: AEEventClass(kCoreEventClass), eventID: id, targetDescriptor: target,
                                               returnID: AEReturnID(kAutoGenerateReturnID), transactionID: AETransactionID(kAnyTransactionID))
            do {
                _ = try event.sendEvent(options: [.noReply], timeout: 60)
            } catch {
                await MainActor.run {
                    let alert = NSAlert()
                    alert.messageText = "Couldn't \(action.title.lowercased())"
                    alert.informativeText = "Allow Caffeinator to control loginwindow in System Settings › Privacy & Security › Automation."
                    NSApp.activate()
                    alert.runModal()
                }
            }
        }
    }
}

/// Main output volume through CoreAudio, for fading out before sleep or shutdown.
enum Volume {
    private static func device() -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                                 mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var id = AudioObjectID(0)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        return AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id) == noErr ? id : nil
    }

    private static func address() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
                                   mScope: kAudioDevicePropertyScopeOutput, mElement: kAudioObjectPropertyElementMain)
    }

    static var level: Float? {
        guard let id = device() else { return nil }
        var address = address()
        var value = Float32(0)
        var size = UInt32(MemoryLayout<Float32>.size)
        return AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value) == noErr ? value : nil
    }

    static func set(_ level: Float) {
        guard let id = device() else { return }
        var address = address()
        var value = Float32(min(max(level, 0), 1))
        AudioObjectSetPropertyData(id, &address, 0, nil, UInt32(MemoryLayout<Float32>.size), &value)
    }

    /// Puts back a volume saved before a fade. Runs on cancel, on wake, and at the next launch after a shutdown.
    static func restore() {
        guard let saved = Prefs.d.object(forKey: Key.restoreVolume) as? Double else { return }
        Prefs.d.removeObject(forKey: Key.restoreVolume)
        set(Float(saved))
    }
}

@MainActor @Observable final class Countdown {
    let action: PowerAction
    let total: Int
    var left: Int

    init(action: PowerAction, seconds: Int) {
        self.action = action
        total = seconds
        left = seconds
    }
}

/// One-off scheduled actions, the nightly routine, and the warning countdown that precedes disruptive actions.
@MainActor final class Scheduler {
    struct Pending {
        let action: PowerAction
        let date: Date
        let minWarning: Int
    }

    private(set) var pending: Pending?
    var onChange: () -> Void = {}

    private var timer: Timer?
    private var countdown: Countdown?
    private var countdownTimer: Timer?
    private var fadeFrom: Float?
    private var assertion: IOPMAssertionID = 0
    private var nightlyDate: Date?
    private var nightlyTimer: Timer?
    private var idleWait: Timer?

    func schedule(_ action: PowerAction, at date: Date, minWarning: Int = 0) {
        pending = Pending(action: action, date: date, minWarning: minWarning)
        arm()
        announce("\(action.title) at \(Fmt.clock(date))")
        onChange()
    }

    /// Cancels the running countdown and the scheduled action, whichever exist.
    func cancel() {
        abort()
        guard let p = pending else { return }
        pending = nil
        arm()
        announce("Canceled \(p.action.title.lowercased())")
        onChange()
    }

    /// Timers count awake time only, so re-arm them by the wall clock after sleep. Anything missed is skipped, never run late,
    /// and a countdown interrupted by sleep is canceled rather than resumed in front of someone who just opened the lid.
    func rearm() {
        abort()
        if let p = pending, p.date.timeIntervalSinceNow < -120 {
            pending = nil
            announce("Skipped \(p.action.title.lowercased()) because the Mac was asleep")
            onChange()
        }
        arm()
        nightlyDate = nil
        idleWait?.invalidate()
        idleWait = nil
        configureNightly()
    }

    private func arm() {
        timer?.invalidate()
        timer = pending.map { p in .at(p.date) { [weak self] in self?.fire() } }
        holdAwake()
    }

    private func fire() {
        guard let p = pending else { return }
        pending = nil
        arm()
        onChange()
        if p.date.timeIntervalSinceNow > -120 { run(p.action, minWarning: p.minWarning) }
    }

    /// Keeps the Mac from idling to sleep before a scheduled action or during a countdown. The display may still sleep.
    private func holdAwake() {
        let needed = pending != nil || countdown != nil
        if needed, assertion == 0 {
            IOPMAssertionCreateWithName(kIOPMAssertPreventUserIdleSystemSleep as CFString, IOPMAssertionLevel(kIOPMAssertionLevelOn),
                                        "Caffeinator scheduled action" as CFString, &assertion)
        } else if !needed, assertion != 0 {
            IOPMAssertionRelease(assertion)
            assertion = 0
        }
    }

    // MARK: Countdown

    /// Runs an action now, after the warning countdown and audio fade if they apply.
    /// `minWarning` sets a floor so URL-triggered shutdowns always give a chance to cancel.
    func run(_ action: PowerAction, minWarning: Int = 0) {
        guard action.disruptive else { return Power.perform(action) }
        let warning = max(Prefs.warning, minWarning)
        guard warning > 0 || Prefs.fadeAudio else { return Power.perform(action) }

        abort() // a countdown already running gives way, and gets its volume back first
        let model = Countdown(action: action, seconds: warning > 0 ? warning : 10)
        countdown = model
        fadeFrom = Prefs.fadeAudio ? Volume.level : nil
        if let fadeFrom { Prefs.d.set(Double(fadeFrom), forKey: Key.restoreVolume) }
        if warning > 0 {
            Windows.show("warning", title: "Caffeinator", floating: true, closable: false) {
                WarningView(model: model, cancel: { [weak self] in self?.abort() }, now: { [weak self] in self?.complete() })
            }
            announce("\(action.title) in \(Fmt.spoken(Double(warning))). Press Return to cancel.")
        }
        countdownTimer = .every(1) { [weak self] in self?.step() }
        holdAwake()
    }

    private func step() {
        guard let c = countdown else { return }
        c.left -= 1
        if let fadeFrom { Volume.set(fadeFrom * Float(max(c.left, 0)) / Float(c.total)) }
        if c.left == 10, c.total > 15 { announce("10 seconds") }
        if c.left <= 0 { complete() }
    }

    private func complete() {
        guard let action = countdown?.action else { return }
        stopCountdown()
        Power.perform(action)
        // If an app blocks the shutdown, or the Mac wakes, the volume comes back.
        if fadeFrom != nil {
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(30))
                Volume.restore()
            }
        }
    }

    private func abort() {
        guard let action = countdown?.action else { return }
        stopCountdown()
        Volume.restore()
        announce("Canceled \(action.title.lowercased())")
    }

    private func stopCountdown() {
        countdownTimer?.invalidate()
        countdownTimer = nil
        countdown = nil
        Windows.close("warning")
        holdAwake()
    }

    // MARK: Nightly routine

    func configureNightly() {
        guard Prefs.nightly else {
            nightlyTimer?.invalidate()
            idleWait?.invalidate()
            nightlyTimer = nil
            idleWait = nil
            nightlyDate = nil
            return
        }
        let m = Prefs.nightlyMinutes
        guard let next = Calendar.current.nextDate(after: .now, matching: DateComponents(hour: m / 60, minute: m % 60, second: 0),
                                                   matchingPolicy: .nextTime),
              next != nightlyDate else { return }
        nightlyDate = next
        nightlyTimer?.invalidate()
        nightlyTimer = .at(next) { [weak self] in self?.nightly() }
    }

    private func nightly() {
        defer {
            nightlyDate = nil
            configureNightly()
        }
        guard let date = nightlyDate, date.timeIntervalSinceNow > -300 else { return }
        let action = Prefs.nightlyAction
        guard Prefs.nightlyWaitIdle, idleSeconds() < 600 else { return run(action) }
        // Still at the keyboard: check each minute for 10 idle minutes, and give up after 4 hours.
        idleWait?.invalidate()
        idleWait = .every(60) { [weak self] in
            guard let self else { return }
            let away = idleSeconds() >= 600
            if away { run(action) }
            if away || date.timeIntervalSinceNow < -4 * 3600 {
                idleWait?.invalidate()
                idleWait = nil
            }
        }
    }
}
