import AppKit
import IOKit.pwr_mgt

enum Until: Equatable {
    case process(pid_t, String)
    case activity
}

/// Decides whether the Mac should stay awake and holds the matching IOKit power assertion.
/// Awake = a manual session, or any trigger the user hasn't switched off, unless the away guard has stepped in.
@MainActor final class Engine {
    let triggers = Triggers()
    let scheduler = Scheduler()
    var onChange: () -> Void = {}

    private(set) var manual = false
    private(set) var end: Date?
    private(set) var until: Until?
    private(set) var then: PowerAction?
    private var thenMinWarning = 0

    /// Triggers the user turned off by hand. Each one comes back once its condition clears.
    private var paused: Set<String> = []
    private var awaySuspended = false
    private var awayFired = false
    private var wasAwake = false

    private var assertion: IOPMAssertionID = 0
    private var assertionType: String?
    private var endTimer: Timer?
    private var headsUpTimer: Timer?
    private var tickTimer: Timer?
    private var processWatch: DispatchSourceProcess?
    private var lastSample: Activity?
    private var quietTicks = 0

    var reasons: [String] { triggers.active.filter { !paused.contains($0) } }
    var isAwake: Bool { !awaySuspended && (manual || !reasons.isEmpty) }

    func launch() {
        triggers.onChange = { [weak self] in self?.triggersChanged() }
        triggers.onBatteryLow = { [weak self] level in self?.batteryLow(level) }
        scheduler.onChange = { [weak self] in self?.onChange() }
        let workspace = NSWorkspace.shared.notificationCenter
        workspace.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.didWake() }
        }
        workspace.addObserver(forName: NSWorkspace.willPowerOffNotification, object: nil, queue: .main) { _ in
            Prefs.d.set(true, forKey: Key.resumeNext)
        }
        Volume.restore()
        resume()
        reconfigure()
    }

    /// Called whenever a setting changes.
    func reconfigure() {
        triggers.reconfigure()
        scheduler.configureNightly()
        applyAssertion()
        updateTick()
    }

    // MARK: Sessions

    func toggle() { isAwake ? stop() : start(minutes: Prefs.defaultMinutes) }

    func start(minutes: Int) { start(end: minutes > 0 ? .now + Double(minutes * 60) : nil) }

    func start(end: Date? = nil, until: Until? = nil, quiet: Bool = false) {
        manual = true
        self.end = end
        self.until = until
        awaySuspended = false
        awayFired = false
        arm()
        let detail = switch until {
        case .process(_, let name): " until \(name) quits"
        case .activity: " until activity stops"
        case nil: end.map { " for \(Fmt.spoken($0.timeIntervalSinceNow))" } ?? ""
        }
        changed(quiet ? nil : "Caffeinator on" + detail, sound: !quiet)
    }

    func stop() {
        clearSession()
        paused = Set(triggers.active)
        awaySuspended = false
        changed("Caffeinator off", sound: true)
    }

    func extend(minutes: Int) {
        guard manual, let current = end else { return }
        end = max(current, .now) + Double(minutes * 60)
        arm()
        changed("\(Fmt.spoken(end!.timeIntervalSinceNow)) left")
    }

    func setThen(_ action: PowerAction?, minWarning: Int = 0) {
        then = action
        thenMinWarning = minWarning
        onChange()
    }

    func quit() {
        Prefs.d.removeObject(forKey: Key.resumeNext)
        NSApp.terminate(nil)
    }

    /// A timed or "until" session reached its end. Runs the follow-up action unless the Mac slept through the end time.
    private func finish() {
        guard manual else { return }
        let late = end.map { $0.timeIntervalSinceNow < -120 } ?? false
        let action = then
        let minWarning = thenMinWarning
        clearSession()
        if late, let action {
            changed("Caffeinator off. Skipped \(action.title.lowercased()) because the Mac was asleep.", sound: true)
        } else {
            changed("Caffeinator off", sound: true)
            if let action { scheduler.run(action, minWarning: minWarning) }
        }
    }

    private func clearSession() {
        manual = false
        end = nil
        until = nil
        then = nil
        thenMinWarning = 0
        disarm()
    }

    private func arm() {
        disarm()
        if let end {
            endTimer = .at(end) { [weak self] in self?.finish() }
            if end.timeIntervalSinceNow > 600 {
                headsUpTimer = .at(end - 300) { announce("Caffeinator: 5 minutes left") }
            }
        }
        switch until {
        case .process(let pid, _):
            let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: .main)
            source.setEventHandler { [weak self] in MainActor.assumeIsolated { self?.finish() } }
            source.resume()
            processWatch = source
            if kill(pid, 0) != 0, errno == ESRCH { Task { self.finish() } }
        case .activity:
            lastSample = Activity.sample()
        case nil:
            break
        }
    }

    private func disarm() {
        endTimer?.invalidate()
        headsUpTimer?.invalidate()
        processWatch?.cancel()
        endTimer = nil
        headsUpTimer = nil
        processWatch = nil
        lastSample = nil
        quietTicks = 0
    }

    // MARK: State changes

    private func changed(_ message: String? = nil, sound: Bool = false) {
        applyAssertion()
        updateTick()
        if isAwake != wasAwake {
            wasAwake = isAwake
            runShortcut(isAwake ? Prefs.onShortcut : Prefs.offShortcut)
        }
        if sound, Prefs.sound { NSSound(named: isAwake ? "Tink" : "Pop")?.play() }
        if let message { announce(message) }
        persist()
        onChange()
    }

    private func applyAssertion() {
        let type = isAwake ? (Prefs.keepDisplayOn ? kIOPMAssertPreventUserIdleDisplaySleep : kIOPMAssertPreventUserIdleSystemSleep) : nil
        guard type != assertionType else { return }
        if assertion != 0 { IOPMAssertionRelease(assertion) }
        assertion = 0
        assertionType = type
        if let type {
            IOPMAssertionCreateWithName(type as CFString, IOPMAssertionLevel(kIOPMAssertionLevelOn), "Caffeinator" as CFString, &assertion)
        }
    }

    private func triggersChanged() {
        paused.formIntersection(triggers.active)
        changed()
    }

    private func batteryLow(_ level: Int) {
        guard isAwake else { return }
        clearSession()
        paused = Set(triggers.active)
        changed("Battery at \(level) percent. Caffeinator off.", sound: true)
    }

    private func didWake() {
        awayFired = false
        awaySuspended = false
        if manual, let end {
            if end.timeIntervalSinceNow <= 0 { finish() } else { arm() }
        }
        scheduler.rearm()
        Volume.restore()
        changed()
    }

    // MARK: Tick

    /// A 30-second tick that runs only while something needs it: the away guard, the countdown,
    /// "until activity stops", and remote-session polling.
    private func updateTick() {
        let needed = isAwake || awaySuspended || Prefs.whileRemote
        if needed, tickTimer == nil {
            tickTimer = .every(30) { [weak self] in self?.tick() }
        } else if !needed {
            tickTimer?.invalidate()
            tickTimer = nil
        }
    }

    private func tick() {
        triggers.poll()
        checkAway()
        if manual, until == .activity { checkActivity() }
        if end != nil { onChange() }
    }

    private func checkAway() {
        let idle = idleSeconds()
        if idle < 60 {
            awayFired = false
            if awaySuspended {
                awaySuspended = false
                changed()
            }
            return
        }
        let minutes = Prefs.awayMinutes
        guard minutes > 0, isAwake, !awayFired, idle >= Double(minutes * 60) else { return }
        awayFired = true
        switch Prefs.awayAction {
        case .lock: Power.perform(.lock)
        case .display: Power.perform(.displayOff)
        case .sleep:
            clearSession()
            awaySuspended = true
            changed()
        }
    }

    private func checkActivity() {
        let sample = Activity.sample()
        defer { lastSample = sample }
        guard let last = lastSample else { return }
        quietTicks = sample.isQuiet(since: last) ? quietTicks + 1 : 0
        if quietTicks >= 4 { finish() } // two quiet minutes
    }

    // MARK: Status

    func status(spoken: Bool = false) -> String {
        var parts: [String]
        if awaySuspended {
            parts = ["Off", "you're away"]
        } else if !isAwake {
            parts = ["Off"]
        } else {
            parts = ["On"]
            if manual {
                if let end { parts.append((spoken ? Fmt.spoken : Fmt.short)(end.timeIntervalSinceNow) + " left") }
                switch until {
                case .process(_, let name): parts.append("until \(name) quits")
                case .activity: parts.append("until activity stops")
                case nil: break
                }
                if let then { parts.append("then \(then.title.lowercased())") }
            } else {
                parts += reasons
            }
        }
        if let p = scheduler.pending { parts.append("\(p.action.title) at \(Fmt.clock(p.date))") }
        return parts.joined(separator: spoken ? ", " : " · ")
    }

    var countdown: String? {
        guard manual, let end else { return nil }
        return Fmt.compact(end.timeIntervalSinceNow)
    }

    // MARK: Persistence and hooks

    /// Saves plain timed or open-ended sessions so "Resume after restart" can bring them back.
    private func persist() {
        let value: Double? = manual && until == nil ? (end?.timeIntervalSince1970 ?? 0) : nil
        guard Prefs.d.object(forKey: Key.savedSession) as? Double != value else { return }
        if let value { Prefs.d.set(value, forKey: Key.savedSession) } else { Prefs.d.removeObject(forKey: Key.savedSession) }
    }

    private func resume() {
        defer { Prefs.d.removeObject(forKey: Key.resumeNext) }
        guard Prefs.resume, Prefs.d.bool(forKey: Key.resumeNext), let saved = Prefs.d.object(forKey: Key.savedSession) as? Double else { return }
        if saved == 0 {
            start(quiet: true)
        } else if saved > Date.now.timeIntervalSince1970 + 60 {
            start(end: Date(timeIntervalSince1970: saved), quiet: true)
        }
    }

    private func runShortcut(_ name: String) {
        guard !name.isEmpty else { return }
        run("/usr/bin/shortcuts", "run", name)
    }
}
