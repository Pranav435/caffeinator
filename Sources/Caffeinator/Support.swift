import AppKit

/// UserDefaults keys. Settings views bind to these with @AppStorage; everything else reads through `Prefs`.
enum Key {
    static let shortcut = "shortcut"
    static let defaultMinutes = "defaultMinutes"
    static let keepDisplayOn = "keepDisplayOn"
    static let menuCountdown = "menuCountdown"
    static let sound = "sound"
    static let announce = "announce"
    static let resume = "resume"
    static let whileCall = "whileCall"
    static let whilePlugged = "whilePlugged"
    static let whileDisplay = "whileDisplay"
    static let whileRemote = "whileRemote"
    static let whileApps = "whileApps"
    static let batteryGuard = "batteryGuard"
    static let awayMinutes = "awayMinutes"
    static let awayAction = "awayAction"
    static let warning = "warning"
    static let fadeAudio = "fadeAudio"
    static let nightly = "nightly"
    static let nightlyAction = "nightlyAction"
    static let nightlyMinutes = "nightlyMinutes"
    static let nightlyWaitIdle = "nightlyWaitIdle"
    static let onShortcut = "onShortcut"
    static let offShortcut = "offShortcut"
    // Internal state, not shown in Settings.
    static let savedSession = "savedSession"
    static let resumeNext = "resumeNext"
    static let restoreVolume = "restoreVolume"
}

enum Prefs {
    static var d: UserDefaults { .standard }

    static func register() {
        d.register(defaults: [
            Key.shortcut: Shortcut.default.raw,
            Key.defaultMinutes: 0,
            Key.keepDisplayOn: true,
            Key.menuCountdown: false,
            Key.sound: true,
            Key.announce: true,
            Key.resume: true,
            Key.batteryGuard: 10,
            Key.awayMinutes: 0,
            Key.awayAction: AwayAction.sleep.rawValue,
            Key.warning: 60,
            Key.nightlyAction: PowerAction.sleep.rawValue,
            Key.nightlyMinutes: 60,
            Key.nightlyWaitIdle: true,
        ])
    }

    static var shortcut: Shortcut? { Shortcut(raw: d.string(forKey: Key.shortcut) ?? "") }
    static var defaultMinutes: Int { d.integer(forKey: Key.defaultMinutes) }
    static var keepDisplayOn: Bool { d.bool(forKey: Key.keepDisplayOn) }
    static var menuCountdown: Bool { d.bool(forKey: Key.menuCountdown) }
    static var sound: Bool { d.bool(forKey: Key.sound) }
    static var announce: Bool { d.bool(forKey: Key.announce) }
    static var resume: Bool { d.bool(forKey: Key.resume) }
    static var whileCall: Bool { d.bool(forKey: Key.whileCall) }
    static var whilePlugged: Bool { d.bool(forKey: Key.whilePlugged) }
    static var whileDisplay: Bool { d.bool(forKey: Key.whileDisplay) }
    static var whileRemote: Bool { d.bool(forKey: Key.whileRemote) }
    static var apps: [String] { d.stringArray(forKey: Key.whileApps) ?? [] }
    static var batteryGuard: Int { d.integer(forKey: Key.batteryGuard) }
    static var awayMinutes: Int { d.integer(forKey: Key.awayMinutes) }
    static var awayAction: AwayAction { AwayAction(rawValue: d.string(forKey: Key.awayAction) ?? "") ?? .sleep }
    static var warning: Int { d.integer(forKey: Key.warning) }
    static var fadeAudio: Bool { d.bool(forKey: Key.fadeAudio) }
    static var nightly: Bool { d.bool(forKey: Key.nightly) }
    static var nightlyAction: PowerAction { PowerAction(rawValue: d.string(forKey: Key.nightlyAction) ?? "") ?? .sleep }
    static var nightlyMinutes: Int { d.integer(forKey: Key.nightlyMinutes) }
    static var nightlyWaitIdle: Bool { d.bool(forKey: Key.nightlyWaitIdle) }
    static var onShortcut: String { d.string(forKey: Key.onShortcut) ?? "" }
    static var offShortcut: String { d.string(forKey: Key.offShortcut) ?? "" }
}

enum AwayAction: String, CaseIterable, Identifiable {
    case sleep, lock, display
    var id: String { rawValue }
    var title: String {
        switch self {
        case .sleep: "Let Mac sleep"
        case .lock: "Lock screen"
        case .display: "Turn off display"
        }
    }
}

/// Session lengths offered in the menu and Settings, in minutes. 0 means no time limit.
let presets = [0, 15, 30, 60, 120, 240, 480]

enum Fmt {
    static func preset(_ minutes: Int) -> String {
        switch minutes {
        case 0: "Indefinitely"
        case ..<60: "\(minutes) Minutes"
        case 60: "1 Hour"
        default: "\(minutes / 60) Hours"
        }
    }

    /// "1 hr, 5 min"
    static func short(_ t: TimeInterval) -> String { format(t, .short) }
    /// "1 hour, 5 minutes", for VoiceOver.
    static func spoken(_ t: TimeInterval) -> String { format(t, .full) }
    /// "1h 5m", for the menu bar.
    static func compact(_ t: TimeInterval) -> String { format(t, .abbreviated) }

    private static func format(_ t: TimeInterval, _ style: DateComponentsFormatter.UnitsStyle) -> String {
        let f = DateComponentsFormatter()
        f.unitsStyle = style
        if t < 59.5 {
            f.allowedUnits = [.second]
            return f.string(from: max(t.rounded(), 0)) ?? ""
        }
        f.allowedUnits = [.hour, .minute]
        return f.string(from: (t / 60).rounded(.up) * 60) ?? ""
    }

    /// "11:30 PM", or "Sat 11:30 PM" when more than a day away.
    static func clock(_ date: Date) -> String {
        date.timeIntervalSinceNow < 86_400
            ? date.formatted(date: .omitted, time: .shortened)
            : date.formatted(.dateTime.weekday(.abbreviated).hour().minute())
    }
}

enum TimeParse {
    /// Turns "90", "45m", "1h30", "1.5 hours" (durations) or "5pm", "5:30 am", "17:30" (clock times) into an end date.
    static func end(_ text: String, from now: Date = .now, calendar: Calendar = .current) -> Date? {
        var s = text.lowercased().trimmingCharacters(in: .whitespaces)
        for prefix in ["for ", "in ", "until ", "till ", "at "] where s.hasPrefix(prefix) {
            s = String(s.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        }
        if let date = clock(s, now, calendar) { return date }
        if let seconds = duration(s), seconds > 0 { return now.addingTimeInterval(seconds) }
        return nil
    }

    private static func clock(_ s: String, _ now: Date, _ calendar: Calendar) -> Date? {
        guard let m = s.wholeMatch(of: /(\d{1,2})(?::(\d{2}))?\s*(am|pm|a|p)?/),
              m.output.2 != nil || m.output.3 != nil,
              let hour = Int(m.output.1), hour < 24 else { return nil }
        let minute = m.output.2.flatMap { Int($0) } ?? 0
        guard minute < 60 else { return nil }
        var hours: [Int]
        if let suffix = m.output.3 {
            guard (1...12).contains(hour) else { return nil }
            hours = [hour % 12 + (suffix.hasPrefix("p") ? 12 : 0)]
        } else {
            // "2:30" could mean 2:30 or 14:30; pick whichever comes first.
            hours = (1...12).contains(hour) ? [hour % 12, hour % 12 + 12] : [hour]
        }
        return hours.compactMap {
            calendar.nextDate(after: now, matching: DateComponents(hour: $0, minute: minute, second: 0), matchingPolicy: .nextTime)
        }.min()
    }

    private static func duration(_ s: String) -> TimeInterval? {
        let token = /(\d+(?:\.\d+)?)\s*([a-z]*)/
        guard !s.isEmpty, s.replacing(token, with: "").allSatisfy({ $0 == " " || $0 == "," }) else { return nil }
        var total: TimeInterval = 0
        for m in s.matches(of: token) {
            guard let value = Double(m.output.1) else { return nil }
            switch m.output.2 {
            case "", "m", "min", "mins", "minute", "minutes": total += value * 60
            case "h", "hr", "hrs", "hour", "hours": total += value * 3600
            case "s", "sec", "secs", "second", "seconds": total += value
            case "d", "day", "days": total += value * 86_400
            default: return nil
            }
        }
        return total
    }
}

/// Seconds since the last keyboard, mouse or trackpad input.
func idleSeconds() -> TimeInterval {
    CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: ~0)!)
}

/// Speaks through VoiceOver (or any screen reader) without moving focus.
@MainActor func announce(_ text: String) {
    guard Prefs.announce else { return }
    NSAccessibility.post(element: NSApp as Any, notification: .announcementRequested, userInfo: [
        .announcement: text,
        .priority: NSAccessibilityPriorityLevel.high.rawValue,
    ])
}

func run(_ path: String, _ arguments: String...) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = arguments
    try? p.run()
}

extension Timer {
    /// Main-run-loop timers in common mode, so they also fire while a menu is open.
    @MainActor static func at(_ date: Date, _ block: @escaping @MainActor () -> Void) -> Timer {
        add(Timer(fire: date, interval: 0, repeats: false) { _ in MainActor.assumeIsolated(block) }, tolerance: 1)
    }

    @MainActor static func every(_ interval: TimeInterval, _ block: @escaping @MainActor () -> Void) -> Timer {
        add(Timer(timeInterval: interval, repeats: true) { _ in MainActor.assumeIsolated(block) }, tolerance: interval / 10)
    }

    @MainActor private static func add(_ timer: Timer, tolerance: TimeInterval) -> Timer {
        timer.tolerance = tolerance
        RunLoop.main.add(timer, forMode: .common)
        return timer
    }
}
