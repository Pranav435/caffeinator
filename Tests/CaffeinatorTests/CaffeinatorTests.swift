import Carbon.HIToolbox
import Foundation
import Testing
@testable import Caffeinator

struct TimeParseTests {
    let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()
    /// Friday 9 October 2026, 14:00 UTC.
    let now = Date(timeIntervalSince1970: 1_791_554_400)

    func minutes(_ text: String) -> Double? {
        TimeParse.end(text, from: now, calendar: calendar).map { $0.timeIntervalSince(now) / 60 }
    }

    @Test(arguments: [
        ("90", 90.0), ("45m", 45), ("45 min", 45), ("1h30", 90), ("1h 30m", 90), ("1.5h", 90),
        ("2 hours", 120), ("for 20 minutes", 20), ("in 2h", 120), ("1d", 1440), ("1h, 15m", 75),
    ])
    func durations(text: String, expected: Double) {
        #expect(minutes(text) == expected)
    }

    @Test(arguments: [
        ("5pm", 180.0), ("5:30 pm", 210), ("17:30", 210), ("until 3pm", 60),
        ("2:30", 30),      // 14:30 comes before 02:30 tomorrow
        ("1:00", 660),     // 13:00 already passed, so 01:00 tomorrow
        ("9am", 1140),     // tomorrow morning
        ("12am", 600), ("12pm", 1320), ("at 0:15", 615),
    ])
    func clockTimes(text: String, expected: Double) {
        #expect(minutes(text) == expected)
    }

    @Test(arguments: ["", "abc", "25:00", "5x", "13pm", "10:75", "0", "m"])
    func rejects(text: String) {
        #expect(minutes(text) == nil)
    }
}

struct ShortcutTests {
    @Test func defaultIsOptionCommandZ() {
        #expect(Shortcut.default.keyCode == UInt32(kVK_ANSI_Z))
        #expect(Shortcut.default.problem == nil)
        #expect(Shortcut(raw: Shortcut.default.raw) == .default)
    }

    @Test func screensaverDefaultIsOptionCommandA() {
        #expect(Shortcut.screensaver.keyCode == UInt32(kVK_ANSI_A))
        #expect(Shortcut.screensaver.problem == nil)
        #expect(Shortcut.screensaver != .default)
    }

    @Test func rejectsVoiceOverModifiers() {
        let voiceOver = Shortcut(keyCode: UInt32(kVK_ANSI_K), mods: UInt32(controlKey | optionKey | cmdKey))
        #expect(voiceOver.problem != nil)
    }

    @Test func needsCommandOrControlUnlessFunctionKey() {
        #expect(Shortcut(keyCode: UInt32(kVK_ANSI_K), mods: UInt32(optionKey | shiftKey)).problem != nil)
        #expect(Shortcut(keyCode: UInt32(kVK_ANSI_K), mods: UInt32(controlKey | cmdKey)).problem == nil)
        #expect(Shortcut(keyCode: UInt32(kVK_F19), mods: 0).problem == nil)
    }

    @Test func rejectsGarbage() {
        #expect(Shortcut(raw: "") == nil)
        #expect(Shortcut(raw: "abc") == nil)
    }
}

struct PowerActionTests {
    @Test(arguments: [
        ("sleep", PowerAction.sleep), ("display-off", .displayOff), ("lock", .lock),
        ("logout", .logOut), ("log-out", .logOut), ("restart", .restart), ("shutdown", .shutDown), ("Shut-Down", .shutDown),
    ])
    func urlNames(name: String, expected: PowerAction) {
        #expect(PowerAction(urlName: name) == expected)
    }

    @Test func unknownURLName() {
        #expect(PowerAction(urlName: "explode") == nil)
    }

    @Test func onlyInterruptingActionsWarn() {
        #expect(PowerAction.allCases.filter(\.disruptive) == [.sleep, .logOut, .restart, .shutDown])
    }
}

struct AstroTests {
    func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }

    @Test func julianDayAtJ2000() {
        #expect(Astro.julianDay(date("2000-01-01T12:00:00Z")) == 2_451_545.0)
        #expect(abs(Astro.gmst(date("2000-01-01T12:00:00Z")) - 280.46) < 0.01)
    }

    @Test func sunAtSolsticeAndEquinox() {
        #expect(abs(Astro.sun(date("2026-06-21T08:24:00Z")).dec - 23.44) < 0.05)
        #expect(abs(Astro.sun(date("2026-03-20T14:46:00Z")).dec) < 0.05)
    }

    @Test func moonPhases() {
        // The total solar eclipse of 8 April 2024 was at new moon; 23 April was full.
        #expect(Astro.moonPhase(date("2024-04-08T18:21:00Z")).lit < 0.01)
        #expect(Astro.moonPhase(date("2024-04-23T23:49:00Z")).lit > 0.99)
        #expect(Astro.moonPhase(date("2024-04-15T19:13:00Z")).waxing)
    }

    @Test func subsolarPointAtNoonUTC() {
        let p = Astro.subpoint(Astro.sun(date("2026-03-20T12:00:00Z")), date("2026-03-20T12:00:00Z"))
        #expect(abs(p.lon) < 3) // within the equation of time
    }

    @Test func zoneTableCoordinates() {
        let la = Location.parse("+3403-11814")!
        #expect(abs(la.lat - 34.05) < 0.01 && abs(la.lon + 118.233) < 0.01)
        let ny = Location.parse("+404251-0740023")!
        #expect(abs(ny.lat - 40.714) < 0.001 && abs(ny.lon + 74.006) < 0.001)
        #expect(Location.timeZoneGuess("Asia/Kolkata").place == "Kolkata")
    }
}
