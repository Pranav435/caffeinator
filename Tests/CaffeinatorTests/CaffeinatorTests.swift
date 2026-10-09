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
