import AppKit
import Carbon.HIToolbox

/// A key plus Carbon modifier flags, stored in defaults as "keyCode:modifiers".
struct Shortcut: Equatable {
    var keyCode: UInt32
    var mods: UInt32

    static let `default` = Shortcut(keyCode: UInt32(kVK_ANSI_Z), mods: UInt32(optionKey | cmdKey))
    static let screensaver = Shortcut(keyCode: UInt32(kVK_ANSI_A), mods: UInt32(optionKey | cmdKey))

    init(keyCode: UInt32, mods: UInt32) {
        self.keyCode = keyCode
        self.mods = mods
    }

    init?(raw: String) {
        let parts = raw.split(separator: ":").compactMap { UInt32($0) }
        guard parts.count == 2 else { return nil }
        self.init(keyCode: parts[0], mods: parts[1])
    }

    init(event: NSEvent) {
        let flags = event.modifierFlags
        var mods = 0
        if flags.contains(.control) { mods |= controlKey }
        if flags.contains(.option) { mods |= optionKey }
        if flags.contains(.shift) { mods |= shiftKey }
        if flags.contains(.command) { mods |= cmdKey }
        self.init(keyCode: UInt32(event.keyCode), mods: UInt32(mods))
    }

    var raw: String { "\(keyCode):\(mods)" }

    private static let modifiers = [(controlKey, "⌃", "Control"), (optionKey, "⌥", "Option"), (shiftKey, "⇧", "Shift"), (cmdKey, "⌘", "Command")]
    private var held: [(Int, String, String)] { Self.modifiers.filter { mods & UInt32($0.0) != 0 } }

    /// "⌥⌘Z"
    var display: String { held.map(\.1).joined() + key.symbol }
    /// "Option Command Z", for VoiceOver.
    var spoken: String { (held.map(\.2) + [key.name]).joined(separator: " ") }

    /// Why this combination can't be used, if it can't.
    var problem: String? {
        let voiceOver = UInt32(controlKey | optionKey)
        if mods & voiceOver == voiceOver { return "Control-Option is the VoiceOver key. Pick another combination." }
        if Self.functionKeys[Int(keyCode)] == nil, mods & UInt32(cmdKey | controlKey) == 0 { return "Include Command or Control." }
        return nil
    }

    private static let functionKeys: [Int: Int] = [
        kVK_F1: 1, kVK_F2: 2, kVK_F3: 3, kVK_F4: 4, kVK_F5: 5, kVK_F6: 6, kVK_F7: 7, kVK_F8: 8, kVK_F9: 9, kVK_F10: 10,
        kVK_F11: 11, kVK_F12: 12, kVK_F13: 13, kVK_F14: 14, kVK_F15: 15, kVK_F16: 16, kVK_F17: 17, kVK_F18: 18, kVK_F19: 19, kVK_F20: 20,
    ]
    private static let special: [Int: (symbol: String, name: String)] = [
        kVK_Return: ("↩", "Return"), kVK_Tab: ("⇥", "Tab"), kVK_Space: ("Space", "Space"), kVK_Delete: ("⌫", "Delete"),
        kVK_ForwardDelete: ("⌦", "Forward Delete"), kVK_Escape: ("⎋", "Escape"), kVK_Home: ("↖", "Home"), kVK_End: ("↘", "End"),
        kVK_PageUp: ("⇞", "Page Up"), kVK_PageDown: ("⇟", "Page Down"), kVK_LeftArrow: ("←", "Left Arrow"),
        kVK_RightArrow: ("→", "Right Arrow"), kVK_UpArrow: ("↑", "Up Arrow"), kVK_DownArrow: ("↓", "Down Arrow"),
    ]
    private static let punctuation = [
        ",": "Comma", ".": "Period", "/": "Slash", ";": "Semicolon", "'": "Quote", "[": "Left Bracket",
        "]": "Right Bracket", "\\": "Backslash", "-": "Minus", "=": "Equals", "`": "Grave Accent",
    ]

    private var key: (symbol: String, name: String) {
        if let s = Self.special[Int(keyCode)] { return s }
        if let n = Self.functionKeys[Int(keyCode)] { return ("F\(n)", "F\(n)") }
        let c = Self.character(for: keyCode)
        return (c, Self.punctuation[c] ?? c)
    }

    /// The character this key produces on the current keyboard layout, so AZERTY users see "W" where US users see "Z".
    private static func character(for keyCode: UInt32) -> String {
        guard let source = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
              let pointer = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData) else { return "?" }
        let layout = Unmanaged<CFData>.fromOpaque(pointer).takeUnretainedValue() as Data
        var dead: UInt32 = 0
        var length = 0
        var chars = [UniChar](repeating: 0, count: 4)
        let capacity = chars.count
        let status = layout.withUnsafeBytes { buffer in
            UCKeyTranslate(buffer.bindMemory(to: UCKeyboardLayout.self).baseAddress, UInt16(keyCode), UInt16(kUCKeyActionDisplay), 0,
                           UInt32(LMGetKbdType()), OptionBits(kUCKeyTranslateNoDeadKeysMask), &dead, capacity, &length, &chars)
        }
        return status == noErr && length > 0 ? String(utf16CodeUnits: chars, count: length).uppercased() : "?"
    }
}

/// System-wide hotkeys through Carbon's RegisterEventHotKey, which needs no Accessibility permission.
@MainActor final class HotKey {
    /// Turns Caffeinator on and off.
    static let toggle = HotKey(id: 1)
    /// Starts and stops the screensaver.
    static let screensaver = HotKey(id: 2)
    static let all = [toggle, screensaver]

    var action: () -> Void = {}
    private(set) var shortcut: Shortcut?
    /// True when another app already owns the combination.
    private(set) var failed = false
    private let id: UInt32
    private var ref: EventHotKeyRef?
    private static var suspended = false
    private static var installed = false

    private init(id: UInt32) {
        self.id = id
        guard !Self.installed else { return }
        Self.installed = true
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var pressed = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &pressed)
            MainActor.assumeIsolated { HotKey.all.first { $0.id == pressed.id }?.action() }
            return noErr
        }, 1, &spec, nil, nil)
    }

    func set(_ shortcut: Shortcut?) {
        self.shortcut = shortcut
        apply()
    }

    /// Releases every combination while a Settings recorder listens for a new one.
    static func suspend() { suspended = true; all.forEach { $0.apply() } }
    static func resume() { suspended = false; all.forEach { $0.apply() } }

    private func apply() {
        if let ref { UnregisterEventHotKey(ref) }
        ref = nil
        failed = false
        guard !Self.suspended, let shortcut else { return }
        let hotKeyID = EventHotKeyID(signature: OSType(0x4341_4646), id: id) // 'CAFF'
        failed = RegisterEventHotKey(shortcut.keyCode, shortcut.mods, hotKeyID, GetApplicationEventTarget(), 0, &ref) != noErr
    }
}
