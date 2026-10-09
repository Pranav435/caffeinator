import AppKit
import Carbon.HIToolbox
import ServiceManagement
import SwiftUI

@MainActor enum Windows {
    private static var open: [String: NSWindow] = [:]

    static func show<V: View>(_ id: String, title: String, floating: Bool = false, closable: Bool = true, reuse: Bool = false,
                              _ content: () -> V) {
        if reuse, let window = open[id] { return present(window) }
        open[id]?.close()
        let window = NSWindow(contentViewController: NSHostingController(rootView: content()))
        window.title = title
        window.styleMask = closable ? [.titled, .closable] : [.titled]
        window.isReleasedWhenClosed = false
        if floating { window.level = .floating }
        window.center()
        open[id] = window
        present(window)
    }

    static func close(_ id: String) {
        open[id]?.close()
        open[id] = nil
    }

    static func showSettings() { show("settings", title: "Caffeinator Settings", reuse: true) { SettingsView() } }

    static func showTimeEntry(_ mode: TimeEntryView.Mode, _ engine: Engine) {
        show("time", title: mode == .keepAwake ? "Keep Awake" : "Schedule") { TimeEntryView(mode: mode, engine: engine) }
    }

    private static func present(_ window: NSWindow) {
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }
}

// MARK: Settings

struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralTab().tabItem { Label("General", systemImage: "gearshape") }
            TriggersTab().tabItem { Label("Triggers", systemImage: "bolt") }
            PowerTab().tabItem { Label("Power", systemImage: "power") }
            AutomationTab().tabItem { Label("Automation", systemImage: "terminal") }
        }
        .formStyle(.grouped)
        .frame(width: 480, height: 470)
    }
}

private struct GeneralTab: View {
    @AppStorage(Key.defaultMinutes) private var minutes = 0
    @AppStorage(Key.keepDisplayOn) private var display = true
    @AppStorage(Key.menuCountdown) private var countdown = false
    @AppStorage(Key.sound) private var sound = true
    @AppStorage(Key.announce) private var speak = true
    @AppStorage(Key.resume) private var resume = true
    @State private var login = SMAppService.mainApp.status == .enabled

    var body: some View {
        Form {
            LabeledContent("Shortcut") { ShortcutRecorder() }
            Picker("Turn on for", selection: $minutes) {
                ForEach(presets, id: \.self) { Text(Fmt.preset($0)).tag($0) }
            }
            Toggle("Keep display on", isOn: $display)
            Toggle("Show time left in menu bar", isOn: $countdown)
            Toggle("Sounds", isOn: $sound)
            Toggle("VoiceOver announcements", isOn: $speak)
            Toggle("Resume after restart", isOn: $resume)
            Toggle("Launch at login", isOn: $login)
                .onChange(of: login) { _, on in
                    try? on ? SMAppService.mainApp.register() : SMAppService.mainApp.unregister()
                    login = SMAppService.mainApp.status == .enabled
                }
        }
    }
}

private struct TriggersTab: View {
    @AppStorage(Key.whileCall) private var call = false
    @AppStorage(Key.whilePlugged) private var plugged = false
    @AppStorage(Key.whileDisplay) private var display = false
    @AppStorage(Key.whileRemote) private var remote = false
    @AppStorage(Key.batteryGuard) private var battery = 10
    @AppStorage(Key.awayMinutes) private var away = 0
    @AppStorage(Key.awayAction) private var awayAction = AwayAction.sleep
    @State private var apps = Prefs.apps

    var body: some View {
        Form {
            Section("Stay awake while") {
                Toggle("Camera or mic is in use", isOn: $call)
                Toggle("Plugged in", isOn: $plugged)
                Toggle("External display is connected", isOn: $display)
                Toggle("Someone is connected by SSH or Screen Sharing", isOn: $remote)
            }
            Section("Stay awake while these apps run") {
                ForEach(apps, id: \.self) { id in AppRow(bundleID: id) { apps.removeAll { $0 == id } } }
                Button("Add App…", action: addApps)
            }
            Section("Turn off") {
                Picker("When battery drops below", selection: $battery) {
                    Text("Never").tag(0)
                    ForEach([10, 20, 30, 50], id: \.self) { Text("\($0)%").tag($0) }
                }
                Picker("When I'm away for", selection: $away) {
                    Text("Never").tag(0)
                    ForEach([5, 10, 15, 30, 60], id: \.self) { Text("\($0) minutes").tag($0) }
                }
                if away > 0 {
                    Picker("Then", selection: $awayAction) {
                        ForEach(AwayAction.allCases) { Text($0.title).tag($0) }
                    }
                }
            }
        }
        .onChange(of: apps) { _, ids in Prefs.d.set(ids, forKey: Key.whileApps) }
    }

    private func addApps() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.application]
        panel.allowsMultipleSelection = true
        panel.directoryURL = URL(fileURLWithPath: "/Applications")
        guard panel.runModal() == .OK else { return }
        for id in panel.urls.compactMap({ Bundle(url: $0)?.bundleIdentifier }) where !apps.contains(id) { apps.append(id) }
    }
}

private struct AppRow: View {
    let bundleID: String
    let remove: () -> Void

    var body: some View {
        let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
        let name = url?.deletingPathExtension().lastPathComponent ?? bundleID
        HStack {
            if let url {
                Image(nsImage: NSWorkspace.shared.icon(forFile: url.path)).resizable().frame(width: 18, height: 18).accessibilityHidden(true)
            }
            Text(name)
            Spacer()
            Button(action: remove) { Image(systemName: "minus.circle") }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove \(name)")
        }
    }
}

private struct PowerTab: View {
    @AppStorage(Key.warning) private var warning = 60
    @AppStorage(Key.fadeAudio) private var fade = false
    @AppStorage(Key.nightly) private var nightly = false
    @AppStorage(Key.nightlyAction) private var action = PowerAction.sleep
    @AppStorage(Key.nightlyMinutes) private var minutes = 60
    @AppStorage(Key.nightlyWaitIdle) private var waitIdle = true

    var body: some View {
        Form {
            Section("Sleep, log out, restart and shut down") {
                Picker("Warning", selection: $warning) {
                    Text("None").tag(0)
                    Text("30 seconds").tag(30)
                    Text("1 minute").tag(60)
                    Text("5 minutes").tag(300)
                }
                Toggle("Fade out audio first", isOn: $fade)
            }
            Section("Nightly") {
                Toggle("Run every night", isOn: $nightly)
                if nightly {
                    Picker("Action", selection: $action) {
                        ForEach([PowerAction.sleep, .shutDown, .restart, .logOut, .lock]) { Text($0.title).tag($0) }
                    }
                    DatePicker("Time", selection: time, displayedComponents: .hourAndMinute)
                    Toggle("Wait until I've been idle for 10 minutes", isOn: $waitIdle)
                }
            }
        }
    }

    private var time: Binding<Date> {
        Binding {
            Calendar.current.date(bySettingHour: minutes / 60, minute: minutes % 60, second: 0, of: .now) ?? .now
        } set: {
            let c = Calendar.current.dateComponents([.hour, .minute], from: $0)
            minutes = (c.hour ?? 0) * 60 + (c.minute ?? 0)
        }
    }
}

private struct AutomationTab: View {
    @AppStorage(Key.onShortcut) private var onName = ""
    @AppStorage(Key.offShortcut) private var offName = ""
    @State private var names: [String] = []

    var body: some View {
        Form {
            Section("Run a shortcut from the Shortcuts app") {
                picker("When Caffeinator turns on", $onName)
                picker("When it turns off", $offName)
            }
            Section("URL commands") {
                ForEach(URLCommand.examples, id: \.self) {
                    Text($0).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                }
            }
        }
        .task {
            let found = await Task.detached { Self.shortcutNames() }.value
            names = Array(Set(found + [onName, offName].filter { !$0.isEmpty })).sorted()
        }
    }

    private func picker(_ title: String, _ selection: Binding<String>) -> some View {
        Picker(title, selection: selection) {
            Text("None").tag("")
            ForEach(names, id: \.self) { Text($0).tag($0) }
        }
    }

    nonisolated private static func shortcutNames() -> [String] {
        let p = Process()
        let pipe = Pipe()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/shortcuts")
        p.arguments = ["list"]
        p.standardOutput = pipe
        guard (try? p.run()) != nil else { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init)
    }
}

// MARK: Shortcut recorder

private struct ShortcutRecorder: View {
    @AppStorage(Key.shortcut) private var raw = Shortcut.default.raw
    @State private var recording = false
    @State private var monitor: Any?
    @State private var message: String?

    private var shortcut: Shortcut? { Shortcut(raw: raw) }

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack(spacing: 6) {
                Button(recording ? "Type shortcut…" : shortcut?.display ?? "Record") { recording ? stop() : start() }
                    .accessibilityLabel("Shortcut")
                    .accessibilityValue(recording ? "Recording" : shortcut?.spoken ?? "None")
                    .accessibilityHint(recording ? "Press the new combination. Escape cancels." : "Records a new combination")
                if shortcut != nil, !recording {
                    Button {
                        raw = ""
                        announce("Shortcut cleared")
                    } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Clear shortcut")
                }
            }
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
        }
        .onDisappear { if recording { stop() } }
    }

    private func start() {
        recording = true
        message = nil
        HotKey.shared.suspend()
        announce("Press the new shortcut")
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            record(event)
            return nil
        }
    }

    private func record(_ event: NSEvent) {
        let new = Shortcut(event: event)
        if new.keyCode == kVK_Escape, new.mods == 0 { return stop() }
        if let problem = new.problem {
            message = problem
            return announce(problem)
        }
        raw = new.raw
        HotKey.shared.set(new)
        stop()
        message = HotKey.shared.failed ? "Another app already uses \(new.display)." : nil
        announce(HotKey.shared.failed ? "Another app already uses \(new.spoken)" : "Shortcut set to \(new.spoken)")
    }

    private func stop() {
        recording = false
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        HotKey.shared.resume()
    }
}

// MARK: Custom time and scheduling

struct TimeEntryView: View {
    enum Mode { case keepAwake, schedule }

    let mode: Mode
    let engine: Engine
    @State private var text = ""
    @State private var action: PowerAction?
    @FocusState private var focused: Bool

    private var end: Date? { TimeParse.end(text) }

    private var preview: String {
        guard let end else { return text.isEmpty ? "Type a length or a clock time" : "Use a length like 90m or a time like 5:30pm" }
        return "Ends at \(Fmt.clock(end)), \(Fmt.short(end.timeIntervalSinceNow)) from now"
    }

    var body: some View {
        Form {
            if mode == .schedule {
                Picker("Action", selection: $action) {
                    ForEach(PowerAction.allCases) { Text($0.title).tag(PowerAction?.some($0)) }
                }
            }
            TextField(mode == .keepAwake ? "Keep awake" : "When", text: $text, prompt: Text("90m or 5:30pm"))
                .focused($focused)
                .onSubmit(submit)
            Text(preview).foregroundStyle(.secondary)
            if mode == .keepAwake {
                Picker("Then", selection: $action) {
                    Text("Do Nothing").tag(PowerAction?.none)
                    ForEach(PowerAction.allCases) { Text($0.title).tag(PowerAction?.some($0)) }
                }
            }
        }
        .formStyle(.grouped)
        .safeAreaInset(edge: .bottom) {
            HStack {
                Spacer()
                Button("Cancel") { Windows.close("time") }.keyboardShortcut(.cancelAction)
                Button(mode == .keepAwake ? "Start" : "Schedule", action: submit).keyboardShortcut(.defaultAction).disabled(end == nil)
            }
            .padding([.horizontal, .bottom])
        }
        .frame(width: 380, height: 206)
        .onAppear {
            focused = true
            if mode == .schedule { action = .shutDown }
        }
    }

    private func submit() {
        guard let end else { return announce("Use a length like 90m or a time like 5:30pm") }
        if mode == .keepAwake {
            engine.setThen(action)
            engine.start(end: end)
        } else if let action {
            engine.scheduler.schedule(action, at: end)
        }
        Windows.close("time")
    }
}

// MARK: Warning

struct WarningView: View {
    let model: Countdown
    let cancel: () -> Void
    let now: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: model.action.symbol).font(.system(size: 30)).accessibilityHidden(true)
            Text("\(model.action.title) in \(model.left / 60):\(String(format: "%02d", model.left % 60))")
                .font(.title2.monospacedDigit())
                .accessibilityLabel("\(model.action.title) in \(Fmt.spoken(Double(model.left)))")
            HStack {
                Button("\(model.action.title) Now", action: now)
                Button("Cancel", action: cancel).keyboardShortcut(.defaultAction)
            }
            // Escape cancels too. One button can't hold both shortcuts, so this invisible one takes Escape.
            .background { Button("", action: cancel).keyboardShortcut(.cancelAction).opacity(0).accessibilityHidden(true) }
        }
        .padding(24)
        .frame(width: 300)
    }
}
