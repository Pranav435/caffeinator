import AppKit

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var item: NSStatusItem!
    private let engine = Engine()

    // Set up in willFinishLaunching so a caffeinator:// URL that launched the app finds everything ready.
    func applicationWillFinishLaunching(_ notification: Notification) {
        Prefs.register()
        NSApp.mainMenu = Self.keyboardMenu()
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.imagePosition = .imageLeading
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu

        engine.onChange = { [weak self] in self?.refresh() }
        HotKey.shared.action = { [weak self] in self?.engine.toggle() }
        HotKey.shared.set(Prefs.shortcut)
        NotificationCenter.default.addObserver(forName: UserDefaults.didChangeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.prefsChanged() }
        }
        engine.launch()
        refresh()
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        urls.forEach { URLCommand.run($0, engine) }
    }

    private func prefsChanged() {
        if HotKey.shared.shortcut != Prefs.shortcut { HotKey.shared.set(Prefs.shortcut) }
        engine.reconfigure()
        refresh()
    }

    private func refresh() {
        guard let button = item.button else { return }
        let image = NSImage(systemSymbolName: engine.isAwake ? "cup.and.saucer.fill" : "cup.and.saucer", accessibilityDescription: nil)
        image?.isTemplate = true
        button.image = image
        button.title = Prefs.menuCountdown ? engine.countdown.map { " " + $0 } ?? "" : ""
        button.toolTip = "Caffeinator: " + engine.status()
        button.setAccessibilityLabel("Caffeinator, " + engine.status(spoken: true))
    }

    // MARK: Menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        let e = engine
        let status = NSMenuItem(title: e.status(), action: nil, keyEquivalent: "")
        status.isEnabled = false
        status.setAccessibilityLabel(e.status(spoken: true))
        var items: [NSMenuItem] = [status, .separator(), MenuItem(e.isAwake ? "Turn Off" : "Turn On") { e.toggle() }]
        items.append(submenu("Keep Awake", keepAwakeItems()))
        items.append(submenu("When It Ends", [MenuItem("Do Nothing", checked: e.then == nil) { e.setThen(nil) }]
                + PowerAction.allCases.map { a in MenuItem(a.title, checked: e.then == a) { e.setThen(a) } }))
        if e.end != nil { items.append(MenuItem("Add 15 Minutes") { e.extend(minutes: 15) }) }
        items.append(.separator())
        items.append(submenu("Power", powerItems()))
        if let p = e.scheduler.pending {
            items.append(MenuItem("Cancel \(p.action.title) at \(Fmt.clock(p.date))") { e.scheduler.cancel() })
        }
        items += [.separator(), MenuItem("Settings…", key: ",") { Windows.showSettings() }, MenuItem("Quit Caffeinator", key: "q") { e.quit() }]
        menu.items = items
    }

    private func keepAwakeItems() -> [NSMenuItem] {
        let e = engine
        let apps = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && $0 != .current }
            .sorted { ($0.localizedName ?? "") < ($1.localizedName ?? "") }
            .map { app in
                let name = app.localizedName ?? "App"
                let item = MenuItem(name) { e.start(until: .process(app.processIdentifier, name)) }
                item.image = (app.icon?.copy() as? NSImage).map { icon in
                    icon.size = NSSize(width: 16, height: 16)
                    return icon
                }
                return item
            }
        return presets.map { m in MenuItem(Fmt.preset(m)) { e.start(minutes: m) } } + [
            .separator(),
            submenu("Until App Quits", apps),
            MenuItem("Until Activity Stops") { e.start(until: .activity) },
            MenuItem("Custom…") { Windows.showTimeEntry(.keepAwake, e) },
        ]
    }

    private func powerItems() -> [NSMenuItem] {
        let e = engine
        let action = { (a: PowerAction) in MenuItem(a.title) { e.scheduler.run(a) } }
        return [action(.sleep), action(.displayOff), action(.lock), .separator(),
                action(.logOut), action(.restart), action(.shutDown), .separator(),
                MenuItem("Schedule…") { Windows.showTimeEntry(.schedule, e) }]
    }

    private func submenu(_ title: String, _ items: [NSMenuItem]) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        let menu = NSMenu(title: title)
        menu.items = items
        item.submenu = menu
        item.isEnabled = !items.isEmpty
        return item
    }

    /// Never visible for a menu bar app, but it gives Settings windows ⌘W, ⌘Q and the Edit shortcuts.
    private static func keyboardMenu() -> NSMenu {
        let main = NSMenu()
        for (title, items) in [
            ("Caffeinator", [("Close Window", #selector(NSWindow.performClose(_:)), "w"), ("Quit Caffeinator", #selector(NSApplication.terminate(_:)), "q")]),
            ("Edit", [("Undo", Selector(("undo:")), "z"), ("Cut", #selector(NSText.cut(_:)), "x"), ("Copy", #selector(NSText.copy(_:)), "c"),
                      ("Paste", #selector(NSText.paste(_:)), "v"), ("Select All", #selector(NSText.selectAll(_:)), "a")]),
        ] {
            let menu = NSMenu(title: title)
            for (name, action, key) in items { menu.addItem(NSMenuItem(title: name, action: action, keyEquivalent: key)) }
            let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
            item.submenu = menu
            main.addItem(item)
        }
        return main
    }
}

/// An NSMenuItem that runs a closure.
final class MenuItem: NSMenuItem {
    private let handler: @MainActor () -> Void

    init(_ title: String, key: String = "", checked: Bool = false, _ handler: @escaping @MainActor () -> Void) {
        self.handler = handler
        super.init(title: title, action: #selector(fire), keyEquivalent: key)
        target = self
        state = checked ? .on : .off
    }

    required init(coder: NSCoder) { fatalError("init(coder:) is not used") }

    @objc private func fire() {
        let handler = handler
        MainActor.assumeIsolated { handler() }
    }
}

/// caffeinator:// commands for Terminal, Shortcuts, Raycast and the like.
@MainActor enum URLCommand {
    static let examples = [
        "open \"caffeinator://on?for=90m\"",
        "open \"caffeinator://on?until=5pm&then=sleep\"",
        "open \"caffeinator://on?app=Xcode\"",
        "open \"caffeinator://on?pid=1234&then=shutdown\"",
        "open \"caffeinator://on?until=activity\"",
        "open \"caffeinator://shutdown?in=2h\"",
        "open \"caffeinator://off\"",
    ]

    static func run(_ url: URL, _ engine: Engine) {
        guard url.scheme == "caffeinator", let command = url.host()?.lowercased() else { return }
        var q: [String: String] = [:]
        for item in URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? [] { q[item.name.lowercased()] = item.value ?? "" }
        let time = (q["for"] ?? q["in"] ?? q["until"] ?? q["at"]).flatMap { TimeParse.end($0) }

        switch command {
        case "on":
            if let then = q["then"].flatMap(PowerAction.init(urlName:)) { engine.setThen(then, minWarning: 30) }
            if let pid = q["pid"].flatMap({ pid_t($0) }) {
                let name = NSRunningApplication(processIdentifier: pid)?.localizedName ?? "process \(pid)"
                engine.start(until: .process(pid, name))
            } else if let wanted = q["app"]?.lowercased() {
                guard let app = NSWorkspace.shared.runningApplications.first(where: {
                    $0.localizedName?.lowercased() == wanted || $0.bundleIdentifier?.lowercased() == wanted
                }) else { return announce("\(q["app"] ?? "That app") isn't running") }
                engine.start(until: .process(app.processIdentifier, app.localizedName ?? wanted))
            } else if ["activity", "idle"].contains(q["until"]?.lowercased()) {
                engine.start(until: .activity)
            } else {
                engine.start(end: time)
            }
        case "off": engine.stop()
        case "toggle": engine.toggle()
        case "cancel": engine.scheduler.cancel()
        case "settings": Windows.showSettings()
        default:
            // Power commands that arrive by URL always get at least a 30-second warning,
            // so a link clicked on a web page can't shut the Mac down unannounced.
            guard let action = PowerAction(urlName: command) else { return }
            if let time { engine.scheduler.schedule(action, at: time, minWarning: 30) } else { engine.scheduler.run(action, minWarning: 30) }
        }
    }
}
