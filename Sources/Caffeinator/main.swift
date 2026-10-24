import AppKit

switch CommandLine.arguments.dropFirst().first {
case "--snapshot": exit(Snapshot.run(Array(CommandLine.arguments.dropFirst(2))))
case "--preview": Snapshot.preview(Array(CommandLine.arguments.dropFirst(2)))
default: break
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
