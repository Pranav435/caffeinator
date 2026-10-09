import AppKit
import CoreAudio
import CoreMediaIO
import IOKit.ps

/// Conditions that keep the Mac awake without a manual session. Everything is notification-driven
/// except remote sessions, which `poll()` checks on the engine's 30-second tick.
@MainActor final class Triggers {
    var onChange: () -> Void = {}
    var onBatteryLow: (Int) -> Void = { _ in }
    /// Human-readable reasons, such as "Xcode running".
    private(set) var active: [String] = []

    private var mediaWatched = false
    private var powerWatched = false
    private var audioDevices: Set<AudioObjectID> = []
    private var cameras: Set<CMIOObjectID> = []
    private var remote = false
    private var wasLow = false

    init() {
        let workspace = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.didLaunchApplicationNotification, NSWorkspace.didTerminateApplicationNotification] {
            workspace.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.refresh() }
            }
        }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
    }

    /// Starts listeners the first time a setting needs them. They stay installed; they cost nothing while quiet.
    func reconfigure() {
        if Prefs.whileCall, !mediaWatched { watchMedia() }
        if Prefs.whilePlugged || Prefs.batteryGuard > 0, !powerWatched { watchPower() }
        if !Prefs.whileRemote { remote = false }
        refresh()
    }

    func poll() {
        guard Prefs.whileRemote else { return }
        let now = Self.remoteSession()
        if now != remote {
            remote = now
            refresh()
        }
    }

    func refresh() {
        var reasons: [String] = []
        if Prefs.whileCall, audioDevices.contains(where: Self.micRunning) || cameras.contains(where: Self.cameraRunning) {
            reasons.append("Camera or mic in use")
        }
        let ids = Set(Prefs.apps)
        if !ids.isEmpty {
            let names = NSWorkspace.shared.runningApplications.filter { ids.contains($0.bundleIdentifier ?? "") }.compactMap(\.localizedName)
            reasons += Set(names).sorted().map { "\($0) running" }
        }
        let power = Self.power()
        if Prefs.whilePlugged, power.onAC { reasons.append("Plugged in") }
        if Prefs.whileDisplay, NSScreen.screens.contains(where: Self.isExternal) { reasons.append("External display") }
        if Prefs.whileRemote, remote { reasons.append("Remote session") }

        let changed = reasons != active
        active = reasons
        let low = !power.onAC && power.level >= 0 && power.level < Prefs.batteryGuard
        if low, !wasLow { onBatteryLow(power.level) }
        wasLow = low
        if changed { onChange() }
    }

    // MARK: Camera and microphone

    private func watchMedia() {
        mediaWatched = true
        var audio = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
                                               mElement: kAudioObjectPropertyElementMain)
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &audio, .main) { [weak self] _, _ in
            MainActor.assumeIsolated { self?.watchDevices() }
        }
        var video = Self.cmioAddress(kCMIOHardwarePropertyDevices)
        CMIOObjectAddPropertyListenerBlock(CMIOObjectID(kCMIOObjectSystemObject), &video, .main) { [weak self] _, _ in
            MainActor.assumeIsolated { self?.watchDevices() }
        }
        watchDevices()
    }

    /// Listens to "running somewhere" on every input device and camera, including ones plugged in later.
    private func watchDevices() {
        for id in Self.audioInputs() where audioDevices.insert(id).inserted {
            var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
                                                     mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            AudioObjectAddPropertyListenerBlock(id, &address, .main) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.refresh() }
            }
        }
        for id in Self.videoDevices() where cameras.insert(id).inserted {
            var address = Self.cmioAddress(kCMIODevicePropertyDeviceIsRunningSomewhere)
            CMIOObjectAddPropertyListenerBlock(id, &address, .main) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.refresh() }
            }
        }
        refresh()
    }

    private static func audioInputs() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        let system = AudioObjectID(kAudioObjectSystemObject)
        guard AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr else { return [] }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(system, &address, 0, nil, &size, &ids) == noErr else { return [] }
        return ids.filter { id in
            var streams = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: kAudioDevicePropertyScopeInput,
                                                     mElement: kAudioObjectPropertyElementMain)
            var count: UInt32 = 0
            return AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &count) == noErr && count > 0
        }
    }

    private static func micRunning(_ id: AudioObjectID) -> Bool {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
                                                 mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value) == noErr && value != 0
    }

    private static func cmioAddress(_ selector: Int) -> CMIOObjectPropertyAddress {
        CMIOObjectPropertyAddress(mSelector: CMIOObjectPropertySelector(selector), mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                                  mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
    }

    private static func videoDevices() -> [CMIOObjectID] {
        var address = cmioAddress(kCMIOHardwarePropertyDevices)
        var size: UInt32 = 0
        let system = CMIOObjectID(kCMIOObjectSystemObject)
        guard CMIOObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr else { return [] }
        var ids = [CMIOObjectID](repeating: 0, count: Int(size) / MemoryLayout<CMIOObjectID>.size)
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(system, &address, 0, nil, size, &used, &ids) == noErr else { return [] }
        return ids
    }

    private static func cameraRunning(_ id: CMIOObjectID) -> Bool {
        var address = cmioAddress(kCMIODevicePropertyDeviceIsRunningSomewhere)
        var value: UInt32 = 0
        var used: UInt32 = 0
        return CMIOObjectGetPropertyData(id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &used, &value) == noErr && value != 0
    }

    // MARK: Power and displays

    private func watchPower() {
        powerWatched = true
        let context = Unmanaged.passUnretained(self).toOpaque()
        guard let source = IOPSNotificationCreateRunLoopSource({ context in
            MainActor.assumeIsolated { Unmanaged<Triggers>.fromOpaque(context!).takeUnretainedValue().refresh() }
        }, context)?.takeRetainedValue() else { return }
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .defaultMode)
    }

    private static func power() -> (onAC: Bool, level: Int) {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else { return (true, -1) }
        let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String?
        var level = -1
        let sources = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] ?? []
        for source in sources {
            guard let d = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue() as? [String: Any],
                  let current = d[kIOPSCurrentCapacityKey] as? Int, let max = d[kIOPSMaxCapacityKey] as? Int, max > 0 else { continue }
            level = current * 100 / max
        }
        return (type != kIOPSBatteryPowerValue, level)
    }

    private static func isExternal(_ screen: NSScreen) -> Bool {
        guard let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else { return false }
        return CGDisplayIsBuiltin(id) == 0
    }

    // MARK: Remote sessions

    /// True while someone is logged in over SSH (a utmpx entry with a remote host) or connected through Screen Sharing.
    nonisolated static func remoteSession() -> Bool {
        setutxent()
        defer { endutxent() }
        while let entry = getutxent() {
            guard Int32(entry.pointee.ut_type) == USER_PROCESS else { continue }
            let host = withUnsafeBytes(of: entry.pointee.ut_host) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
            if !host.isEmpty, !host.hasPrefix(":"), !host.hasPrefix("tmux") { return true }
        }
        return processRunning("screensharingd")
    }

    nonisolated private static func processRunning(_ name: String) -> Bool {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0]
        var size = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &size, nil, 0) == 0 else { return false }
        let stride = MemoryLayout<kinfo_proc>.stride
        var procs = [kinfo_proc](repeating: kinfo_proc(), count: size / stride + 16)
        size = procs.count * stride
        guard sysctl(&mib, UInt32(mib.count), &procs, &size, nil, 0) == 0 else { return false }
        return procs.prefix(size / stride).contains { p in
            withUnsafeBytes(of: p.kp_proc.p_comm) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) } == name
        }
    }
}

/// A CPU and network snapshot. Two snapshots tell whether the Mac has gone quiet,
/// which ends "Until Activity Stops" sessions.
struct Activity {
    var busy: UInt32
    var total: UInt32
    var bytes: UInt64
    var time: Date

    static func sample() -> Activity {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics(hostPort, HOST_CPU_LOAD_INFO, $0, &count) }
        }
        let t = info.cpu_ticks // user, system, idle, nice
        let busy = t.0 &+ t.1 &+ t.3
        return Activity(busy: busy, total: busy &+ t.2, bytes: networkBytes(), time: .now)
    }

    /// Quiet means under half a core busy on average and under 50 KB/s of network traffic.
    func isQuiet(since earlier: Activity) -> Bool {
        let seconds = time.timeIntervalSince(earlier.time)
        let ticks = total &- earlier.total
        guard seconds > 0, ticks > 0 else { return false }
        let cores = Double(busy &- earlier.busy) / Double(ticks) * Double(ProcessInfo.processInfo.activeProcessorCount)
        let rate = Double(bytes > earlier.bytes ? bytes - earlier.bytes : 0) / seconds
        return cores < 0.5 && rate < 50_000
    }

    /// Total bytes in and out on every non-loopback interface, using the 64-bit counters.
    private static func networkBytes() -> UInt64 {
        var mib: [Int32] = [CTL_NET, PF_ROUTE, 0, 0, NET_RT_IFLIST2, 0]
        var length = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &length, nil, 0) == 0 else { return 0 }
        var buffer = [UInt8](repeating: 0, count: length)
        guard sysctl(&mib, UInt32(mib.count), &buffer, &length, nil, 0) == 0 else { return 0 }
        var total: UInt64 = 0
        buffer.withUnsafeBytes { raw in
            var offset = 0
            while offset + MemoryLayout<if_msghdr>.size <= length {
                let header = raw.loadUnaligned(fromByteOffset: offset, as: if_msghdr.self)
                guard header.ifm_msglen > 0 else { break }
                if Int32(header.ifm_type) == RTM_IFINFO2, offset + MemoryLayout<if_msghdr2>.size <= length {
                    let data = raw.loadUnaligned(fromByteOffset: offset, as: if_msghdr2.self).ifm_data
                    if Int32(data.ifi_type) != IFT_LOOP { total &+= data.ifi_ibytes &+ data.ifi_obytes }
                }
                offset += Int(header.ifm_msglen)
            }
        }
        return total
    }
}

private let hostPort = mach_host_self()
