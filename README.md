# Caffeinator

Keep your Mac awake from the menu bar. Press ⌥⌘Z, or click the cup.

Caffeinator can also sleep, lock, log out, restart or shut down the Mac on a timer, and it can stay awake by itself while conditions you choose are true. It works fully with VoiceOver. It's a native Swift app with no dependencies, and the download is under 1 MB.

## Install

1. Download `Caffeinator.zip` from [Releases](https://github.com/Pranav435/caffeinator/releases/latest).
2. Unzip it and move Caffeinator to Applications.
3. Open it. macOS blocks the first launch because the app isn't notarized. Go to System Settings › Privacy & Security and click **Open Anyway**, or run:

   ```sh
   xattr -dr com.apple.quarantine /Applications/Caffeinator.app
   ```

Requires macOS 14 or later. Runs natively on Apple silicon and Intel.

## Use

Click the cup to open the menu. A filled cup means the Mac is staying awake.

| Menu item | What it does |
|---|---|
| Turn On / Turn Off | Same as ⌥⌘Z |
| Keep Awake | 15 minutes to 8 hours, indefinitely, until an app quits, until CPU and network go quiet, or a custom time |
| When It Ends | Sleep, turn off the display, lock, log out, restart or shut down when the session ends |
| Add 15 Minutes | Extends a timed session |
| Power | Runs any of those actions now, or schedules one |

Custom times take a length (`90m`, `2h`, `1h30`) or a clock time (`5pm`, `17:30`).

"Until activity stops" ends after two minutes with less than half a CPU core busy and under 50 KB/s of network traffic. Use it for renders, builds, exports and downloads.

## Settings

**General:** the shortcut, how long the shortcut and Turn On keep the Mac awake, whether the display stays on, time left in the menu bar, sounds, VoiceOver announcements, resume after restart, launch at login.

**Triggers:** stay awake while the camera or mic is in use, while plugged in, while an external display is connected, while someone is connected over SSH or Screen Sharing, or while apps you pick are running. Turn off when the battery drops below a set level. After you've been away for a set time, let the Mac sleep, lock the screen, or turn off the display.

**Power:** a warning of 30 seconds to 5 minutes before sleep, log out, restart and shut down, with an optional audio fade-out. A nightly routine runs one action at a set time and can wait until you've been idle for 10 minutes.

**Automation:** run a shortcut from the Shortcuts app when Caffeinator turns on or off.

Turning Caffeinator off while a trigger is active pauses that trigger until its condition clears.

## URL commands

Use these from Terminal, Shortcuts, Raycast or scripts. Quote the URL in a shell.

| Command | Effect |
|---|---|
| `caffeinator://on` | On, no time limit |
| `caffeinator://on?for=90m` | On for 90 minutes |
| `caffeinator://on?until=5pm&then=sleep` | On until 5 PM, then sleep |
| `caffeinator://on?app=Xcode` | On until Xcode quits |
| `caffeinator://on?pid=1234` | On until process 1234 exits |
| `caffeinator://on?until=activity` | On until CPU and network go quiet |
| `caffeinator://off`, `caffeinator://toggle` | Off, or toggle |
| `caffeinator://sleep` | Sleep now. Also `display-off`, `lock`, `logout`, `restart`, `shutdown` |
| `caffeinator://shutdown?in=30m` | Schedule it. `at=11pm` works too |
| `caffeinator://cancel` | Cancel the countdown and any scheduled action |
| `caffeinator://settings` | Open Settings |

`then=` takes the same action names. For example, to keep the Mac awake through a render and shut down when it exits:

```sh
./render.sh & open "caffeinator://on?pid=$!&then=shutdown"
```

Log out, restart and shut down requested by URL always show at least a 30-second warning, so a link on a web page can't power off the Mac without notice.

## VoiceOver

- Every control has a label. The menu bar item reads its state, such as "Caffeinator, On, 42 minutes left".
- Turning on or off, timer endings, battery and away guards, and warnings are announced without moving focus.
- The shortcut recorder refuses Control-Option combinations, since VoiceOver uses them.
- Cancel is the default button in warnings, so Return or Escape stops a shutdown.

## How it works

Caffeinator holds an IOKit power assertion (`PreventUserIdleDisplaySleep`, or `PreventUserIdleSystemSleep` when the display may sleep), the same mechanism `caffeinate` uses. Run `pmset -g assertions` to see it.

The shortcut uses Carbon's `RegisterEventHotKey`, which needs no Accessibility permission. Triggers listen for system notifications instead of polling. A 30-second timer runs only while the Mac is being kept awake or the remote-session trigger is on.

Log out, restart and shut down send loginwindow the same Apple events as the Apple menu, without its confirmation dialog, so apps with unsaved work can still cancel them. macOS asks for Automation permission the first time. Lock calls `SACLockScreenImmediate` from login.framework.

Scheduled actions keep the Mac from idling to sleep until they run. If the Mac sleeps anyway, for example with the lid closed, a missed action is skipped instead of running late.

## Build

```sh
swift test
scripts/build.sh            # build/Caffeinator.app, universal, ad-hoc signed
scripts/build.sh --install  # also copies it to /Applications and relaunches
```

Requires Xcode 16 or later. Pushing a `v*` tag runs the GitHub Actions workflow, which tests, builds and publishes a release with the zip.

## License

MIT
