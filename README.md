# Caffeinator

Your Mac is a gifted napper. Leave it alone for a few minutes and it dozes off mid-download, mid-render or mid-presentation. Caffeinator is the coffee: press ⌥⌘Z, or click the cup in the menu bar, and the Mac will stay up later than you do.

Caffeinator can also put the Mac to bed on your schedule (sleep, lock, log out, restart or shut down), and it can keep the Mac up by itself during calls or while the apps you pick are running. Everything works with VoiceOver. The app is native Swift with no dependencies, and the download is under 1 MB.

Free, no ads, no tracking. If it rescues a render, [buy its developer a coffee](#donate).

## Install

1. Download `Caffeinator.zip` from [Releases](https://github.com/Pranav435/caffeinator/releases/latest).
2. Unzip it and move Caffeinator to Applications.
3. Open it. macOS blocks the first launch because the app isn't notarized. Notarizing costs $99 a year; this app costs nothing. ([Donate](#donate) has a plan for that.) Go to System Settings › Privacy & Security and click **Open Anyway**, or run:

   ```sh
   xattr -dr com.apple.quarantine /Applications/Caffeinator.app
   ```

Requires macOS 14 or later. Runs natively on Apple silicon and Intel.

## Use

Click the cup to open the menu. A filled cup means the Mac is staying awake. An empty one means it's allowed to nap.

| Menu item | What it does |
|---|---|
| Turn On / Turn Off | Same as ⌥⌘Z |
| Keep Awake | 15 minutes to 8 hours, indefinitely, until an app quits, until CPU and network go quiet, or a custom time |
| When It Ends | Sleep, turn off the display, lock, log out, restart or shut down when the session ends |
| Add 15 Minutes | Extends a timed session |
| Power | Runs any of those actions now, or schedules one |

Custom times take a length (`90m`, `2h`, `1h30`) or a clock time (`5pm`, `17:30`).

"Until activity stops" ends after two minutes with less than half a CPU core busy and under 50 KB/s of network traffic. Use it for renders, builds, exports, downloads, and anything else you'd otherwise babysit.

## Settings

**General:** the shortcut, how long the shortcut and Turn On keep the Mac awake, whether the display stays on, time left in the menu bar, sounds, VoiceOver announcements, resume after restart, launch at login.

**Triggers:** stay awake while the camera or mic is in use, while plugged in, while an external display is connected, while someone is connected over SSH or Screen Sharing, or while apps you pick are running. It won't doze off during your video call, which is more than can be said for some of the attendees. Turn off when the battery drops below a set level, because a dead battery wins every argument. After you've been away for a set time, let the Mac sleep, lock the screen, or turn off the display. When you leave, it takes the hint.

**Power:** a warning of 30 seconds to 5 minutes before sleep, log out, restart and shut down, with an optional audio fade-out for anyone who falls asleep to podcasts. A nightly routine gives the Mac a bedtime: one action at a set time. It can wait until you've been idle for 10 minutes, so it won't pull the plug mid-sentence at 1 AM.

**Automation:** run a shortcut from the Shortcuts app when Caffeinator turns on or off. Set a Focus, start a playlist, dim the lights: whatever your ritual is.

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

Log out, restart and shut down requested by URL always show at least a 30-second warning. A web page can order a coffee, but it can't order a shutdown without notice.

## VoiceOver

- Every control has a label. The menu bar item reads its state, such as "Caffeinator, On, 42 minutes left".
- Turning on or off, timer endings, battery and away guards, and warnings are announced without moving focus.
- The shortcut recorder refuses Control-Option combinations, since VoiceOver got there first.
- Cancel is the default button in warnings, so Return or Escape stops a shutdown.

## Donate

Caffeinator has no ads, no account, no subscription, no "Pro" tier and no analytics, and it never touches the network. This section is the only part of the project that wants anything from you.

Your Mac stays up on Caffeinator. Caffeinator stays up on its developer. Its developer stays up on coffee. You can see where this is going.

| Order | What it does |
|---|---|
| [Espresso, $3](https://paypal.me/theblindiephoenix/3USD) | Covers one cup and one bug fix, in that order |
| [Flat white, $5](https://paypal.me/theblindiephoenix/5USD) | Costs the same as a café coffee, which lasts an afternoon. Caffeinator lasts as long as your Mac does |
| [Bag of beans, $20](https://paypal.me/theblindiephoenix/20USD) | Fuels a whole feature, from the first idea to the 2 AM bug it ships with |
| [A year of notarization, $99](https://paypal.me/theblindiephoenix/99USD) | Goes toward Apple's developer fee, so the app can be notarized and "Open Anyway" disappears for everyone |
| [Your own amount](https://paypal.me/theblindiephoenix) | All currencies are converted to coffee at a fair rate |

If Caffeinator has saved one download, render or presentation from a surprise nap, that's about one coffee's worth. Can't spare the money? A star on the repo costs nothing, and yes, someone checks.

## How it works

Caffeinator holds an IOKit power assertion (`PreventUserIdleDisplaySleep`, or `PreventUserIdleSystemSleep` when the display may sleep), the same mechanism `caffeinate` uses. Run `pmset -g assertions` to see it, along with whatever else has been keeping your Mac up at night.

The shortcut uses Carbon's `RegisterEventHotKey`, which needs no Accessibility permission. Triggers listen for system notifications instead of polling. A 30-second timer runs only while the Mac is being kept awake or the remote-session trigger is on.

Log out, restart and shut down send loginwindow the same Apple events as the Apple menu, without its confirmation dialog, so apps with unsaved work can still cancel them. macOS asks for Automation permission the first time. Lock calls `SACLockScreenImmediate` from login.framework.

Scheduled actions keep the Mac from idling to sleep until they run. If the Mac sleeps anyway, for example with the lid closed, a missed action is skipped instead of running late. A 2 AM shutdown has no business happening at 9 AM.

## Build

```sh
scripts/build.sh            # build/Caffeinator.app, universal, ad-hoc signed
scripts/build.sh --install  # also copies it to /Applications and relaunches
swift test --scratch-path /tmp/caffeinator-build
```

Requires Xcode 16 or later. The scripts compile outside the repo, so a checkout in iCloud Drive works.

## Release

Releases are built and published from a Mac. No CI, no YAML:

```sh
scripts/release.sh 1.0.1 --dry-run  # tests and builds, then shows the release notes
scripts/release.sh 1.0.1            # also tags, pushes and publishes the GitHub release
```

It needs a clean `main` and the [GitHub CLI](https://cli.github.com) signed in. The release gets the zip, install steps, its SHA-256 and GitHub's generated changelog.

## License

MIT, see [LICENSE](LICENSE). Use it, change it, ship it; just keep the copyright notice. It comes with no warranty, not even a warranty of wakefulness.
