#!/bin/bash
# Builds build/Caffeinator.app as a universal (Apple silicon + Intel), ad-hoc signed app.
#   scripts/build.sh            build
#   scripts/build.sh --zip      also write build/Caffeinator.zip for a release
#   scripts/build.sh --install  also copy to /Applications and relaunch
set -euo pipefail
cd "$(dirname "$0")/.."

version="${VERSION:-$(git describe --tags --abbrev=0 2>/dev/null || echo 1.0.0)}"
version="${version#v}"
number="$(git rev-list --count HEAD 2>/dev/null || echo 1)"

# Compile outside the repo. Synced folders like iCloud Documents add Finder metadata
# that codesign refuses, and would upload the build cache.
flags=(-c release --arch arm64 --arch x86_64 --scratch-path "${SCRATCH:-${TMPDIR:-/tmp}/caffeinator-build}")
swift build "${flags[@]}"
bin="$(swift build "${flags[@]}" --show-bin-path)"

# Assemble and sign in a temp folder for the same reason.
stage="$(mktemp -d)/Caffeinator.app"
mkdir -p "$stage/Contents/MacOS" "$stage/Contents/Resources"
# SwiftPM records the deployment target as the SDK version. Stamp the real one so
# newer macOS releases don't run the app in compatibility mode.
vtool -set-build-version macos 14.0 "$(xcrun --show-sdk-version)" -replace -output "$stage/Contents/MacOS/Caffeinator" "$bin/Caffeinator"
cp Resources/Info.plist "$stage/Contents/"
cp Resources/AppIcon.icns "$stage/Contents/Resources/"
plutil -replace CFBundleShortVersionString -string "$version" "$stage/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$number" "$stage/Contents/Info.plist"
codesign --force --sign - --timestamp=none "$stage"
codesign --verify --strict "$stage"

mkdir -p build
rm -rf build/Caffeinator.app
ditto "$stage" build/Caffeinator.app
echo "Built build/Caffeinator.app $version ($number)"

for arg in "$@"; do
    case "$arg" in
    --zip)
        rm -f build/Caffeinator.zip
        ditto -c -k --keepParent "$stage" build/Caffeinator.zip
        echo "Wrote build/Caffeinator.zip"
        ;;
    --install)
        pkill -x Caffeinator || true
        rm -rf /Applications/Caffeinator.app
        ditto "$stage" /Applications/Caffeinator.app
        open /Applications/Caffeinator.app
        echo "Installed /Applications/Caffeinator.app"
        ;;
    esac
done
