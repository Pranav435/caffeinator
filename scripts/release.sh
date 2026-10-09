#!/bin/bash
# Publishes a GitHub release from this Mac: tests, builds the universal zip, tags, pushes and uploads.
#   scripts/release.sh 1.0.1            release v1.0.1
#   scripts/release.sh 1.0.1 --dry-run  everything except the tag, push and upload
# Needs the GitHub CLI, signed in (gh auth login).
set -euo pipefail
cd "$(dirname "$0")/.."

version="${1:?Usage: scripts/release.sh <version> [--dry-run]}"
version="${version#v}"
fail() { echo "$1" >&2; exit 1; }
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "Use a version like 1.2.3."
[[ -z "$(git status --porcelain)" ]] || fail "Commit or stash your changes first."
[[ "$(git branch --show-current)" == main ]] || fail "Switch to main first."
git rev-parse -q --verify "refs/tags/v$version" >/dev/null && fail "v$version already exists."
gh auth status >/dev/null 2>&1 || fail "Sign in to GitHub first: gh auth login"

export SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/caffeinator-build}"
swift test --scratch-path "$SCRATCH"
VERSION="$version" scripts/build.sh --zip

sum="$(shasum -a 256 build/Caffeinator.zip | cut -d ' ' -f 1)"
notes="Download **Caffeinator.zip**, unzip it, and move Caffeinator to Applications.

The app is ad-hoc signed, not notarized, so macOS blocks the first launch. Open it once, then go to System Settings › Privacy & Security and click **Open Anyway**. Or run:

\`\`\`
xattr -dr com.apple.quarantine /Applications/Caffeinator.app
\`\`\`

macOS 14 or later, Apple silicon and Intel.

Free, no ads, no tracking. If it rescues a render, [buy its developer a coffee](https://paypal.me/theblindiephoenix).

SHA-256: \`$sum\`"

if [[ "${2:-}" == "--dry-run" ]]; then
    printf 'Dry run. A real run tags v%s, pushes main and the tag, and publishes build/Caffeinator.zip with:\n\n%s\n' "$version" "$notes"
    exit 0
fi

git tag -a "v$version" -m "Caffeinator $version"
git push origin main "v$version"
# --notes goes above GitHub's generated changelog.
gh release create "v$version" build/Caffeinator.zip --verify-tag --title "Caffeinator $version" --notes "$notes" --generate-notes
