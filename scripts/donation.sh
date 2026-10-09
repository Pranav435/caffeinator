#!/bin/bash
# Records a donation in the README, then commits and pushes it.
#   scripts/donation.sh 5            adds $5 to the notarization fund
#   scripts/donation.sh 5 "Ada L."   also lists Ada L. under "Fueled by"
# Only list people who asked to be listed.
set -euo pipefail
cd "$(dirname "$0")/.."

amount="${1:?Usage: scripts/donation.sh <dollars> [name]}"
name="${2:-}"
[[ "$amount" =~ ^[0-9]+$ ]] || { echo "Use whole dollars, like 5." >&2; exit 1; }

total=$(( $(perl -ne 'print $1 if /Notarization fund: \*\*\$(\d+)/' README.md) + amount ))
perl -pi -e "s/(Notarization fund: \\*\\*\\\$)\\d+/\${1}$total/" README.md

if [[ -n "$name" ]]; then
    # Newest donor first, right under the marker. The empty-mug line goes away with the first name.
    NAME="$name" perl -0pi -e 's/(<!-- donors:[^\n]*-->\n)(?:Empty, like a mug on Monday morning\.\n)?/$1- $ENV{NAME}\n/' README.md
fi

git add README.md
git commit -qm "Notarization fund: \$$total of \$99${name:+ (thanks, $name)}"
git push -q origin main
echo "Fund is at \$$total of \$99${name:+, and $name is listed}."
