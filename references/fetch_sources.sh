#!/bin/sh
# THE MIRROR OF THE DOWNLOADABLE SOURCES THE CODE CITES (2026-10-20). Git keeps this script, not the files (the
# repo's .gitignore ignores references/*/). Every file is pinned by its sha256: a source that changed under its URL
# fails loudly here, and then the numbers cited from it must be re-checked before the pin is moved.
#
#   sh references/fetch_sources.sh
#
#   sources/sengpiel-comparative-representation-of-sound-field-quantities.pdf
#       E. Sengpiel, "Comparative representation of sound field quantities": dry air at 20 degC, 101325 Pa: density
#       1.204 kg/m^3, speed of sound 343 m/s, rho c 413 N s/m^3 (lines 6-10 of `pdftotext -layout`).
#       src/acoustic_solve/room/spec.py's DENSITY_KG_PER_M3, SOUND_SPEED_M_PER_S.
set -eu
here=$(dirname "$(readlink -f "$0")")
fetch() {  # url, destination (relative to references/), sha256
  destination="$here/$2"
  mkdir -p "$(dirname "$destination")"
  curl --fail --silent --show-error --location --user-agent "Mozilla/5.0" --output "$destination.part" "$1"
  actual=$(sha256sum "$destination.part" | cut -d' ' -f1)
  if [ "$actual" != "$3" ]; then
    echo "fetch_sources.sh (╯°□°)╯ $1 changed: sha256 $actual, pinned $3. Re-check what cites it, then move the pin." >&2
    rm -f "$destination.part"
    exit 1
  fi
  mv "$destination.part" "$destination"
  echo "fetched $2"
}
fetch https://www.sengpielaudio.com/ComparativeRepresentationOfSoundFieldQuantities.pdf \
  sources/sengpiel-comparative-representation-of-sound-field-quantities.pdf \
  f13f978d7783c7ee319d7ac954fd5527bb35c150b25a954c1c0fa65ff1023f8d
