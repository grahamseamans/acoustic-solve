#!/bin/bash
# Boundary Lab's own docs AS THIS REPO SOLVES WITH THEM: docs/ and the headless result writer at boundary_lab_commit
# (pins.toml) with patches/boundary-lab/ applied (0002 edits three of the docs), into references/boundary-lab/
# (gitignored: re-made by this script, from the clone's git objects, no network unless the clone lacks the commit).
#
#   references/mirror_boundary_lab_docs.sh [<Boundary Lab clone, default ~/boundary-lab>]
set -euo pipefail
repository=$(dirname "$(dirname "$(readlink -f "$0")")")
clone=${1:-$HOME/boundary-lab}
commit=$(PYTHONPATH="$repository/src" python3 -c 'from acoustic_solve import pins; print(pins.BOUNDARY_LAB_COMMIT)')
out="$repository/references/boundary-lab"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
git -C "$clone" archive "$commit" | tar -x -C "$scratch"
(cd "$scratch" && git apply "$repository"/patches/boundary-lab/*.patch)
rm -rf "$out"
mkdir -p "$out/src/blab"
cp -r "$scratch/docs" "$out/docs"
cp "$scratch/src/blab/headless.py" "$scratch/src/blab/physical_model.py" "$out/src/blab/"
echo "boundary_lab_commit $commit + $(ls "$repository"/patches/boundary-lab/*.patch | wc -l) patches, $(date -u +%F)" \
  > "$out/MIRRORED.txt"
echo "mirrored into $out: $(find "$out" -type f | wc -l) files"
