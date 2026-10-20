#!/bin/bash
# THE BLAB VENV FROM SCRATCH: the recipe the live ~/blab-venv is checked against (bin/blabrun) and rebuilt by.
#
#   venv/build-venv.sh <new venv dir> [<Boundary Lab clone, default ~/boundary-lab>]
#
# 1. python3.12 -m venv (Ubuntu 24.04's python3.12; the lock is for 3.12).
# 2. venv/requirements.lock with --require-hashes --no-deps: exactly those files, no resolver (the lock is closed:
#    `pip check` passed on the venv it was written from).
# 3. Boundary Lab = boundary_lab_commit (pins.toml) from the clone's git objects + patches/boundary-lab/*.patch,
#    installed non-editable (the clone's working tree is never read; it is cloned if absent).
# 4. podd, editable from $PODD_ROOT (default ~/podd): its job contract (acoustic_solve.solve_job imports it).
# 5. bin/blabrun --acceptance against the new venv.
# Downloads ~400 MB of wheels and builds one pure-Python wheel (Boundary Lab): no compile.
set -euo pipefail
repository=$(dirname "$(dirname "$(readlink -f "$0")")")
venv=${1:?usage: venv/build-venv.sh <new venv dir> [<Boundary Lab clone>]}
clone=${2:-$HOME/boundary-lab}
podd_root=${PODD_ROOT:-$HOME/podd}
[ ! -e "$venv" ] || { echo "build-venv.sh (╯°□°)╯ $venv exists: a venv is built new, never over another" >&2; exit 1; }
commit=$(PYTHONPATH="$repository/src" python3 -c 'from acoustic_solve import pins; print(pins.BOUNDARY_LAB_COMMIT)')

python3.12 -m venv "$venv"
"$venv/bin/pip" install --no-deps --require-hashes -r "$repository/venv/requirements.lock"

source_tree=$(mktemp -d)
trap 'rm -rf "$source_tree"' EXIT
[ -d "$clone/.git" ] || git clone https://github.com/JWSound/boundary-lab "$clone"
git -C "$clone" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$clone" fetch origin
git -C "$clone" archive "$commit" | tar -x -C "$source_tree"
(cd "$source_tree" && git apply "$repository"/patches/boundary-lab/*.patch)
"$venv/bin/pip" install --no-deps "$source_tree"

"$venv/bin/pip" install --no-deps -e "$podd_root"

BLAB_VENV="$venv" BOUNDARY_LAB_CLONE="$clone" "$repository/bin/blabrun" --acceptance
