# acoustic-solve -- the shared acoustic simulation tooling

Boundary Lab + BEAT Engine solves on [podd](../podd) pods, for any project that needs them (the speaker
project's cabinets and rooms, the active bass trap's rooms). Extracted from monitoring on 2026-10-19
(`PLANS/2026-10-19-extraction-from-monitoring.md`). The partitions and their patterns: `architecture/README.md`.

License: GPL-3.0-or-later (`LICENSE`), because it ships patched Boundary Lab and BEAT Engine (`patches/`), both GPL-3.

| Path | What |
|---|---|
| `pins.toml` | THE PINS: the pod image, `engine`, `solver_version`, Boundary Lab's commit, Julia, the CUDA base. Written nowhere else (`tests/test_pins.py`). |
| `image/` | The solver image (Dockerfile, start.sh, beat-workload.sh); `image/README.md` is its story and its measurements. |
| `.github/workflows/image.yml` | Builds and pushes the image, by hand only (`gh workflow run image.yml`). |
| `patches/boundary-lab/`, `patches/beat-engine/` | The local patch series (one copy each), with why they exist. |
| `venv/` | The blab venv: `build-venv.sh` (the recipe), `requirements.lock` (hash-locked), `lock_from_venv.py` (writes the lock). |
| `bin/blabrun` | THE way into the blab venv; checks it against the lock first; `--acceptance` also proves the installed `blab` is the pin + patches. |
| `src/acoustic_solve/` | The package: `pins`, `canonical_digest`, `weighted_sums`, `boundary_lab_result`, `solve_job` (+ `run-job.sh` and `collect_result`, the pod side: raw/ built per frequency, reduced to a room's kept nodes), `blab_venv`, `room/` (`spec`, `build`, `analysis`, `traps`, `field`). |
| `tests/` | One file per module; no pod, no solve. `tests/pod_gates/`: checks staged as podd solves, run by hand. |
| `references/` | What the code cites (`references/README.md`). |

## Install (this machine)

    ln -s ~/acoustic-solve/bin/blabrun ~/.local/bin/blabrun     # blabrun on PATH, for every client
    blabrun --acceptance                                         # the live ~/blab-venv is the lock + the pinned blab

A client imports the package through `blabrun` (it puts `src/` on `PYTHONPATH`; the package reads `pins.toml` and the
lock from this checkout, so it is never pip-installed). A client that runs in another interpreter (monitoring's
`tools/uvrun`, build123d) puts `~/acoustic-solve/src` on its own `PYTHONPATH` and imports only the standard-library
modules (`canonical_digest`, `weighted_sums`, `pins`).

A new venv from scratch (never over the live one): `venv/build-venv.sh <new venv dir>`, then point `BLAB_VENV` at it
or move it into place. After a deliberate change to the venv: `python3 venv/lock_from_venv.py <venv> >
venv/requirements.lock`, review the diff, bump `solver_version` (the image installs the same lock).

## Use

    from acoustic_solve import pins, solve_job, weighted_sums, canonical_digest
    from acoustic_solve.boundary_lab_result import BoundaryLabResult
    from acoustic_solve.room import analysis, field, traps

    blabrun -m acoustic_solve.room.build <stage> <room.json>               # a room: interior FEM project + seat nodes
    blabrun -m acoustic_solve.room.build freefield <stage> <room.json>...  # the same box alone: exterior BEM
    blabrun -m acoustic_solve.room.build submit <stage> <expected_s> <ram_gb> <cpu>   # the JOB (stage in a podd queue)

A solve is queued by `solve_job.queue_solve` (the one JOB writer: run-job.sh shipped, the pins required) into a podd
queue directory; podd (`~/podd`) runs it on a pod and leaves DONE; `BoundaryLabResult(<stage>/raw)` reads it.

## Tests

    for test in tests/test_*.py; do blabrun "$test" || break; done

No pod, no solver run; `test_room_build.py` and `test_room_traps.py` mesh one small room with gmsh (seconds). Nothing
heavy runs on the local machine: solves go to podd's pods. The pod gates are run by hand, e.g. the trap face as a
velocity port (PLANS/2026-10-20-trap-velocity-port.md, gate 2):

    blabrun tests/pod_gates/trap_port_equivalence.py stage <podd queue dir>    # three CPU solves, JOBs written
    blabrun tests/pod_gates/trap_port_equivalence.py check <podd queue dir>    # once podd brought them home
