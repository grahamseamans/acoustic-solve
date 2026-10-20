# 2026-10-19 -- acoustic-solve: extracted from monitoring (refactor level 3)

Brief: the active bass trap project's shared-sim-stack plan (`PLANS/2026-10-19-shared-sim-stack.md`, not public,
"The three repos", steps 0 and 3). User decisions: its own repo, named acoustic-solve, level 3 (no shims, monitoring
moves to the new shape), podd stays separate (`~/podd`, its job contract is imported, never copied).

## What moves, from monitoring's HEAD 72f8efb (after podd left)

| monitoring | acoustic-solve | shape change |
|---|---|---|
| tools/runpod/image/{Dockerfile,start.sh,beat-workload.sh,README.md,.github/workflows/build.yml} | image/, .github/workflows/image.yml | pinned values become build args read from pins.toml; built by this repo's own workflow (proposal below) |
| tools/boundary-lab/*.patch + README, tools/runpod/image/*.patch (byte-identical copies) | patches/boundary-lab/ | ONE copy |
| tools/beat-engine/*.patch + README | patches/beat-engine/ | |
| POD_IMAGE / ENGINE / SOLVER_VERSION in tools/run.py and tools/room/build.py, the Dockerfile ARG defaults, the image README | pins.toml + acoustic_solve.pins | ONE source; tests/test_pins.py fails on any hand copy |
| tools/blabrun | bin/blabrun | the acceptance check reads venv/requirements.lock (every package), not five hand pins; `--acceptance` proves the installed blab is the pin + patches; puts this repo's src/ on the path |
| (none: the venv had no recipe) | venv/build-venv.sh, venv/requirements.lock, venv/lock_from_venv.py | written from the live ~/blab-venv's freeze, hashes from PyPI |
| tools/run-job.sh | src/acoustic_solve/run-job.sh | julia from PATH, not a hard-coded /opt/julia-<version> |
| tools/tests/test_run_job.py | tests/test_run_job.py | |
| the two JOB writers (run.py name_and_queue, room/build.py submit) | acoustic_solve.solve_job.queue_solve | ONE writer of a blab-solve JOB: run-job.sh shipped, requires from the pins |
| tools/room/build.py | acoustic_solve.room.build | pins from acoustic_solve.pins, its JOB through queue_solve |
| tools/room/analyze.py (generic half: load, free field, modal sum, smoothing, peaks, decay, band metrics) | acoustic_solve.room.analysis | reads through the one result reader; the band metrics are one function (analyze and crossover each had their own) |
| model/identity.py canonical JSON + digest + file_sha256 | acoustic_solve.canonical_digest | |
| tools/sounds.py filters / term weights / declaration check / resolution / words | acoustic_solve.weighted_sums | no `<driver>_front/_rear`, no params.parts |
| tools/blab-results/export.py `Result` (and the readers in room/analyze, room/crossover, runstore, model/acoustics/report, build_index) | acoustic_solve.boundary_lab_result | ONE reader of a `blab project solve` result directory |

Stays in monitoring: model/acoustics (scene, assembler), run.py's speaker client, runstore's schema, sounds.py's speaker
half (default sums, roles, the sound file), export.py's CSV writer, polars, viewer, room/crossover.py (A2) and the
cardioid-sub study's patterns and tables (tools/room/cardioid_sub.py).

## Steps

1. Phase A (podd was mid-extraction): build this repo from monitoring's committed HEAD, no edits there.
2. Phase B (after podd's commit landed, 72f8efb): monitoring imports the new shape, the moved files and the dead code
   go, its tests that touch moved code pass.
3. Nothing is pushed: acoustic-solve, monitoring and beat-runpod wait for the user's review.

## Who builds the image (taken)

This repo's own workflow (.github/workflows/image.yml, `workflow_dispatch`) builds
`ghcr.io/grahamseamans/acoustic-solve:<acoustic-solve sha>` from the repo root (Dockerfile at image/, patches at
patches/boundary-lab/, the venv from venv/requirements.lock, build args from pins.toml), so `image_commit` is a commit
of this repo. It ran on 2026-10-20 (fd29901); the package is public. What remains is in SHORTCUTS.md: pin `pod_image`
to the build of the lock-installing commit, then retire the beat-runpod mirror, its deploy key and the subtree push.
