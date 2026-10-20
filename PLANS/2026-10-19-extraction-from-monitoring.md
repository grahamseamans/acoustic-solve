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

## Proposal: who builds the image

Today: `git subtree push --prefix=tools/runpod/image github-beat-runpod:grahamseamans/beat-runpod.git main`, a mirror
repo whose Actions build `ghcr.io/grahamseamans/beat-runpod:<mirror sha>`. The image's `image_commit` fact is then a
mirror commit no source repo has.

Proposed: this repo's own workflow (.github/workflows/image.yml, `workflow_dispatch` only until the user switches it on)
builds `ghcr.io/grahamseamans/acoustic-solve:<acoustic-solve sha>` from the repo root (Dockerfile at image/, patches at
patches/boundary-lab/, build args from pins.toml). image_commit is then a commit of this repo. The beat-runpod mirror,
its deploy key and the subtree push retire. Steps for the user (none done here):

    git -C ~/acoustic-solve push -u origin main                       # the workflow does not run on push
    gh workflow run image.yml -R grahamseamans/acoustic-solve          # build + push :<sha> and :latest
    # GHCR -> package acoustic-solve -> settings -> visibility Public (RunPod pulls without a credential)
    # then in pins.toml: pod_image = "ghcr.io/grahamseamans/acoustic-solve:<sha>", and solver_version bumped if the
    # rebuild can change a number (see SHORTCUTS: the image venv resolves its dependencies at build time)

Until then pins.toml keeps pointing at the current beat-runpod image, which this tree describes.
