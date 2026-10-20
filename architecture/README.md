# architecture -- acoustic-solve's partitions and their patterns

Written 2026-10-19 with the extraction from monitoring (PLANS/2026-10-19-extraction-from-monitoring.md). What this
repo is FOR: one solver (Boundary Lab + BEAT Engine, pinned and patched), run the same way on a pod for every
project, and read back the same way. A project brings its geometry and its questions; everything between "a staged
project" and "the numbers it solved to" is here, once.

```
  pins.toml ------------------------------------------------------------------------------.
     |  (python: acoustic_solve.pins; the image: build arguments; the venv: blab_venv)      |
     v                                                                                      v
  (1) THE SOLVER         image/ + patches/ + venv/ + bin/blabrun          what a solve RUNS ON, here and on a pod
     |
  (2) THE SOLVE          solve_job.py + run-job.sh                        a staged project -> a podd JOB -> raw/
     |                   (podd.job_contract, imported from ~/podd)
  (3) THE RESULT         boundary_lab_result.py + weighted_sums.py        raw/ -> numbers, and sums of ports
     |
  (4) STUDIES' TOOLS     room/ (spec, build, analysis, traps)             a kind of model and its measures
     |
  (0) LEAVES             canonical_digest.py                              standard library, depends on nothing
```

Imports point DOWN this list only (a study's tool may use the result reader; the reader never knows a room), and
nothing here imports a client (monitoring, the bass trap). podd is the one outside dependency, through its job
contract only (`from podd import job_contract`): never a copy of it, never its internals.

## (pins) ONE WRITTEN VALUE PER PIN

`pins.toml` is the only place a pod image tag, the engine version, the solver version, Boundary Lab's commit, Julia's
version or the CUDA base is written. Python reads it through `acoustic_solve.pins` (constants, read once, refused
loudly when malformed); the image gets every one as a build argument with NO default in the Dockerfile; the venv's
lock is checked against it. `tests/test_pins.py` fails on a hand copy in any code line. A client never writes a pin:
it imports `pins` (monitoring's run id hashes `pins.ENGINE` and `pins.SOLVER_VERSION`).

## (1) THE SOLVER -- reproducible from this repo alone

- The patch series live in `patches/` once; the image COPYs them, the venv recipe applies them. A second copy of a
  patch anywhere is a violation (monitoring held two byte-identical copies until 2026-10-19).
- The local venv is `venv/requirements.lock` exactly (hash-locked, `--no-deps`), plus three things installed from
  source by `venv/build-venv.sh`: Boundary Lab (the pinned commit + patches, from git objects), podd (editable), pip.
  `bin/blabrun` refuses a venv that is anything else, on every run; `--acceptance` proves the installed blab byte for
  byte. Nothing is installed into the venv by hand: change the recipe, rebuild, re-lock.
- KNOWN SEAM (SHORTCUTS.md): the image's venv still resolves its packages at build time instead of installing the
  lock. The next image build closes it.
- `bin/blabrun` is the only way into the venv; it puts `src/` on the path. The package is never pip-installed: it
  reads pins.toml and the lock from its own checkout.

## (2) THE SOLVE -- one JOB writer, one pod-side command

- `solve_job.queue_solve` is the ONLY writer of a solve's JOB, for every client: it ships run-job.sh, writes the
  command, and sets `requires` to the pins (a client adds only its own: `gpu_memory_gb_min`, `gpu: "none"`; naming
  a pinned key is refused). What a client measured about its solves (cpu, card memory, RAM, expected_s) is its own
  argument.
- `run-job.sh` is the pod side: one `blab project solve`, its logs, `progress`, and the per-frequency resume. It
  finds julia and blab on PATH (the image puts them there); it knows no client.
- Nothing here talks to RunPod, ssh-es to a pod, or reads podd's state beyond `job_contract`.

## (3) THE RESULT -- one reader, one algebra

- `BoundaryLabResult` is the ONLY reader of a result directory: manifest first (cheap: an index over thousands of
  rows reads manifests only), every array on first use, every inconsistency refused loudly (incomplete, ports or
  convention per frequency, a quantity at some frequencies only, axes). A second reader of manifest.json /
  domains.json / frequencies/*.json anywhere -- here or in a client -- is a violation.
- `weighted_sums` is THE algebra of summing ports: filters, delay, polarity, gain, declarations, resolution, words.
  Standard library only. It knows ports and sums, never a driver, a face or a client's naming of either.
  `BoundaryLabResult.weighted_terms` / `weighted_sum` apply it to arrays.

## (4) STUDIES' TOOLS -- a kind of model, generic over the studies that use it

- `room/` is the room solver: `spec` (the room.json grammar's gmsh-free half: air, seat points and their names, the
  box frame), `build` (gmsh + the Boundary Lab project, interior FEM and the box's free field; its JOB via solve_job),
  `analysis` (read a room solve, join chunks, the free field, the modal-sum validation, smoothing, peaks, decay, the
  room-gain band metrics), `traps` (a room whose trap faces are velocity ports: every port's exact face-averaged
  pressure, the trap mics, the closure of a law over them, the watts each port moves). A study's own mix of ports
  (cardioid patterns, a crossover) and its own control laws and tables stay in the study's repo; they import these.
- Every name a room stage's files share between the builder and the readers (seat and trap point names, trap groups,
  sensed quantities, the stage's file names) is spelled once, in `spec`. Anything that is a port -- a source patch or
  a velocity_port trap face -- is built by one function (`build.velocity_port_parts`) and read by one rule (the port
  id is "excitation:" + its physical group).
- A new kind of model (a loop sim's plant, a duct) is a new folder here only when two projects need it.

## (0) LEAVES

`canonical_digest`: the canonical JSON and the 12-hex-digit id. Standard library, imported by any environment
(monitoring's build123d venv too). What a client hashes is the client's.

## Tests

One test file per module, plain scripts (`python3` for the standard-library ones, `blabrun` for the rest), each
proving its refusals as well as its answers. No test solves, opens a pod or needs the network.

`tests/pod_gates/` holds the checks that need a real solve: each one STAGES its solves for podd (`stage <queue dir>`)
and CHECKS them once they are home (`check <queue dir>`), its thresholds written before the solve. The test loop
(`tests/test_*.py`) never runs them.
