# SHORTCUTS -- jank to come back to (each with when it was taken)

## 2026-10-19 14:00 -- the image's venv is not the lock

image/Dockerfile's venv step still installs Boundary Lab with its dependencies RESOLVED AT BUILD TIME (plus
manifold3d==3.5.3), as the image pinned in pins.toml was built on 2026-10-14. venv/requirements.lock is the local
venv's exact set. So the pod and the local venv can differ in numpy/scipy/meshio patch versions. Not changed in the
extraction because a rebuilt image that installs the lock can change numbers, which needs `solver_version` bumped AND
a new image built AND `pod_image` moved in one go (the user's call: the image build is the proposal in
PLANS/2026-10-19-extraction-from-monitoring.md). Fix: the Dockerfile copies venv/requirements.lock and installs it
`--require-hashes --no-deps` (pymeshlab included or split out), then Boundary Lab `--no-deps`; bump solver_version.

## 2026-10-19 14:00 -- the beat-runpod mirror still builds the live image

pins.toml's `pod_image` is `ghcr.io/grahamseamans/beat-runpod:<sha>`, built by the beat-runpod mirror from
monitoring's old tools/runpod/image/. This repo's workflow (`image.yml`) is dispatch-only and has never run. Until the
proposal is taken, a change to image/ here cannot reach a pod. Fix: the proposal's steps, then retire the mirror and
its deploy key (`github-beat-runpod` in ~/.ssh/config).

(moved from monitoring's SHORTCUTS.md on 2026-10-19 with the patches; dated as taken)

## Boundary Lab carries a local patch series and pins a fork wheel (2026-10-03, series 2026-10-04)
Upstream Boundary Lab `bb9030c4` requests `interface_radiated_pressure` but pins BEAT Engine 0.2.0, which lacks it.
`patches/boundary-lab/0001-pin-beat-engine-ti2.patch` pins the fork release `0.3.0+ti2` (grahamseamans/BEAT_Engine
`v0.3.0-ti2`, built locally, not by a CI workflow). `0002-transfer-impedance-layer.patch` adds the layers (S4).
They are applied by image/Dockerfile and venv/build-venv.sh, from the one copy in patches/boundary-lab/.
- **No GUI editor for R and M.** The System dialog shows and keeps a layer; it is authored in the project file.
- **To close (2026-10-06).** When upstream pins a release with the layers and merges an equivalent of 0002, move the commit pins
  and delete the patches and the apply steps (`patches/boundary-lab/README.md`).
- **status 2026-10-10:** waits on upstream (Boundary Lab pinning a release with the layers); nothing to do here.

## BEAT Engine transfer-impedance S1–S4: fork wheel 0.3.0+ti2, Boundary Lab gated (2026-10-04)
`patches/beat-engine/000{1,2,3,4}-*.patch` against v0.3.0 (core, tests, contract, precompile workload); branch `transfer-impedance` on the fork.
Plan: monitoring's `research/beat-transfer-impedance-plan-2026-10-04.md`.
- **UNPATCHED ENGINES DROP THE FELT SILENTLY.** v0.3.0 given `transfer_impedance` returns the no-layer result.
  Boundary Lab + 0002 is guarded three ways. The `0.3.0+ti2` pin is one. `negotiate_submission` on the persistent
  worker is the second; the one-shot path refuses layers. A result that does not echo the layers in
  `diagnostics["interface_transfer_impedance"]` fails, which is the third. A raw wire request sent to any other
  engine has none of these guards.
- **FP32 LOSES THE EXTERIOR PRESSURE AT LARGE R, AND NOTHING SAYS SO.** Unreduced layout (CPU default `:none`,
  monolithic): relative error 3.9e-4 at 100 ρc, ~10× per decade; `pressure_continuity_error` stays 0. `:pressure`
  and `BLAB_COUPLED_DENSE_REFINEMENT=1` hold 7e-6 to 10⁶ ρc. Fine for felt (0.3 ρc); V-TI1's large-R sweep must
  use FP64, `:pressure` or refinement. A diagnostic that flags it is not written.
- **`pressure_continuity_error` CHANGED MEANING.** With a layer it is the residual of the jump condition. The key was
  not renamed (an upstream question).
- **To close (2026-10-06).** When upstream merges an
  equivalent, delete `patches/beat-engine/` and this entry.
- **status 2026-10-10:** waits on upstream merging an equivalent; nothing to do here.
