# patches/boundary-lab: local patch series to Boundary Lab

A `git format-patch` series against Boundary Lab `bb9030c4ae0b5906569b3b3932e221a0c97670ac` (upstream `main` HEAD on
2026-10-03, the merge of `feat/interface-radiation`). The same commits are branch `transfer-impedance` on the fork
`https://github.com/grahamseamans/boundary-lab`. A working clone lives outside this repo (`~/boundary-lab`).

| Patch | What it holds |
|---|---|
| `0001-pin-beat-engine-ti2.patch` | The BEAT Engine pin: wheel URL + sha256 in `pyproject.toml`, the version check in `src/blab/solvers/engine_distribution.py`, its test |
| `0002-transfer-impedance-layer.patch` | Step S4 of the speaker project's research note `beat-transfer-impedance-plan-2026-10-04.md` (not public): interface transfer-impedance layers on Boundary Lab's side |

## 0001: the pin

- `bb9030c4` makes every coupled project with FEM-BEM interfaces request `interface_radiated_pressure`. Its pinned
  BEAT Engine 0.2.0 rejects that output (`Unsupported coupled output quantity`), so the polars are lost
  (the speaker project's research note `cabinet-mesh-spike-2026-10-03.md` (not public)). 0.3.0 implements it.
- The pin is the fork release `0.3.0+ti2` = v0.3.0 + transfer-impedance layers + the coupled precompile workload
  (`patches/beat-engine/`): `https://github.com/grahamseamans/BEAT_Engine/releases/tag/v0.3.0-ti2`, wheel sha256
  `69905d7353b7bbc132ec6d01d6eba465c77b8d65b42b0be508d860604d5bf111`. The tag is `v0.3.0-ti2` because git tags
  cannot carry the PEP 440 `+`. Built locally from the tag with `python -m build`: upstream's release workflow only
  accepts stable/rc versions released from a CI-green `main`.
- Checked 2026-10-03, CPU: 0.3.0 is bit-identical to 0.2.0 on `horns/core-1.4in/results` (24 points) and on the
  coupon `model/acoustics/tests/fixtures/ath-mouth-box-coupon/`. With no layer, 0.3.0+ti1 is bit-identical to 0.3.0
  (`patches/beat-engine/README.md`).

## 0002: transfer-impedance layers

- **Where R and M are set.** In the project file, on the interface's bounded (FEM-side) boundary:
  `physical_system.boundaries[...].parameters = {"transfer_impedance": {"model": "resistance_mass",
  "resistance_pa_s_per_m": R, "mass_kg_per_m2": M}}`. R and M are referred to the interface face area. All three
  keys are required. `blab project solve` takes it as is. The System dialog shows the layer on the interface row
  ("Layer R Pa·s/m / M kg/m²") and keeps it on save, but has no editor for it. Before 0002 it dropped any
  interface parameter on save.
- **Validation.** `acoustic_materials.transfer_impedance_parameters` refuses missing or unknown keys, other
  models, and R or M that is negative, not finite or not a number. The compiler and the coupled backend refuse a
  layer anywhere but the bounded side of an interface.
- **Gate.** The persistent worker's client (`beat_engine.EngineWorker`) runs `negotiate_submission`. That refuses a
  layer unless the worker advertises `optional_physics: ["interface_transfer_impedance"]`. The one-shot stdin
  path cannot negotiate, so it refuses layers.
- **Read-back.** Every result must carry `diagnostics["interface_transfer_impedance"]` with the requested boundaries
  and R, M, to FP32 rounding. Otherwise the solve fails. A result that carries a layer the request did not ask
  for also fails.
- **Docs.** `Physical System Model.md` (new "Transfer-impedance layers" section), `Coupled Solver.md` (the matrix
  entry; the unsupported list), `advanced/Phasor Convention.md` (a table row).

**Acceptance, 2026-10-04, CPU aarch64, FP32 production path.** `blab project solve` on `examples/Vented_Sub`, with
and without a layer of R = 125 Pa·s/m and M = 0.004 kg/m² on the port interface:

| f (Hz) | port velocity: none → layer (m/s) | Change | Far field at 3 m: none → layer (Pa) | Change |
|---|---|---|---|---|
| 25 | 0.351 → 0.216 | −4.2 dB | 0.0880 → 0.0681 | −2.2 dB |
| 40 | 0.491 → 0.203 | −7.7 dB | 0.499 → 0.230 | −6.7 dB |
| 63 | 0.0951 → 0.0835 | −1.1 dB | 0.385 → 0.354 | −0.7 dB |

The diagnostics carry R = 125, M = 0.004 (as FP32) and Im Z_s = ωM = 1.005 Pa·s/m at 40 Hz. The jump residual
(`pressure_continuity_error`) is ≤ 4e-8. Each solve takes about 45 s.

**Tests.** From `src/`, in a venv with the pinned wheel, Julia on PATH and `QT_QPA_PLATFORM=offscreen`: `python -m
pytest`. Set `BLAB_RUN_COUPLED_REFERENCE=1` to add the real FP64 fixture solve with a layer.

| | bb9030c4 + old pin | + 0001–0002 |
|---|---|---|
| passed | 1104 | 1123 |
| skipped | 27 | 28 (the gated solve; it passes when enabled) |
| failed + errors | 13 + 18 | the same 13 + 18 |

The 31 failures and errors are environment gaps in the test venv, not Boundary Lab bugs: no `pyvista`, `pyvistaqt`,
`vtk` or `wine`, plus one `test_project_io` assertion.

## Where it is applied

Always with `git apply` onto a checkout of `boundary_lab_commit` (pins.toml), so `HEAD` stays the pin. These two
files are the ONE copy; both users read them from here:
- `image/Dockerfile` (the pod image: COPY from this folder, the repository root is the build context);
- `venv/build-venv.sh` (this machine's `~/blab-venv`; `bin/blabrun --acceptance` proves the installed `blab` is the pin
  plus exactly these patches).

After a patch changes: bump `solver_version` (pins.toml), rebuild the venv (`venv/build-venv.sh`) and the image.

**When it goes away.**
- When upstream pins a BEAT Engine with transfer-impedance layers and merges an equivalent of 0002:
  1. Move the commit pins.
  2. Delete this folder, the links and the apply steps.
  3. Delete the SHORTCUTS.md entries.
- If only the pin moves upstream, 0001 goes away and 0002 is rebased.
