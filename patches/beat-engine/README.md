# patches/beat-engine: local patch series to BEAT Engine

A `git format-patch` series against BEAT Engine tag `v0.3.0` (`e6b3037`, the merge of `release/0.3.0`). Steps S1–S3
of the speaker project's research note `beat-transfer-impedance-plan-2026-10-04.md` (not public). The same commits are branch `transfer-impedance` on the fork
`https://github.com/grahamseamans/BEAT_Engine`.

| Patch | Step | What it holds |
|---|---|---|
| `0001-transfer-impedance-core.patch` | S1 | Julia core: the jump block, the parse, all paths, refusals, diagnostics, the block test |
| `0002-transfer-impedance-tests.patch` | S2 | Physics and wire tests; a row in `julia_local/tests/README.md` |
| `0003-transfer-impedance-contract.patch` | S3 | Worker capability, client gate, conformance case, version `0.3.0+ti1` |
| `0004-precompile-workload.patch` | — | Coupled worker in the bundles with a precompile workload, version `0.3.0+ti2` |

**Why it exists.**
- The felt across the head slots is a two-sided resistive layer. The solver's FEM-BEM interface holds only pressure
  continuity, so it cannot represent the felt (`cabinets/comparisons/slots-2026-10-03/README.md`).
- Upstream has no transfer-impedance interface.

**Apply.** Into a working clone outside this repo (call it `<clone>`):
```
git clone --branch v0.3.0 https://github.com/JWSound/BEAT_Engine <clone> && git -C <clone> am $PWD/patches/beat-engine/000*.patch
```
or `git clone --branch transfer-impedance https://github.com/grahamseamans/BEAT_Engine <clone>`.

**Release.** Tag `v0.3.0-ti2` (git tags cannot carry the PEP 440 `+`) on `5d21ebf`, package version `0.3.0+ti2`:
`https://github.com/grahamseamans/BEAT_Engine/releases/tag/v0.3.0-ti2`, with the wheel, the sdist and
`SHA256SUMS.txt`. Wheel sha256 `69905d7353b7bbc132ec6d01d6eba465c77b8d65b42b0be508d860604d5bf111`. Built with
`python -m build` from a clean clone of the tag and uploaded through the API (`v0.3.0-ti1` on `cce2dd2` the same way).
Upstream's `release.yml` cannot release it: `release_tools.py check` accepts only `X.Y.Z(rcN)` with tag `v` + version,
and `require-ci` wants a green `main` push CI on the commit. Verified 2026-10-06: anonymous download matches
`SHA256SUMS.txt`; `pip install` gives `__version__ == "0.3.0+ti2"`, `worker-v1.json` lists
`optional_physics: ["interface_transfer_impedance"]`, and `julia_engine/BeatEngineCoupledWorker.jl` (the workload,
`WORKLOAD[]`) is in the wheel. Boundary Lab pins it in `patches/boundary-lab/0001-pin-beat-engine-ti2.patch`.

**Test.** From `<clone>/src/beat_engine/julia_local`, with Julia 1.12 on PATH:
```
julia --threads=2 --startup-file=no --project=. tests/runtests.jl
julia --threads=2 --startup-file=no --project=. tests/reference_tests.jl
```
Python, from `<clone>` in any venv with `pip install -e ".[dev]"`: `python -m pytest`.
GPU arms (the pod only): `BLAB_RUN_COUPLED_CUDA=1` (or `_ROCM=1`) with the `julia_cuda` (`julia_rocm`) project.

## 0001: the core (S1)

- **Physics.** The interface condition becomes `p_F - p_B = Z_s v_n` with `Z_s = R + time_derivative(omega) M`.
  This is one diagonal entry `-d` per interface dof on the flux-flux block, with `d = Z_s / neumann_scale`.
- **Where the block is built.** `interface_transfer_impedance_jump_coefficients` in `BeatEngineCoupled.jl` is the
  one place.
- **Which paths use it.** Host monolithic; CUDA/ROCm, as a sparse scatter; condensed `:none` and `:pressure`.
- **Refusals.** `:flux` elimination refuses under `on` and falls back to `:pressure` under `auto`. The speaker ROM
  refuses. The interior-only solve refuses.
- **Wire format.** `boundary.parameters["transfer_impedance"] = {"model": "resistance_mass",
  "resistance_pa_s_per_m": R, "mass_kg_per_m2": M}` on the interface's bounded (FEM-side) boundary. R and M are
  referred to the interface face area. Refused on any other kind of boundary, on the BEM side, on a boundary with no
  interface, with unknown or missing keys, and when R or M is negative, not a number, or not finite.
- **Diagnostics.** `pressure_continuity_error` is now the residual of the jump condition. `interface_transfer_impedance`
  gives R, M, Z_s, the convention and the normal orientation. It is present only when the request carries a layer.
- **Zero layer.** With no layer, or with R = M = 0 exactly, nothing is written, so the results are the unpatched ones.

Checked 2026-10-04 by wire requests on the fixtures (FP64, 300 and 900 Hz; monolithic, condensed `:none`, `:pressure`,
`:flux`): no layer and R = M = 0 are **bit-identical** to v0.3.0 (every array and diagnostic except timings and source
hashes); R = 125 rayl, M = 0.004 kg/m² agree across the four paths to 2e-14; v0.3.0 given the same layer returns the
no-layer result, i.e. **it drops the felt silently**.

## 0002: tests (S2)

All in `julia_local/tests/interface_transfer_impedance_tests.jl` (both runners include it) except the wire test.
Fixture: `tests/fixtures` (a piston, a 785 cm³ box and a 20 mm, r = 19 mm neck opening through the interface: a
Helmholtz resonator at 308 Hz).

| Test | Oracle | Tolerance | Measured |
|---|---|---|---|
| Voltage-driven transducer | Monolithic vs condensed `:none` (retained and condensed transducer) and `:pressure` | 1e-9 | ≤ 2e-14 |
| Phasor conjugation, M ≠ 0 | e^{+iωt} solve = conj of e^{-iωt}, monolithic and condensed `:pressure` | rtol 1e-8 | pass |
| V-TI3 power | Piston = layer + radiated; layer = ½R∫\|v\|², consistent and nodal | 1e-9 | ~1e-15 |
| V-TI2 lumped vent flow | Series-impedance prediction, 60 and 100 Hz, R ∈ {0.3, 1, 3} ρc, M = 0.02, (125, 0.004) | 1 %, 0.5° | 0.47 %, 0.08° |
| V-TI2 Helmholtz shift | Mass layer sized to lower f_H by 30 % | 2 % | 0.23 % (215.23 vs 215.73 Hz) |
| FP32, R = 10^k ρc, k ≤ 6 | FP64, same R | 1e-4 (see below) | below |
| CUDA, ROCm | CPU FP32 | 1e-4 fields, 1e-3 jump | **not run**: no GPU; skip as broken with a warning |
| Wire (`contract_tests.jl`) | `solve_request` on a fixture request | exact messages | 33 pass |

- **V-TI3 numbers.** 500 Hz monolithic: piston 14.05 mW = layer 13.42 mW + radiated 0.63 mW (no layer: 1.71 mW, all
  radiated). 300 Hz condensed `:pressure`: 65.2 = 64.3 + 0.94 mW. The balance is exact because the interior is
  lossless and all powers use the consistent interface mass, so the FEM rows make it a discrete identity.
- **V-TI2 tolerance.** The vent branch Z_v0 is read per frequency from the R = M = 0 run; every layered run is a
  prediction `U = U_in / (1 + Y_C (Z_v0 + Z_s/S))` with `Y_C = time_derivative(ω) V/(ρc²)`. The residual is the
  cavity's distributed correction, which a shunt term cannot absorb. It is a clean f² law, |ΔU/U| ≈ 4.7e-7 f²
  (0.47 % at 100 Hz, 4.2 % at 300 Hz), the same for every R and M. A wrong sign or area scaling misses by tens of
  percent.
- **FP32 with large R (500 Hz, relative error vs FP64).** Flux and interior pressure: ≤ 2e-6 on every path up to
  10⁶ ρc. Exterior (BEM) pressure:

  | R / ρc | 1 | 10 | 100 | 10³ | 10⁶ |
  |---|---|---|---|---|---|
  | `:none` (CPU default) | 7.6e-6 | 2.9e-5 | 3.9e-4 | 4.2e-3 | 2.4 |
  | `:pressure` | 6.5e-6 | | | | 7.0e-6 |
  | `:none` + `BLAB_COUPLED_DENSE_REFINEMENT=1` | 6.6e-6 | | | | 7.3e-6 |

  The unreduced layout forms `p_F - p_B` from two nearly equal pressures once R ≫ ρc (the cavity pressure stays O(1),
  the radiated one falls as 1/R), so a ComplexF32 dense solve loses p_B, ~10× per decade above 10 ρc. **The
  diagnostics do not see it**: `pressure_continuity_error` reads 0 at 10⁶ ρc. Monolithic FP32 does the same (3.5e-4
  at 100 ρc). The felt (R ≈ 0.3 ρc) is far below it. The test asserts 1e-4 on the exterior for `:none` only up to
  10 ρc; nothing is clamped.

## 0003: contract (S3)

- `beat_contract/worker-v1.json`: `"optional_physics": ["interface_transfer_impedance"]`; `engine.version`
  `0.3.0+ti1`. Also `pyproject.toml` and `beat_engine.__version__`.
- `beat_contract/worker.py`: `negotiate_submission` refuses any request with `transfer_impedance` in a boundary's
  parameters unless the worker advertises the capability: `Incompatible BEAT worker: interface transfer-impedance
  layers are unavailable (this engine would solve without them); update BEAT Engine.` A v0.3.0 worker has no
  `optional_physics`, so it is refused. `validate_worker_ready` checks the list.
- `beat_contract/conformance.json`: case `coupled_interface_transfer_impedance_layer` (valid). The corpus checks the
  schema only; the capability refusal is tested in `tests/test_worker_negotiation.py`.

**For S4 (Boundary Lab), done 2026-10-04 (`patches/boundary-lab/`).** Pin the fork's `0.3.0+ti1`. Before submitting a layer, require
`"interface_transfer_impedance" in ready["optional_physics"]`; Boundary Lab's own client path must call
`negotiate_submission` (or copy the check) so an old engine is refused, not run. The wire object is exactly the one in
0001; R and M are per interface face area. Read back `diagnostics["interface_transfer_impedance"]` for provenance.

## Results (2026-10-04, CPU aarch64, Julia 1.12.6, 2 threads)

| Suite | v0.3.0 | + 0001 | + 0001–0003 | Wall | Peak RSS |
|---|---|---|---|---|---|
| `runtests.jl` | 1013 pass, 2 broken | 1047, 2 | **1221 pass, 4 broken, 0 fail** | 273 s | 2.9 GB |
| `reference_tests.jl` | 1642 | 1676 | **1815 pass, 2 broken, 0 fail** | 297 s | 4.8 GB |
| `pytest` | 138 | 138 | **143 pass** | 2 s | — |

The new broken entries are the CUDA and ROCm transfer-impedance arms (no GPU). The other two in `runtests.jl` are
upstream's Metal and CUDA pipeline skips. Wall and RSS are for the final tree (v0.3.0: 146 s / 1.5 GB and
377 s / 4.4 GB). `ruff check` and `ruff format --check` are clean.

## Where the plan was wrong

- **S2 file layout.** The plan names `coupled_condensed_tests.jl`, `coupled_solver_tests.jl`, `phasor_tests.jl`.
  The tests went into the one feature file S1 created; only the wire test is in `contract_tests.jl`.
- **`:pressure` with a retained transducer.** Upstream refuses it; it needs `BLAB_COUPLED_TRANSDUCER_CONDENSATION=1`.
- **V-TI2.** The plan fits M_a once and expects 3 % / 2° to 120 Hz on the coupon. The fixture's cavity correction is
  an f² law in the shunt, so Z_v0 is read per frequency and the gate is 1 % / 0.5° at ≤ 100 Hz (f/f_H ≤ 0.33).
- **V-TI3.** The plan's 1e-3 (FP64) is loose: with the consistent interface mass and a lossless interior the balance
  is a discrete identity, 1e-15.
- **FP32 risk.** The plan puts the FP32 limit at 10⁵–10⁶ ρc. On the unreduced layout the exterior pressure leaves
  1e-4 at 100 ρc. V-TI1's sweep to 10⁶ ρc (S7) must run FP64, `:pressure` or dense refinement, or its large-R points
  measure rounding.
- **Wheel release.** The plan says to release through the fork's copy of upstream's workflow. That workflow refuses
  local versions, so the wheel was built locally (see Release).

**When it goes away.** Upstream merges an equivalent. Then pin that release, delete this folder and the SHORTCUTS.md
entry.

## 0004: precompile workload

- **Why.** On the pod a coupled solve spent 38.6 s to ready and 50.8 s on the first frequency, mostly single-threaded
  JIT (the speaker project's research note `solver-profile-2026-10-05.md` (not public)). The coupled engine was `include`d from source in every worker, so no
  package image could cache it.
- **What.** The coupled driver moves to `BeatEngineCoupledDriver.jl` and is loaded inside the backend bundles
  (`julia_engine/BeatEngineCoupledWorker.jl`), with a PrecompileTools workload: one float32 frequency of the reference
  fixture on the CPU backend. `CoupledWorker.WORKLOAD[]` is serialized with the package image and says whether the
  workload solved, skipped or failed. `coupled_solver.jl` loads the bundle and falls back to source with the stderr
  line `BEAT coupled worker: compiling the engine from source (<reason>)`. `BeatEngineCudaBundle` is back in
  `julia_cuda`. A sysimage was not used: it goes stale (PackageCompiler.jl's sysimage docs).
- **Checked (aarch64, CPU).** `runtests.jl` 1237 pass / 4 broken / 0 fail (+16 in `coupled_bundle_tests.jl`),
  `reference_tests.jl` 1815/2/0, pytest 143. The CUDA bundle precompiles without a GPU and its workload solves. One
  after-run on the real cabinet: ready 15.2 → 5.6 s, first-frequency JIT about 29 → 1.5 s (shared machine, one run).
- **CPU target.** The code is now written to disk at precompile, so `JULIA_CPU_TARGET=generic` would make it
  SSE2-only on x86_64; the RunPod image uses Julia's official multi-target list
  (the Julia manual, "Environment Variables", `JULIA_CPU_TARGET`). The pod measurement is in
  `image/README.md`.

