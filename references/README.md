# references -- what the code cites, and where it lives

Look here first. Only this README and the mirror scripts are in git; a mirror is re-made by its script (gitignored).

| Source | Where | Cited by |
|---|---|---|
| Boundary Lab's docs at the pinned commit + our patches (`docs/`: `Coupled Solver.md`, `Physical System Model.md`, `advanced/Interior FEM Solver.md`, `advanced/Phasor Convention.md`, `Inputs and Outputs.md`, the BEAT contract/worker docs) and the result writer (`src/blab/headless.py`) and system model (`src/blab/physical_model.py`) | `references/boundary-lab/` from `references/mirror_boundary_lab_docs.sh` (from `~/boundary-lab`'s git objects) | boundary_lab_result (the result layout), weighted_sums (the phasor convention), room/build (interior FEM, walls, loss, probes refused), patches/boundary-lab/README.md |
| Boundary Lab's source at the pin (physical_model.py:160-167 interior FEM; solvers/coupled_backend.py:450-465 sparse direct, :687-712 wall roles; headless.py:270-274 probes refused) | `~/boundary-lab` (a clone; `git -C ~/boundary-lab show <pin>:<path>`) | room/build's docstring |
| BEAT Engine source (BeatEngineCoupled.jl:1262 P2 tetrahedra; BeatEngineCoupledDriver.jl:2370-2392 one LU per frequency) | the release wheel in the venv (`$(blabrun -c 'import beat_engine, os; print(os.path.dirname(beat_engine.__file__))')/julia_local/`) | room/build's docstring |
| E. Sengpiel, "Comparative representation of sound field quantities" (dry air at 20 °C, 101325 Pa: rho 1.204 kg/m³, c 343 m/s) | `references/sources/` from `references/fetch_sources.sh` (sha256-pinned) | room/spec's DENSITY_KG_PER_M3, SOUND_SPEED_M_PER_S |
| Kuttruff, *Room Acoustics*, ch. 3 (the rigid shoebox's modes and modal sum) | not mirrored (a book) | room/analysis modal_sum |
| CUDA.jl "Precompiling CUDA.jl without CUDA", JuliaCI build_envs.sh, Julia manual env variables, Docker build-cache docs, GitHub GHCR docs, RunPod pod API | the URLs in image/Dockerfile's comments; the speaker project that uses this mirrors them (not public), podd's `references/runpod/` | image/ |
| The research behind the patches and the image's measurements (`beat-transfer-impedance-plan-2026-10-04.md`, `solver-profile-2026-10-05.md`, `cabinet-mesh-spike-2026-10-03.md`) | the speaker project that uses this's research notes (not public) | patches/*/README.md, image/README.md |
| podd's job contract | `~/podd/src/podd/job_contract.py`, `~/podd/README.md` | solve_job |
