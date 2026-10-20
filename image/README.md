# image/ -- the solver image: Boundary Lab + BEAT Engine, prebuilt for RunPod pods

**What it is.** A Docker image with all of these installed at build time, under `/opt`, every version from
`pins.toml` (passed as build arguments; image/Dockerfile defaults none of them):
- Julia `julia_version` (x86_64);
- Boundary Lab at `boundary_lab_commit`, with `patches/boundary-lab/0001-pin-beat-engine-ti2.patch` (BEAT Engine pin
  0.2.0 → the fork release `engine`) and `0002-transfer-impedance-layer.patch` applied;
- the venv;
- the precompiled Julia depot for both BEAT backends, CPU and CUDA. Since 0.3.0+ti2 the coupled engine lives in the
  backend bundles (`BeatEngineCpuBundle`, `BeatEngineCudaBundle`) with a precompile workload, so the build caches the
  engine and its coupled call graph instead of every worker compiling them from source.

**Why.** A pod from this image is ready to solve once sshd is up. Before it, a setup script installed all of this on
every empty volume, which took about 17–20 minutes, mostly Julia precompile on a network volume. Measured
2026-10-04: `POST /pods` → ssh in 127 s, CUDA `doctor` 27 s, no install step.

| File | What |
|---|---|
| `Dockerfile` | The image, built from the REPOSITORY ROOT (`docker build -f image/Dockerfile .`; `.dockerignore` lets in only what it copies). Every non-obvious line has its source in a comment. |
| `start.sh` | The container's start command. It writes the env to `/etc/environment` for ssh sessions, starts sshd with `PUBLIC_KEY`, then runs `sleep infinity` under tini. |
| `beat-workload.sh` | Build check: fails unless a bundle's coupled precompile workload recorded "solved" (`CoupledWorker.WORKLOAD[]`). |
| `../.github/workflows/image.yml` | Run by hand (`gh workflow run image.yml`): build with the pins as build arguments and `IMAGE_COMMIT=github.sha` (into the facts file below), then push `ghcr.io/<owner>/<repo>:<sha>` and `:latest`. |
| `../patches/boundary-lab/` | The Boundary Lab patch series, the one copy (the local venv applies the same files: `venv/build-venv.sh`). |

A local build (not on this machine: it is a 30-40 min Julia precompile):

    docker build -f image/Dockerfile $(PYTHONPATH=src python3 -m acoustic_solve.pins docker-build-args | sed 's/^/--build-arg /') \
        --build-arg IMAGE_COMMIT=$(git rev-parse HEAD) .

## Layout inside the image

| Path | What |
|---|---|
| `/opt/julia-<julia_version>/` | Julia. `julia` is symlinked into `/usr/local/bin` (run-job.sh finds it there). |
| `/opt/boundary-lab/` | The clone, patched in the working tree. |
| `/opt/blab-venv/` | The venv. `blab` is symlinked into `/usr/local/bin`. |
| `/opt/julia-depot/` | `JULIA_DEPOT_PATH`. Holds the CPU and CUDA environments, the CUDA 12.8 runtime artifacts and the precompile caches. |
| `/opt/beat-workload.sh` | The build check above. |
| `/opt/beat-cpu-doctor.json` | The CPU worker's `doctor` output from build time. |
| `/etc/podd-facts.json` | THE IMAGE'S FACTS: `image_commit`, `engine` (beat_engine.__version__, asked of the venv at build), `solver_version`. podd reads it once per pod and matches every job's `requires` against it; a pod without it is refused. |

**Environment variables baked in:**
- `JULIA_CPU_TARGET='generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)'`, the
  official Julia binaries' x86_64 list. Plain `generic` would make the now-precompiled engine code SSE2-only;
- `OPENBLAS_NUM_THREADS=8` and `OMP_NUM_THREADS=8`, because inside the pod `nproc` reports the host's cores, not the
  ~9 vCPU it is allotted;
- `JULIA_PKG_OFFLINE=true`.

The network volume (if any) still mounts at `/workspace` and is only for jobs.

## CUDA without a GPU at build time

CUDA.jl picks its runtime artifact when it precompiles. Without a driver it picks none, downloads nothing, and the
build cannot run GPU code. Pinning the runtime version with `CUDA.set_runtime_version!(v"12.8")` makes that choice
independent of the driver. The next `using CUDA` then downloads and precompiles that runtime, and
`CUDA.precompile_runtime()` precompiles GPUCompiler's device library too.
(https://cuda.juliagpu.org/stable/installation/overview/, section "Precompiling CUDA.jl without CUDA".)

- **Upstream's script.** Boundary Lab ships exactly this as `docker/prepare_cuda.py` at the pinned commit. The
  Dockerfile runs that script rather than a copy of it.
- **Driver requirement.** The chosen version has to be one the pod's driver supports. CUDA 12.8 needs driver
  >= 570.26.
- **Enforced at container start.** The base image's `NVIDIA_REQUIRE_CUDA=cuda>=12.8` makes the NVIDIA container
  toolkit refuse an older host, so the failure is loud.
- **Proof in the build.** The build fails unless `libcublas.so.12*` and `libcudss.so*` are in the depot.

## Size (estimate from before the first build)

Measured on the first pod: `du -sh /opt` = 5.7 GB; the pull and extract took about 90 s.


| Layer | Compressed (pushed) | Uncompressed | Source of the number |
|---|---|---|---|
| `nvidia/cuda:12.8.2-base-ubuntu24.04` | 35 MB | ~90 MB | Docker Hub tag API |
| apt (python3, git, openssh, X libs, tini) | ~120 MB | ~350 MB | guess |
| Julia (1.12.6 then) | 290 MB | 1.1 GB | tarball length; local `du` |
| Boundary Lab clone | ~60 MB | 156 MB | local `du` |
| venv | ~350 MB | ~1.0 GB | local `du` (arm64 venv) |
| Julia depot, CPU + CUDA | ~2.2 GB | ~5 GB | CUDA_Runtime 12.8 artifact 1525 MB, CUDA_Driver 153 MB and cuDSS 109 MB (GitHub release assets); 5 GB is the measured volume depot (the old volume install, 2026-10-04) |
| **Total** | **~3 GB** | **~8 GB** | |

**The runtime base is not used.** The `-runtime-` base (2087 MB compressed) would add ~2 GB to every pull for CUDA
libraries CUDA.jl never loads.

**It fits the free runner.** `ubuntu-latest` has 14 GB nominal. Peak need is about the extracted layers plus their
compressed copies during push, ~12 GB. The free-disk step removes Android, .NET, GHC and CodeQL, which frees roughly
25–30 GB more.

## Build and publish

See `PLANS/2026-10-19-extraction-from-monitoring.md` ("Proposal") for the move from the beat-runpod mirror to this
repo's own workflow. Until it is switched on, the image pods run is `pins.toml`'s `pod_image`
(`ghcr.io/grahamseamans/beat-runpod:<sha>`), built 2026-10-14 from monitoring's `tools/runpod/image/` -- the tree this
folder was moved from, with the same build steps.

1. Commit the change here; bump `solver_version` in `pins.toml` in the same commit when it can change a number.
2. `gh workflow run image.yml` (about 30–40 min the first time; the registry cache makes a start.sh-only or facts-only
   change quick: the commit and SOLVER_VERSION are the LAST layer since 2026-10-14).
3. The first time: GHCR → the package → Package settings → visibility **Public**. Otherwise RunPod needs a registry
   credential (`containerRegistryAuthId`).
   (https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility)
4. Move `pod_image` in `pins.toml` to the new `:<sha>`. Every client takes it from there (acoustic_solve.pins).

## SOLVER_VERSION: bump it when a change changes results

`solver_version` (pins.toml) is an integer the image declares in `/etc/podd-facts.json`. **Bump it in the same commit
as any change that can change a solve's numbers**: a Boundary Lab patch, the engine pin, a Julia/CUDA runtime pin, a
solver default, the venv's packages. A client's row id hashes `engine` and `solver_version` (monitoring's
tools/runstore.py), and every JOB written by acoustic_solve.solve_job requires both, so an image whose results differ
is a new row and never overwrites an old one. A change that cannot move a number (start.sh, a comment, sshd) leaves it
alone.

## How pods run it

No template: podd's `pod.sh create <name> <image> <cards> <min vCPU>` passes `imageName` (the exact `:<sha>` a job
requires, never `:latest`), `ports` `22/tcp`, `containerDiskInGb` 30, `volumeInGb` 0, `env` {} (RunPod injects the
account's ssh keys as `PUBLIC_KEY`) and `allowedCudaVersions` (this image pins the CUDA 12.8 runtime) straight to
`POST /pods` (podd's references/runpod/runpod-rest-api-openapi-v1-2026-10-10.json, PodCreateInput); podd is its
own repo, ~/podd (README.md, Images: what any podd image must carry).

## Open questions — first build and first pod answered most of them (2026-10-04)

1. ~~does the depot need nothing at pod start?~~ **Answered 2026-10-04 (RTX 4090, driver 570.211.01):** yes. The
   CUDA `doctor` printed no download and no "Precompiling" line (27.0 s first call, 26.9 s second,
   `"available": true`), and neither did the 18-frequency coupled solve.
2. **UNVERIFIED: does `JULIA_PKG_OFFLINE=true` make a missing lazy artifact fail?** Not exercised: on the first pod
   nothing was missing.
3. ~~is the cache valid at run time?~~ **Answered 2026-10-04:** yes. The blab worker's solve recompiled nothing: no
   "Precompiling" in `solve.log`, every `.so` in `compiled/` kept its build mtime, and `/` (the container layer) grew
   by 2 MB. Julia does rewrite the mtime of the `.ji` files it loads (12 of them), which is not a recompile.
4. ~~gmsh system libraries~~ **Answered 2026-10-04:** the first build failed at `import gmsh` with
   `libGL.so.1: cannot open shared object file`; the apt list is now the full `ldd libgmsh.so` set (see the
   Dockerfile comment).
5. ~~does sshd see the env, and does `PUBLIC_KEY` arrive?~~ **Answered 2026-10-04:** yes to both. `ssh pod env`
   (non-interactive) showed `JULIA_DEPOT_PATH=/opt/julia-depot`, `JULIA_CPU_TARGET`, `JULIA_PKG_OFFLINE`,
   `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `RUNPOD_POD_ID` (29 lines in `/etc/environment`). With no
   `PUBLIC_KEY` in the template or the pod request, RunPod injected the account's 5 keys and ssh worked.
6. ~~does pod start meet ~1 minute?~~ **Answered 2026-10-04:** no, about 2. Container start (pull + extract) 93 s
   after `POST /pods`, ssh 127 s. One sample, on a host that had probably not cached the image.
7. ~~does the image count against `containerDiskInGb` (30)?~~ **Answered 2026-10-04:** no. `/` showed 30 GB size with
   16 MB used while `/opt` held 5.7 GB.
8. ~~does the GitHub runner build fit?~~ **Answered 2026-10-04:** yes, run 37218910088 built and pushed it (its
   `df -h` and duration were not read). That covered the disk peak and the time a 4-vCPU runner takes for
   the CUDA precompile (`timeout-minutes: 120`). The workflow logs `df -h` before and after.
9. **UNVERIFIED: is CUDA 12.8 the right runtime pin?** It is upstream's choice. The runtime pin and the base tag
   (12.8.2) must move together. A lower pin (e.g. 12.4) would admit older-driver hosts. A higher one would need
   `allowedCudaVersions` changed with it.
10. ~~is `JULIA_CPU_TARGET=generic` costing speed?~~ Moot since 0.3.0+ti2: the engine is now precompiled, so the
    image uses the multi-target list. Depot growth not measured.

## Measurement the next batch must make (0.3.0+ti2 image)

Before (the speaker project's research note `solver-profile-2026-10-05.md` (not public), CUDA, `stacked15core` at 100 Hz): start to "BEAT Engine ready"
38.6 s, first frequency 50.8 s, mostly single-threaded JIT. On the first solve with the ti2 image, read `solve.log`:
- **No** `BEAT coupled worker: compiling the engine from source (...)` line and no `Precompiling` line. Either one
  means the bundle cache was not used and the rest is void.
- Ready time and first-frequency time, each against 38.6 / 50.8 s. Expected: both well down; the first frequency
  keeps the CUDA host code and kernel compile, which the CPU-run workload cannot cache. The local CPU after-run gave
  ready 15.2 → 5.6 s and first-frequency JIT ~29 → ~1.5 s (one run, shared machine).
- `beat_engine.__version__` = `0.3.0+ti2` on the pod.

**Measured 2026-10-07** (A40, row `323a3748c750` = `ed5bb13bba17`'s inputs on ti2; detail in
the speaker project's research note `solver-profile-2026-10-05.md` (not public), "After precompile"): `beat_engine.__version__` `0.3.0+ti2`; no "compiling
the engine from source" and no "Precompiling" line; ready 11.4 s (was 38.6), first frequency 27.1 s (was 50.8, a
different case), wall 47.1 s against 108.6 s for the same inputs on ti1. Container start 122 s, ssh 159 s after
`POST /pods`.

## On a pod

| Path | What |
|---|---|
| `/opt/` (image) | Julia, `/opt/boundary-lab` (pinned + patched), `/opt/blab-venv`, `/opt/julia-depot`; layout above |
| `/etc/podd-facts.json` (image) | what the image declares; podd reads it once per pod |
| `/workspace/podd/podd-run.sh` (pod disk) | podd's runner, shipped once per pod |
| `/workspace/jobs/<digest>-<stage>/` (pod disk) | one job's directory while it runs |

A solve's command is `src/acoustic_solve/run-job.sh` (shipped with each job by acoustic_solve.solve_job): `blab
project solve ... --backend beat_cuda --julia-threads 8` (or `beat_cpu`), `nvidia-smi` sampled into `gpu.csv`
(`peak_gpu_mib_whole_card`: every job on the card), the RSS of its own session in `peak_rss_mib`, `DONE` or `SOLVE
FAILED` in `solve.log`, `progress` (frequencies done of the request's, every second), and a non-zero exit when the
solve failed. Given a `raw/` that came back from a stopped attempt it RESUMES: folds a stopped `raw/.resume` into
`raw/`, solves only the missing frequencies into `raw/.resume` and folds them in (one complete blab result, `resumed`
records in its manifest). The project is made locally first, by the client (monitoring's model/acoustics for a
cabinet, acoustic_solve.room.build for a room).

### Cycle, measured 2026-10-04 (one pod, RTX 4090, EU-RO-1, image `beat-runpod:latest` from Actions run
37218910088; seconds after `POST /pods`):

| Step | Time |
|---|---|
| GPU granted, pod created | immediate (first type free) |
| container started (image pulled and extracted) | 93 s |
| ssh answers | 127 s |
| `blab --help` | 2.4 s |
| `python -m beat_engine --backend cuda doctor` (`"available": true`) | 27.0 s first call, 26.9 s second; no download, no precompile |
| `stacked15core-open-d68-withsub`, coupled, 18 frequencies (`list-100-2k-18.json`) | 97.7 s wall (177.4 s on the old volume install) |

Fresh pod to solving: about 2 min (the doctor is optional). Before the image, an empty volume took 17 min of setup
and a fresh pod over a set-up volume about 4.5 min.

**Determinism.** The image's 2026-10-04 run of `stacked15core-open-d68-withsub` against `results-s0-cuda1` (the old volume
  install, also a 4090), monitoring's `tools/blab-results/determinism.py` (deleted 2026-10-12, in its git history), not bit-identical: horizontal polar 0.001 dB, vertical
  polar 0.024 dB (on an entry −50 dB re the peak), interface velocity 0.0013 dB; largest complex error per array −82
  (interface velocity) to −95 dB re its peak. S0's two same-install CUDA runs differed by 0.004 / 0.017 / 0.0005 dB,
  errors −88 to −103 dB (the speaker project's research note `beat-transfer-impedance-plan-2026-10-04.md` (not public)). Same order as CUDA's run-to-run
  noise, up to ~2.5× S0's two-run maximum; the CUDA runtime build and host differ between the two installs.

### Differences from the local CPU runs

- **Quadrature.** BEAT's CPU backend defaults to wavelength-driven regular quadrature. On our meshes that picks
  order 2 below about 10 kHz. CUDA only has fixed order 4.
  - Polars therefore differ wherever the CPU picked order 2 (4–9 kHz on `core-1.4in`, up to 27 dB in the rear
    nulls).
  - With `"solver_options": {"regular_quadrature_mode": "fixed"}` in the request, the CPU matches CUDA to
    0.003 dB.
- **Overhead.** On the old volume install every run paid about 60 s of Julia start-up and CUDA JIT, whatever the
  mesh. On the image the CUDA `doctor` (Julia start + CUDA init) takes 27 s.
- **Threads.** The container shows 96 CPUs but RunPod allots 9. Without `OPENBLAS_NUM_THREADS=8` (image ENV)
  the coupled FEM/Schur step stalled: more than 22 minutes on one frequency instead of about 1 minute.
- **Cabinet wall times** (2026-10-04, x-half, BEAT CUDA):

  | Run | Wall time | Peak GPU memory |
  |---|---|---|
  | `stacked15core-closed-d68-withsub`, coupled, 3 frequencies | 108 s | 6.2 GB |
  | `stacked15core-closed-d68-withsub`, exterior only, 18 frequencies | 99 s | 7.4 GB |
  | `stacked15core-open-d68-withsub`, coupled, 18 frequencies | 154 s | 4.0 GB |
  | `stacked15core-open-d150-withsub`, coupled, 18 frequencies | 162 s | 3.7 GB |
  | `core-1.4in-box15-sharp`, 18 frequencies | 131 s | 15.5 GB |

  The 0.7h coupled closed run never finished: killed after 30 minutes on 100 Hz (monitoring's `SHORTCUTS.md`).
- **GPU memory.** One frequency needs about N² × 8 B (a complex float32 dense matrix, N = P1 nodes of the solved
  half).
  - The A40 holds up to about 70k nodes.
  - CUDA.jl's pool keeps each finished frequency's matrix until it runs short, so `nvidia-smi` climbs to about
    44 GB and then drops back. This is not a leak: a 4-frequency 43.6k-node run reclaimed and finished.
