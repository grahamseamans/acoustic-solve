"""ONE BLAB SOLVE AS A PODD JOB: the one writer of a solve's JOB, for every client (monitoring's rows, the room).

    from acoustic_solve import solve_job
    solve_job.queue_solve(stage, "build/room.blab.json", "request.json", "beat_cpu",
                          resources={"cpu": 4, "gpu_memory_gb": 0, "ram_gb": 6}, expected_s=600,
                          label="room medium-live", threads=4, resumable=True, requires={"gpu": "none"})

The stage is a directory in one of podd's queue directories (podd.job_contract: JOB -> RUNNING -> DONE | FAILED); the
staged project (its directory, e.g. build/) and the request are already in it. queue_solve copies run-job.sh in and
writes JOB through podd's own writer, which checks the contract:

  command   bash run-job.sh <project> <request> <backend> <threads>  (run-job.sh: the solve, solve.log, gpu.csv,
            progress, the resume of a raw/ that came back)
  inputs    the project's top directory, run-job.sh, the request
  outputs   raw, solve.log, gpu.csv
  requires  image = pins.POD_IMAGE, engine = pins.ENGINE, solver_version = pins.SOLVER_VERSION -- always the pins, so a
            row id that hashes the engine and solver version is pinned before its solve -- plus the client's own
            (`gpu_memory_gb_min` for a card, `gpu: "none"` for a CPU pod)
  resumable a job that declares it also declares `progress` (run-job.sh writes it either way): podd requeues a stopped
            one with what came back, and run-job.sh solves only the missing frequencies.

The client owns what it measured about its own solves: resources (cpu, gpu_memory_gb, ram_gb), expected_s, label.
"""
import shutil
from pathlib import Path

from podd import job_contract

from acoustic_solve import pins

RUN_JOB_SCRIPT = Path(__file__).resolve().with_name("run-job.sh")
BACKENDS = ("beat_cuda", "beat_cpu")
SOLVE_OUTPUTS = ("raw", "solve.log", "gpu.csv")
PROGRESS_FILE = "progress"
DEFAULT_THREADS = 8                # the image's OPENBLAS_NUM_THREADS: a job that has a whole pod's cores
RESOURCE_KEYS = {"cpu", "gpu_memory_gb", "ram_gb"}
PINNED_REQUIREMENTS = ("image", "engine", "solver_version")


class SolveJobRefused(RuntimeError):
    """A solve that cannot be queued as asked (named in the message)."""


def pinned_requirements():
    return {"image": pins.POD_IMAGE, "engine": pins.ENGINE, "solver_version": pins.SOLVER_VERSION}


def queue_solve(stage, project_file, request_file, backend, resources, expected_s, label, threads=DEFAULT_THREADS,
                resumable=False, requires=None):
    """Copy run-job.sh into the stage and write its JOB (module docstring). Returns the JOB written."""
    stage = Path(stage)
    if backend not in BACKENDS:
        raise SolveJobRefused(f"(╯°□°)╯ backend {backend!r} is not one of {BACKENDS}")
    if set(resources) != RESOURCE_KEYS:
        raise SolveJobRefused(f"(╯°□°)╯ resources {sorted(resources)}: exactly {sorted(RESOURCE_KEYS)}")
    clashing = sorted(set(requires or {}) & set(PINNED_REQUIREMENTS))
    if clashing:
        raise SolveJobRefused(f"(╯°□°)╯ requires {clashing} are the pins' (pins.toml), never a client's")
    for staged in (project_file, request_file):
        if not (stage / staged).is_file():
            raise SolveJobRefused(f"(╯°□°)╯ {stage / staged} is not staged")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise SolveJobRefused(f"(╯°□°)╯ threads {threads!r} is not a positive integer")
    shutil.copy(RUN_JOB_SCRIPT, stage / RUN_JOB_SCRIPT.name)
    job = {"command": f"bash {RUN_JOB_SCRIPT.name} {project_file} {request_file} {backend} {threads}",
           "inputs": [Path(project_file).parts[0], RUN_JOB_SCRIPT.name, request_file],
           "outputs": list(SOLVE_OUTPUTS),
           "requires": {**pinned_requirements(), **(requires or {})},
           **resources, "expected_s": expected_s, "label": label}
    if resumable:
        job.update({"progress": PROGRESS_FILE, "resumable": True})
    job_contract.write_job(stage, job)
    return job
