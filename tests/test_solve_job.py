"""acoustic_solve.solve_job -- the one writer of a blab solve's JOB: run-job.sh shipped into the stage, the command,
the pins as `requires` (never a client's), podd's own writer checking the contract, `progress` with `resumable`; and
the refusals. No pod, no podd running (podd.job_contract writes the file). In the blab venv (podd installed):

    blabrun tests/test_solve_job.py
"""
import json
import tempfile
import time
from pathlib import Path

from podd import job_contract

from acoustic_solve import pins, solve_job


def a_stage():
    stage = Path(tempfile.mkdtemp(prefix="solve-job-"))
    (stage / "build").mkdir()
    (stage / "build" / "room.blab.json").write_text("{}")
    (stage / "request.json").write_text(json.dumps({"frequencies_hz": [40.0]}))
    return stage


RESOURCES = {"cpu": 4, "gpu_memory_gb": 0, "ram_gb": 6}


def test_a_resumable_cpu_solve_ships_run_job_and_the_collector_resumes_from_the_manifest_and_requires_the_pins():
    stage = a_stage()
    job = solve_job.queue_solve(stage, "build/room.blab.json", "request.json", "beat_cpu", RESOURCES, 600, "a room",
                                threads=4, resumable=True, requires={"gpu": "none"})
    written = json.loads((stage / job_contract.JOB).read_text())
    assert job_contract.stage_state(stage) == job_contract.JOB
    assert written["command"] == "bash run-job.sh build/room.blab.json request.json beat_cpu 4"
    assert written["inputs"] == ["build", "run-job.sh", "collect_result.py", "request.json"]
    assert written["outputs"] == ["raw", "solve.log", "gpu.csv"]
    assert written["requires"] == {"image": pins.POD_IMAGE, "engine": pins.ENGINE,
                                   "solver_version": pins.SOLVER_VERSION, "gpu": "none"}
    assert written["progress"] == "progress" and written["resumable"] is True
    assert written["resume_from"] == ["raw/manifest.json", "solve.log"], "the finished frequencies stay home"
    assert (stage / "run-job.sh").read_bytes() == solve_job.RUN_JOB_SCRIPT.read_bytes()
    assert (stage / "collect_result.py").read_bytes() == solve_job.COLLECTOR_SCRIPT.read_bytes()
    assert {k: written[k] for k in RESOURCES} == RESOURCES and written["expected_s"] == 600


def test_a_card_solve_that_is_not_resumable_declares_no_progress():
    stage = a_stage()
    solve_job.queue_solve(stage, "build/room.blab.json", "request.json", "beat_cuda",
                          {"cpu": 2.5, "gpu_memory_gb": 1, "ram_gb": 2.5}, 55, "a row", requires={"gpu_memory_gb_min": 20})
    written = json.loads((stage / job_contract.JOB).read_text())
    assert "progress" not in written and "resumable" not in written and "resume_from" not in written
    assert written["command"].endswith(" beat_cuda 8") and written["requires"]["gpu_memory_gb_min"] == 20


def test_what_cannot_be_queued_is_refused_before_anything_is_written():
    for changes, words in (({"backend": "beat_metal"}, "backend"),
                           ({"resources": {"cpu": 1}}, "resources"),
                           ({"requires": {"engine": "0"}}, "the pins'"),
                           ({"project_file": "build/none.blab.json"}, "not staged"),
                           ({"threads": 0}, "threads")):
        stage = a_stage()
        arguments = {"stage": stage, "project_file": "build/room.blab.json", "request_file": "request.json",
                     "backend": "beat_cpu", "resources": RESOURCES, "expected_s": 60, "label": "x", **changes}
        try:
            solve_job.queue_solve(**arguments)
        except solve_job.SolveJobRefused as error:
            assert words in str(error), (words, error)
        else:
            raise AssertionError(f"queued ({changes})")
        assert not (stage / job_contract.JOB).exists() and not (stage / "run-job.sh").exists()


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
