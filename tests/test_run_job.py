"""src/acoustic_solve/run-job.sh, the pod-side command of a solve (a podd job's command): it resumes a blab result per
frequency (the room chunks' podd `resumable`), against a fake blab that writes what blab's HeadlessResultWriter writes
(~/boundary-lab/src/blab/headless.py) and can stop, and a fake julia (the image's is on PATH). No pod, no blab, no
julia; and with no julia on PATH it refuses loudly. Moved from monitoring's tools/tests on 2026-10-19.

    python3 tests/test_run_job.py
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

RUN_JOB_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "acoustic_solve" / "run-job.sh"

FAKE_BLAB = r'''#!/usr/bin/env python3
import json, os, sys
arguments = sys.argv[1:]
julia = arguments[arguments.index("--julia-executable") + 1]
if not os.access(julia, os.X_OK):
    sys.exit(f"fake blab: --julia-executable {julia!r} is not an executable")
request = json.load(open(arguments[arguments.index("--request") + 1]))
output = arguments[arguments.index("--output") + 1]
stop_after = int(os.environ.get("FAKE_BLAB_STOP_AFTER", "1000000"))
os.makedirs(output + "/frequencies")
frequencies = request["frequencies_hz"]
manifest = {"status": "running", "project_sha256": "p" * 64, "frequencies_hz": frequencies, "engine_runs": [
    {"run": len(frequencies)}], "completion_mask": [False] * len(frequencies), "results": [None] * len(frequencies),
    "started_at_utc": "now"}
def flush():
    json.dump(manifest, open(output + "/.manifest.json.tmp", "w"))
    os.replace(output + "/.manifest.json.tmp", output + "/manifest.json")
flush()
for index, frequency in enumerate(frequencies):
    if index == stop_after:
        sys.exit(9)
    stem = f"{index:06d}"
    open(f"{output}/frequencies/{stem}.npz", "w").write(f"pressure at {frequency} Hz")
    json.dump({"freq_hz": frequency, "arrays_file": f"{stem}.npz"}, open(f"{output}/frequencies/{stem}.json", "w"))
    manifest["completion_mask"][index] = True
    manifest["results"][index] = {"freq_hz": frequency, "metadata_file": f"frequencies/{stem}.json",
                                  "arrays_file": f"frequencies/{stem}.npz"}
    flush()
manifest["status"] = "complete"
flush()
'''


def test_run_job_sh_is_one_solve_and_manages_no_lifetime():
    """Its lifetime (session, timeout, kill) is podd's; a failed solve exits non-zero, so podd fails the job. (Moved
    from monitoring's test_run_guards.py, 2026-10-19.)"""
    code = "\n".join(line for line in RUN_JOB_SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#"))
    for lifetime_word in ("TIMEOUT", "pkill", "timeout", "setsid"):
        assert lifetime_word not in code, f"run-job.sh manages a lifetime again: {lifetime_word}"
    assert "exit $status" in code, "a failed solve exits non-zero, so podd fails the job"


def a_job_directory(with_julia=True):
    root = Path(tempfile.mkdtemp(prefix="run-job-resume-"))
    (root / "bin").mkdir()
    (root / "bin" / "blab").write_text(FAKE_BLAB)
    (root / "bin" / "blab").chmod(0o755)
    if with_julia:
        (root / "bin" / "julia").write_text("#!/bin/sh\nexit 0\n")
        (root / "bin" / "julia").chmod(0o755)
    job = root / "job"
    job.mkdir()
    (job / "request.json").write_text(json.dumps({"schema_version": 1, "frequencies_hz": [40.0, 41.0, 42.0, 43.0, 44.0],
                                                  "retain": ["fem_nodal_pressure"]}))
    return root, job


def run_job(root, job, stop_after):
    # PATH: the fakes, and the system's own tools, never a julia this machine might have in ~/.local/bin
    environment = {**os.environ, "PATH": f"{root / 'bin'}:/usr/bin:/bin", "FAKE_BLAB_STOP_AFTER": str(stop_after)}
    return subprocess.run(["bash", str(RUN_JOB_SCRIPT), "build/room.blab.json", "request.json", "beat_cpu", "1"],
                          cwd=job, env=environment, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def test_run_job_sh_without_julia_on_path_refuses_and_says_so():
    root, job = a_job_directory(with_julia=False)
    refused = run_job(root, job, 1000)
    assert refused.returncode != 0 and "no julia on PATH" in (job / "solve.log").read_text(), refused
    assert not (job / "raw").exists()


def test_run_job_sh_resumes_only_the_frequencies_that_did_not_come_back():
    root, job = a_job_directory()

    def attempt(stop_after):
        return run_job(root, job, stop_after)
    first = attempt(2)                                       # stopped after 2 of 5 frequencies
    assert first.returncode != 0 and (job / "progress").read_text().split() == ["2", "5"], (job / "solve.log").read_text()
    second = attempt(1)                                      # the resumed attempt is stopped too, after 1 more
    assert second.returncode != 0
    manifest = json.loads((job / "raw" / "manifest.json").read_text())
    assert manifest["completion_mask"] == [True, True, True, False, False] and manifest["status"] == "interrupted", \
        manifest
    assert (job / "progress").read_text().split() == ["3", "5"]
    third = attempt(1000)
    log = (job / "solve.log").read_text()
    assert third.returncode == 0 and "RESUMED: 2 frequencies of request.json left" in log, log
    manifest = json.loads((job / "raw" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and all(manifest["completion_mask"]), manifest
    for index, frequency in enumerate([40.0, 41.0, 42.0, 43.0, 44.0]):
        result = manifest["results"][index]
        assert result["arrays_file"] == f"frequencies/{index:06d}.npz", result
        assert (job / "raw" / result["arrays_file"]).read_text() == f"pressure at {frequency} Hz"
        assert json.loads((job / "raw" / result["metadata_file"]).read_text())["arrays_file"] == f"{index:06d}.npz"
    assert not (job / "raw" / ".resume").exists() and len(manifest["resumed"]) == 2, manifest["resumed"]
    assert (job / "progress").read_text().split() == ["5", "5"] and log.count("DONE") == 1


def test_run_job_sh_reduces_raw_with_blabs_python_when_the_project_keeps_nodes_and_fails_loudly_when_that_fails():
    """The glue only (acoustic_solve.reduce_nodes is tested on a real mesh in tests/test_room_field.py): a project
    directory with keep_nodes.npy has its reducer run on raw/ after a complete solve, by the interpreter of blab's
    shebang; a reducer that fails fails the job and says so."""
    for reducer, expect_status in (("import sys; open(sys.argv[1] + '/reduced-by', 'w').write(sys.argv[2])", 0),
                                   ("import sys; sys.exit('broken reducer')", 1)):
        root, job = a_job_directory()
        (job / "build").mkdir()
        (job / "build" / "keep_nodes.npy").write_bytes(b"coordinates")
        (job / "build" / "reduce_nodes.py").write_text(reducer)
        done = run_job(root, job, 1000)
        log = (job / "solve.log").read_text()
        if expect_status == 0:
            assert done.returncode == 0 and (job / "raw" / "reduced-by").read_text() == "build/keep_nodes.npy", log
            assert "DONE" in log
        else:
            assert done.returncode != 0 and "reducing raw/ to the kept nodes failed" in log and "DONE" not in log, log


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
