"""src/acoustic_solve/run-job.sh, the pod-side command of a solve (a podd job's command): it resumes a blab result per
frequency (the room chunks' podd `resumable`), against a fake blab that writes what blab's HeadlessResultWriter writes
(~/boundary-lab/src/blab/headless.py) and can stop, and a fake julia (the image's is on PATH). No pod, no blab, no
julia; and with no julia on PATH it refuses loudly. Moved from monitoring's tools/tests on 2026-10-19.

    blabrun tests/test_run_job.py       # numpy: the fake blab writes real npz files
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RUN_JOB_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "acoustic_solve" / "run-job.sh"

FAKE_BLAB = r'''#!/usr/bin/env python3
import json, os, sys, time
import numpy as np
arguments = sys.argv[1:]
julia = arguments[arguments.index("--julia-executable") + 1]
if not os.access(julia, os.X_OK):
    sys.exit(f"fake blab: --julia-executable {julia!r} is not an executable")
request = json.load(open(arguments[arguments.index("--request") + 1]))
output = arguments[arguments.index("--output") + 1]
stop_after = int(os.environ.get("FAKE_BLAB_STOP_AFTER", "1000000"))
seconds_per_frequency = float(os.environ.get("FAKE_BLAB_SECONDS_PER_FREQUENCY", "0"))
os.makedirs(output + "/frequencies")
# one ten-node tetrahedron, as blab writes a fem_volume domain (headless.py _write_domains)
np.savez_compressed(output + "/domains.npz", d0000_c_0000=np.arange(30.0).reshape(10, 3),
                    d0000_t_0000=np.arange(10).reshape(1, 10))
json.dump({"domains": [{"id": "fem:room", "kind": "fem_volume", "coordinates": {"points_m": "d0000_c_0000"},
                        "topology": {"tetrahedra10": "d0000_t_0000"}, "metadata": {"node_counts": [10]}}]},
          open(output + "/domains.json", "w"))
open(output + "/request.json", "w").write(json.dumps(request))
frequencies = request["frequencies_hz"]
manifest = {"status": "running", "project_sha256": "p" * 64, "frequencies_hz": frequencies, "engine_runs": [
    {"run": len(frequencies)}], "completion_mask": [False] * len(frequencies), "results": [None] * len(frequencies),
    "started_at_utc": "now", "domains_file": "domains.npz", "domains_metadata_file": "domains.json"}
def flush():
    json.dump(manifest, open(output + "/.manifest.json.tmp", "w"))
    os.replace(output + "/.manifest.json.tmp", output + "/manifest.json")
flush()
for index, frequency in enumerate(frequencies):
    if index == stop_after:
        sys.exit(9)
    time.sleep(seconds_per_frequency)
    stem = f"{index:06d}"
    np.savez_compressed(f"{output}/frequencies/{stem}.npz", q0000=np.full((1, 10), frequency))
    json.dump({"freq_hz": frequency, "arrays_file": f"{stem}.npz", "quantities": [
        {"key": "q0000", "target_id": "fem:room", "axes": ["excitation", "fem_node"], "shape": [1, 10],
         "metadata": {"node_counts": [10]}}]}, open(f"{output}/frequencies/{stem}.json", "w"))
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


COLLECTOR_SCRIPT = RUN_JOB_SCRIPT.with_name("collect_result.py")
OUTPUTS = ("raw", "solve.log", "gpu.csv")
RESUME_FROM = ("raw/manifest.json", "solve.log")          # acoustic_solve.solve_job's, as podd ships them


def a_home_stage(with_julia=True):
    """(the fakes' root, the stage at home): the request, run-job.sh and its collector, as solve_job stages them."""
    root = Path(tempfile.mkdtemp(prefix="run-job-resume-"))
    (root / "bin").mkdir()
    # its shebang is the interpreter running these tests: run-job.sh runs the collector with blab's own Python (under
    # blabrun, the venv's, with numpy for the reduction)
    (root / "bin" / "blab").write_text(f"#!{sys.executable}\n" + FAKE_BLAB.split("\n", 1)[1])
    (root / "bin" / "blab").chmod(0o755)
    if with_julia:
        (root / "bin" / "julia").write_text("#!/bin/sh\nexit 0\n")
        (root / "bin" / "julia").chmod(0o755)
    home = root / "home"
    (home / "build").mkdir(parents=True)
    (home / "request.json").write_text(json.dumps({"schema_version": 1, "frequencies_hz": [40.0, 41.0, 42.0, 43.0, 44.0],
                                                   "retain": ["fem_nodal_pressure"]}))
    for script in (RUN_JOB_SCRIPT, COLLECTOR_SCRIPT):
        shutil.copy(script, home / script.name)
    return root, home


def podd_attempt(root, home, stop_after, seconds_per_frequency=0):
    """One attempt as podd runs it (podd.daemon run_the_job): a fresh pod directory gets the inputs and the
    resume_from files present at home, NEVER the rest of the outputs; then every output that exists is laid over
    home's copy (no deletes). -> (the run, what the pod directory held before the command ran)."""
    pod = Path(tempfile.mkdtemp(prefix="run-job-pod-", dir=root))
    for path in ("build", "run-job.sh", "collect_result.py", "request.json") + RESUME_FROM:
        if (home / path).is_dir():
            shutil.copytree(home / path, pod / path)
        elif (home / path).is_file():
            (pod / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(home / path, pod / path)
    shipped = sorted(str(f.relative_to(pod)) for f in pod.rglob("*") if f.is_file())
    # PATH: the fakes, and the system's own tools, never a julia this machine might have in ~/.local/bin
    environment = {**os.environ, "PATH": f"{root / 'bin'}:/usr/bin:/bin", "FAKE_BLAB_STOP_AFTER": str(stop_after),
                   "FAKE_BLAB_SECONDS_PER_FREQUENCY": str(seconds_per_frequency)}
    run = subprocess.run(["bash", "run-job.sh", "build/room.blab.json", "request.json", "beat_cpu", "1"],
                         cwd=pod, env=environment, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    for output in OUTPUTS:
        if (pod / output).is_dir():
            shutil.copytree(pod / output, home / output, dirs_exist_ok=True)
        elif (pod / output).is_file():
            shutil.copy(pod / output, home / output)
    run.progress = (pod / "progress").read_text().split() if (pod / "progress").exists() else None
    return run, shipped


def test_run_job_sh_without_julia_on_path_refuses_and_says_so():
    root, home = a_home_stage(with_julia=False)
    refused, _ = podd_attempt(root, home, 1000)
    assert refused.returncode != 0 and "no julia on PATH" in (home / "solve.log").read_text(), refused
    assert not (home / "raw").exists()


def test_a_resumed_attempt_gets_only_the_manifest_solves_only_the_rest_and_home_ends_with_one_whole_result():
    """2026-10-21: the raw/ a stopped attempt brought home went back up to the next pod (0.2-2.2 GB a job, ~1 MB/s)
    and timed out. Now the pod gets raw/manifest.json and solve.log only, and the merge is podd's at home."""
    root, home = a_home_stage()
    first, shipped = podd_attempt(root, home, 2)                  # stopped after 2 of 5 frequencies
    assert first.returncode != 0 and first.progress == ["2", "5"], (home / "solve.log").read_text()
    assert not any(path.startswith("raw/") for path in shipped), shipped
    assert sorted(p.name for p in (home / "raw" / "frequencies").iterdir()) == [
        "000000.json", "000000.npz", "000001.json", "000001.npz"], "every file that came home is final"
    second, shipped = podd_attempt(root, home, 1)                 # the resumed attempt is stopped too, after 1 more
    assert second.returncode != 0
    assert [p for p in shipped if p.startswith("raw/")] == ["raw/manifest.json"], shipped
    manifest = json.loads((home / "raw" / "manifest.json").read_text())
    assert manifest["completion_mask"] == [True, True, True, False, False] and manifest["status"] == "interrupted", \
        manifest
    assert second.progress == ["3", "5"], "progress counts the kept frequencies too"
    third, shipped = podd_attempt(root, home, 1000, seconds_per_frequency=1.5)   # collected while blab runs
    assert [p for p in shipped if p.startswith("raw/")] == ["raw/manifest.json"], shipped
    log = (home / "solve.log").read_text()
    assert third.returncode == 0 and "RESUMED: 2 frequencies of request.json left" in log, log
    manifest = json.loads((home / "raw" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and all(manifest["completion_mask"]), manifest
    for index, frequency in enumerate([40.0, 41.0, 42.0, 43.0, 44.0]):
        result = manifest["results"][index]
        assert result["arrays_file"] == f"frequencies/{index:06d}.npz", result
        assert (np.load(home / "raw" / result["arrays_file"])["q0000"] == frequency).all(), result
        assert json.loads((home / "raw" / result["metadata_file"]).read_text())["arrays_file"] == f"{index:06d}.npz"
    assert len(manifest["resumed"]) == 2 and [r["frequencies_folded"] for r in manifest["resumed"]] == [1, 2], \
        manifest["resumed"]
    assert sorted(p.name for p in (home / "raw").iterdir()) == ["domains.json", "domains.npz", "frequencies",
                                                                "manifest.json", "request.json"], \
        "the result and nothing else: blab's own directory (.solving) never comes home"
    assert json.loads((home / "raw" / "request.json").read_text())["frequencies_hz"] == manifest["frequencies_hz"], \
        "raw/'s files are the first attempt's (the whole request), never a resumed attempt's partial request"
    assert third.progress == ["5", "5"] and log.count("DONE") == 1 and log.count("RESUMED") == 2


def test_a_project_that_keeps_nodes_has_each_frequency_reduced_and_a_keep_file_it_cannot_use_fails_the_job():
    """The glue (the reduction on a real room mesh is tests/test_room_field.py's): with build/keep_nodes.npy every
    frequency comes home reduced (here 9 of the tetrahedron's 10 nodes: no whole cell is kept), resumed or not, with
    blab's own Python (its shebang); a keep file the collector cannot use stops the solve and fails the job loudly."""
    root, home = a_home_stage()
    np.save(home / "build" / "keep_nodes.npy", np.arange(30.0).reshape(10, 3)[1:])
    podd_attempt(root, home, 2)
    resumed, _ = podd_attempt(root, home, 1000)
    manifest = json.loads((home / "raw" / "manifest.json").read_text())
    assert resumed.returncode == 0 and manifest["status"] == "complete", (home / "solve.log").read_text()
    assert manifest["reduced_to_kept_nodes"] == {"kept_nodes": 9, "of_nodes": 10, "kept_tetrahedra": 0,
                                                 "of_tetrahedra": 1}, manifest
    for index in range(5):
        values = np.load(home / "raw" / "frequencies" / f"{index:06d}.npz")["q0000"]
        assert values.shape == (1, 9) and (values == 40.0 + index).all(), (index, values)
    domains = np.load(home / "raw" / "domains.npz")
    assert domains["d0000_c_0000"].shape == (9, 3) and domains["d0000_t_0000"].shape == (0, 10)
    root, home = a_home_stage()
    (home / "build" / "keep_nodes.npy").write_bytes(b"not coordinates")
    failed, _ = podd_attempt(root, home, 1000)
    log = (home / "solve.log").read_text()
    assert failed.returncode != 0 and "frequencies into raw/ failed" in log and "pickled" in log, log
    assert "SOLVE FAILED" in log and "DONE" not in log, log


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
