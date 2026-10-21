#!/bin/bash
# ON A POD, as a podd job's command (acoustic_solve.solve_job writes the JOB and ships this file with it; podd runs it
# in the job's directory, in a session of its own, and owns its timeout and kill: podd.job_contract, ~/podd):
#
#   bash run-job.sh <project.blab.json> <request.json> <beat_cuda|beat_cpu> [threads]
#
# [threads]: Julia's and OpenBLAS's thread count (default 8, the image's OPENBLAS_NUM_THREADS); a job that declares
# fewer cores than a whole pod (a room's interior FEM on a CPU pod: acoustic_solve.room.build) passes its own `cpu`.
# ONE SOLVE of a staged project (any client's: a cabinet's build/cabinet.blab.json, a horn's build/core.blab.json +
# build/runs/, a room's build/room.blab.json) on BEAT CUDA or the pod's CPU into ./raw. The image (image/) puts julia
# and blab on PATH (/usr/local/bin) and its env (depot, CPU target, BLAS thread cap) into every ssh session, so nothing
# is sourced here. Exits non-zero when the solve fails (podd then FAILS the job, with what came back).
#
# Writes, in the job's directory:
#   solve.log  blab's progress, then wall_s, cpu_s (the largest CPU-seconds sum over this session's processes,
#              sampled every second: cpu_s / wall_s is the cores it kept busy), peak_gpu_mib_whole_card, peak_rss_mib, then DONE (or SOLVE FAILED first)
#              (a resumed attempt appends to the solve.log that came back)
#   gpu.csv    nvidia-smi every 0.5 s; on a pod with no card (a CPU pod) one line saying so
#   progress   `<frequencies done> <frequencies of the request>`, every second (podd's `progress`: podd.job_contract)
#
# RESUMABLE (podd's `resumable`; acoustic_solve.room.build's jobs declare it): when raw/manifest.json is already here (podd
# shipped back the raw/ an earlier, stopped attempt brought home; blab writes every frequency to its own file and the
# manifest after it), this attempt first folds raw/.resume (a stopped resumed attempt's) into raw/, then solves ONLY
# the frequencies raw/ lacks into raw/.resume (blab refuses an existing output directory) and folds that in: raw/ ends
# as one complete blab result, status "complete", with a `resumed` record per fold.
# SEVERAL JOBS SHARE A POD: podd starts each in its own session, so the RSS summed over this session is this job's
# (its python and julia workers); peak_gpu_mib is the whole card's (nvidia-smi sees every job on it), as its name says.
set -eu
threads=${4:-8}
julia=$(command -v julia) || { echo "run-job.sh (╯°□°)╯ no julia on PATH: not the solver image" >> solve.log; exit 1; }
export OPENBLAS_NUM_THREADS=$threads
smi=
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 500 > gpu.csv &
  smi=$!
else
  echo "no nvidia-smi on this pod (a CPU pod): no card to sample" > gpu.csv
fi
session=$(ps -o sid= -p $$ | tr -d ' ')

# fold: raw/.resume's finished frequencies into raw/ (renumbered to raw's frequency list), its engine runs, a `resumed`
# record; raw/.resume goes; raw/'s status is "complete" when every frequency is in, else "interrupted". Then print how
# many frequencies raw/ still lacks, and write them as .resume-request.json (the request with only those).
fold() {
  python3 - "$1" <<'FOLD'
import json, math, os, shutil, sys
from datetime import datetime, timezone
request_path = sys.argv[1]
def read(path):
    with open(path) as stream:
        return json.load(stream)
def write(path, value):
    with open(path + ".tmp", "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)
manifest = read("raw/manifest.json")
part = "raw/.resume"
if os.path.isfile(part + "/manifest.json"):
    resumed = read(part + "/manifest.json")
    if resumed["project_sha256"] != manifest["project_sha256"]:
        sys.exit(f"run-job.sh fold (╯°□°)╯ raw/.resume solved another project ({resumed['project_sha256']} against "
                 f"{manifest['project_sha256']}): refusing to fold it")
    folded = 0
    for done, result in zip(resumed["completion_mask"], resumed["results"]):
        if not done or result is None:
            continue
        frequency = result["freq_hz"]
        distances = [abs(f - frequency) for f in manifest["frequencies_hz"]]
        index = distances.index(min(distances))
        if distances[index] > max(1e-5, abs(frequency) * 2e-6):            # blab's own tolerance (headless.py)
            sys.exit(f"run-job.sh fold (╯°□°)╯ raw/.resume solved {frequency} Hz, which raw/ never asked for")
        stem = f"{index:06d}"
        os.replace(os.path.join(part, result["arrays_file"]), f"raw/frequencies/{stem}.npz")
        metadata = read(os.path.join(part, result["metadata_file"]))
        metadata["arrays_file"] = f"{stem}.npz"
        write(f"raw/frequencies/{stem}.json", metadata)
        manifest["completion_mask"][index] = True
        manifest["results"][index] = {"freq_hz": frequency, "metadata_file": f"frequencies/{stem}.json",
                                      "arrays_file": f"frequencies/{stem}.npz"}
        folded += 1
    for run in resumed.get("engine_runs") or []:
        if run not in manifest["engine_runs"]:
            manifest["engine_runs"].append(run)
    manifest.setdefault("resumed", []).append({"started_at_utc": resumed.get("started_at_utc"),
                                               "status": resumed.get("status"), "frequencies_folded": folded})
    print(f"run-job.sh: folded {folded} frequencies of raw/.resume (status {resumed.get('status')}) into raw/",
          file=sys.stderr)
    shutil.rmtree(part)
elif os.path.isdir(part):
    print("run-job.sh: (・_・;) raw/.resume has no manifest (stopped before blab wrote one): removed", file=sys.stderr)
    shutil.rmtree(part)
complete = all(manifest["completion_mask"])
manifest["status"] = "complete" if complete else "interrupted"
if complete:
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
write("raw/manifest.json", manifest)
request = read(request_path)
wanted = request["frequencies_hz"]
have = [f for f, done in zip(manifest["frequencies_hz"], manifest["completion_mask"]) if done]
left = [f for f in wanted if not any(math.isclose(f, h, rel_tol=2e-6, abs_tol=1e-5) for h in have)]
write(".resume-request.json", {**request, "frequencies_hz": left})
print(len(left))
FOLD
}

request=$2
output=raw
if [ -d raw ] && [ ! -f raw/manifest.json ]; then
  echo "run-job.sh: (・_・;) raw/ came back without a manifest (stopped before blab wrote one): solving from scratch" >> solve.log
  rm -rf raw
fi
if [ -f raw/manifest.json ]; then
  left=$(fold "$2" 2>> solve.log)
  echo "run-job.sh: RESUMED: $left frequencies of $2 left to solve" >> solve.log
  request=.resume-request.json
  output=raw/.resume
fi
total=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["frequencies_hz"]))' "$2")
done_now() { find raw -path '*/frequencies/*.json' -not -name '.*' 2>/dev/null | wc -l; }
start=$(date +%s.%N)
solve=
if [ "$output" = raw/.resume ] && [ "$left" -eq 0 ]; then
  echo "run-job.sh: every frequency came back: nothing to solve" >> solve.log
else
  blab project solve "$1" --request "$request" --backend "$3" --output "$output" \
      --julia-executable "$julia" --julia-threads "$threads" >> solve.log 2>&1 &
  solve=$!
fi
peak_rss_kib=0
peak_cpu_s=0
# every second: the RSS and the cumulative CPU seconds (ps `times`) summed over this session; julia lives through the
# whole solve, so the largest CPU sum is the solve's (2026-10-15: bash's `times` missed julia, which blab never reaps)
while [ -n "$solve" ] && kill -0 $solve 2>/dev/null; do
  read -r rss cpu < <(ps -eo sid=,rss=,times= | awk -v session=$session '$1 == session {r += $2; c += $3} END {print r + 0, c + 0}')
  [ $rss -gt $peak_rss_kib ] && peak_rss_kib=$rss
  [ $cpu -gt $peak_cpu_s ] && peak_cpu_s=$cpu
  echo "$(done_now) $total" > progress.writing && mv progress.writing progress
  sleep 1
done
status=0
[ -z "$solve" ] || wait $solve || status=$?
if [ "$output" = raw/.resume ]; then
  fold "$2" >/dev/null 2>> solve.log || { echo "run-job.sh (╯°□°)╯ folding raw/.resume into raw/ failed" >> solve.log; [ $status -ne 0 ] || status=1; }
fi
echo "$(done_now) $total" > progress.writing && mv progress.writing progress
[ $status -eq 0 ] || echo "SOLVE FAILED (blab exited $status)" >> solve.log
# REDUCED RESULTS: a project directory holding keep_nodes.npy + reduce_nodes.py (acoustic_solve.room.build, a room with
# field planes) has raw/ cut to those nodes here, before it goes home, with the Python that runs blab (its shebang)
project_dir=$(dirname "$1")
if [ $status -eq 0 ] && [ -f "$project_dir/keep_nodes.npy" ]; then
  blab_python=$(sed -n '1s/^#!//p' "$(command -v blab)")
  if [ -z "$blab_python" ] || [ ! -f "$project_dir/reduce_nodes.py" ]; then
    echo "run-job.sh (╯°□°)╯ keep_nodes.npy without reduce_nodes.py, or blab has no python shebang" >> solve.log
    status=1
  elif ! $blab_python "$project_dir/reduce_nodes.py" raw "$project_dir/keep_nodes.npy" >> solve.log 2>&1; then
    echo "run-job.sh (╯°□°)╯ reducing raw/ to the kept nodes failed: the full raw/ comes home" >> solve.log
    status=1
  fi
fi
end=$(date +%s.%N)
[ -z "$smi" ] || kill $smi
python3 -c "print('wall_s %.1f' % ($end - $start))" >> solve.log
echo "cpu_s $peak_cpu_s" >> solve.log
if [ -n "$smi" ]; then
  echo "peak_gpu_mib_whole_card $(cut -d, -f2 gpu.csv | tr -dc '0-9\n' | sort -n | tail -1)" >> solve.log
else
  echo "peak_gpu_mib_whole_card none (no card)" >> solve.log
fi
echo "peak_rss_mib $(( peak_rss_kib / 1024 ))" >> solve.log
[ $status -eq 0 ] && echo DONE >> solve.log
exit $status
