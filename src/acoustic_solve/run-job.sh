#!/bin/bash
# ON A POD, as a podd job's command (acoustic_solve.solve_job writes the JOB and ships this file and
# collect_result.py with it; podd runs it in the job's directory, in a session of its own, and owns its timeout and
# kill: podd.job_contract, ~/podd):
#
#   bash run-job.sh <project.blab.json> <request.json> <beat_cuda|beat_cpu> [threads]
#
# [threads]: Julia's and OpenBLAS's thread count (default 8, the image's OPENBLAS_NUM_THREADS); a job that declares
# fewer cores than a whole pod (a room's interior FEM on a CPU pod: acoustic_solve.room.build) passes its own `cpu`.
# ONE SOLVE of a staged project (any client's: a cabinet's build/cabinet.blab.json, a horn's build/core.blab.json +
# build/runs/, a room's build/room.blab.json) on BEAT CUDA or the pod's CPU. The image (image/) puts julia and blab on
# PATH (/usr/local/bin) and its env (depot, CPU target, BLAS thread cap) into every ssh session, so nothing is sourced
# here. Exits non-zero when the solve fails (podd then FAILS the job, with what came back).
#
# Writes, in the job's directory:
#   raw/       the blab result, built frequency by frequency by collect_result.py from blab's own output (.solving/,
#              never an output): every file in raw/ is final (renumbered, reduced to the kept nodes when the project
#              directory holds keep_nodes.npy), so whatever stops this, what comes home is the finished frequencies
#   solve.log  blab's progress, then wall_s, cpu_s (the largest CPU-seconds sum over this session's processes,
#              sampled every second: cpu_s / wall_s is the cores it kept busy), peak_gpu_mib_whole_card, peak_rss_mib, then DONE (or SOLVE FAILED first)
#              (a resumed attempt appends to the solve.log podd shipped back)
#   gpu.csv    nvidia-smi every 0.5 s; on a pod with no card (a CPU pod) one line saying so
#   progress   `<frequencies raw/ has> <frequencies of the request>`, every second (podd's `progress`)
#
# RESUMABLE (podd's `resumable`, `resume_from` raw/manifest.json + solve.log: acoustic_solve.solve_job): podd ships
# back raw/manifest.json ONLY, never the frequencies earlier attempts brought home, and lays this attempt's raw/ over
# the stage's at home. So with raw/manifest.json here this attempt solves only the frequencies it lacks, and its raw/
# holds the manifest (every frequency done, kept or new) and the new frequencies' files.
# SEVERAL JOBS SHARE A POD: podd starts each in its own session, so the RSS summed over this session is this job's
# (its python and julia workers); peak_gpu_mib is the whole card's (nvidia-smi sees every job on it), as its name says.
set -eu
threads=${4:-8}
julia=$(command -v julia) || { echo "run-job.sh (╯°□°)╯ no julia on PATH: not the solver image" >> solve.log; exit 1; }
blab=$(command -v blab) || { echo "run-job.sh (╯°□°)╯ no blab on PATH: not the solver image" >> solve.log; exit 1; }
# the Python that runs blab (its shebang: numpy, for the reduction) runs the collector
python=$(sed -n '1s/^#!//p' "$blab")
[ -n "$python" ] || { echo "run-job.sh (╯°□°)╯ blab ($blab) has no python shebang" >> solve.log; exit 1; }
collector="$(dirname "$0")/collect_result.py"
keep="$(dirname "$1")/keep_nodes.npy"
export OPENBLAS_NUM_THREADS=$threads
smi=
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 500 > gpu.csv &
  smi=$!
else
  echo "no nvidia-smi on this pod (a CPU pod): no card to sample" > gpu.csv
fi
session=$(ps -o sid= -p $$ | tr -d ' ')

resumed=
[ -f raw/manifest.json ] && resumed=--resumed
rm -rf .solving
left=$($python "$collector" plan "$2" raw .solving-request.json 2>> solve.log) || {
  echo "run-job.sh (╯°□°)╯ planning the solve failed" >> solve.log; exit 1; }
[ -z "$resumed" ] || echo "run-job.sh: RESUMED: $left frequencies of $2 left to solve" >> solve.log
total=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["frequencies_hz"]))' "$2")
start=$(date +%s.%N)
solve=
if [ "$left" -eq 0 ]; then
  echo "run-job.sh: every frequency is in raw/: nothing to solve" >> solve.log
else
  blab project solve "$1" --request .solving-request.json --backend "$3" --output .solving \
      --julia-executable "$julia" --julia-threads "$threads" >> solve.log 2>&1 &
  solve=$!
fi
status=0
peak_rss_kib=0
peak_cpu_s=0
# every second: the RSS and the cumulative CPU seconds (ps `times`) summed over this session; julia lives through the
# whole solve, so the largest CPU sum is the solve's (2026-10-15: bash's `times` missed julia, which blab never reaps);
# and every frequency blab finished moved into raw/
while [ -n "$solve" ] && kill -0 $solve 2>/dev/null; do
  read -r rss cpu < <(ps -eo sid=,rss=,times= | awk -v session=$session '$1 == session {r += $2; c += $3} END {print r + 0, c + 0}')
  [ $rss -gt $peak_rss_kib ] && peak_rss_kib=$rss
  [ $cpu -gt $peak_cpu_s ] && peak_cpu_s=$cpu
  if ! done_now=$($python "$collector" collect .solving raw "$keep" 2>> solve.log); then
    echo "run-job.sh (╯°□°)╯ collecting the finished frequencies into raw/ failed: the solve is stopped" >> solve.log
    kill $solve 2>/dev/null || true
    status=1
    break
  fi
  echo "$done_now $total" > progress.writing && mv progress.writing progress
  sleep 1
done
if [ -n "$solve" ]; then
  wait $solve || { solved=$?; [ $status -ne 0 ] || status=$solved; echo "run-job.sh: blab exited $solved" >> solve.log; }
fi
if done_now=$($python "$collector" finish .solving raw "$keep" $resumed 2>> solve.log); then
  echo "$done_now $total" > progress.writing && mv progress.writing progress
  if [ $status -eq 0 ] && [ "$done_now" -ne "$total" ]; then
    echo "run-job.sh (╯°□°)╯ blab exited 0 but raw/ holds $done_now of $total frequencies" >> solve.log
    status=1
  fi
else
  echo "run-job.sh (╯°□°)╯ collecting the last frequencies into raw/ failed" >> solve.log
  [ $status -ne 0 ] || status=1
fi
[ $status -eq 0 ] || echo "SOLVE FAILED (exit $status)" >> solve.log
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
