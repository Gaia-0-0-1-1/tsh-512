#!/bin/bash
cd ~/workstation/lab
echo "$(date) E7 v3 launching (28 jobs, 7-way parallel)" >> runs/e7_grid_v3.log

python3 - <<'INNER'
from concurrent.futures import ThreadPoolExecutor
import subprocess, sys, os

jobs = [l.strip() for l in open("runs/jobs_e7.txt") if l.strip()]
print(f"{len(jobs)} jobs", flush=True)

def run(line):
    args = line.split()
    out = args[args.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    with open(out + ".out", "w") as f:
        r = subprocess.run(
            [sys.executable, "train_math.py"] + args,
            stdout=f, stderr=subprocess.STDOUT, env=env, cwd="."
        )
    return ("FAIL " if r.returncode else "done ") + out

with ThreadPoolExecutor(max_workers=7) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)
print("E7 GRID DONE", flush=True)
INNER

echo "$(date) E7 v3 done" >> runs/e7_grid_v3.log
