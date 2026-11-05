# Scripts

Pod scripts run on a Runpod GPU pod, inside the assistant-axis environment, with this repo checked out on the pod.

| script | runs on | what it does |
|---|---|---|
| `judge_and_build_vectors.sh` | laptop or pod | Judge, vectors and assistant axis for one run directory, into `scores-<judge>/`, `vectors-<judge>/` and `axis-<judge>.pt`. Rescores a copied run without a GPU; `pod/run_pilot.sh` calls it for steps 3-5. |
| `pod/bootstrap.sh` | pod | Sets up a fresh pod: clones the assistant-axis fork, installs it with vLLM, puts caches on the volume. |
| `pod/run_pilot.sh` | pod | System-prompt extraction for one checkpoint (`MODEL`): generate, activations, judge, vectors. |
| `pod/run_chain.sh` | pod | Runs several extractions in sequence (`method:model:run-name`) and prints `CHAIN DONE` at the end. |
| `pod/remove_pod_when_done.sh` | pod | Removes the pod when a run log shows `CHAIN DONE`, or after a maximum number of hours. |

The judge, vector and axis steps are `3_judge.py`, `4_vectors.py` and `5_axis.py` from assistant-axis.
The axis needs the `default` role (`configs/phase-4/roles_default.json`), which the judge skips and whose vector averages all responses.

## Judge

The judge is `z-ai/glm-5.3-flash` on OpenRouter, Z.AI pinned.

## Unattended runs on a pod

Two tmux sessions, so the pod stops billing when the work is done:

```bash
LOG=/workspace/persona-shift/phase-4/chain.log
tmux new -d -s run "bash scripts/pod/run_chain.sh sysprompt:allenai/Olmo-3-7B-Instruct-DPO:dpo-sysprompt > $LOG 2>&1"
tmux new -d -s remove "bash scripts/pod/remove_pod_when_done.sh $LOG 24"
```

The watcher is the only safeguard.
It runs inside the pod and cannot act if the pod's processes die, and `runpodctl` 2.14 has no option to stop a pod at a set time.
Check the pod list after an unattended run.
