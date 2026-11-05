#!/bin/bash
#
# Run several extractions one after another on the pod.
# A failed extraction does not stop the ones after it.
# The last line printed is "CHAIN DONE", which remove_pod_when_done.sh waits for.
#
# Each argument is method:model:run-name, with method "sysprompt":
#   bash run_chain.sh \
#       sysprompt:allenai/Olmo-3-7B-Instruct-DPO:dpo-sysprompt \
#       sysprompt:allenai/Olmo-3-7B-Instruct:rlvr-sysprompt
#
# Output goes to $ROOT/<run-name>, by default $WORKSPACE/persona-shift/phase-4/<run-name>.
# Run inside tmux, with stdout and stderr sent to a log file on the volume.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="${ROOT:-$WORKSPACE/persona-shift/phase-4}"
ROLES_JSON="${ROLES_JSON:-$SCRIPT_DIR/../../configs/phase-4/roles_275.json}"

for spec in "$@"; do
    IFS=: read -r method model run_name <<< "$spec"
    echo "##### $run_name start $(date -u +%H:%M)"
    case "$method" in
        sysprompt)
            MODEL="$model" ROLES_JSON="$ROLES_JSON" OUT="$ROOT/$run_name" \
                bash "$SCRIPT_DIR/run_pilot.sh" || echo "##### $run_name FAILED"
            ;;
        *)
            echo "##### $run_name FAILED: unknown method '$method'"
            ;;
    esac
done
echo "##### CHAIN DONE $(date -u +%H:%M)"
