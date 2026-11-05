#!/bin/bash
#
# Remove this pod once a run has finished, so it stops billing with nobody watching.
# Waits until the log contains "CHAIN DONE", or until MAX_HOURS have passed.
# Run outputs live on the network volume and survive the removal.
#
#   bash remove_pod_when_done.sh <log-file> <max-hours>
#
# Run in its own tmux session, next to the one that runs run_chain.sh.
# This is the only safeguard, and it dies with the pod's processes.
# runpodctl 2.14 has no option to stop a pod at a set time, so check the pod list after a run.
#
# Needs RUNPOD_API_KEY, exported or in a .env file next to this script.
# It has to be the account key: the key Runpod injects into a pod cannot remove the pod.

set -uo pipefail

LOG="$1"
MAX_HOURS="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# Runpod sets RUNPOD_POD_ID for the container's first process only, not for SSH or tmux sessions.
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')}"
if [ -z "$POD_ID" ]; then
    echo "cannot determine the pod id" >&2
    exit 1
fi

DEADLINE=$(( $(date -u +%s) + MAX_HOURS * 3600 ))
until grep -q "CHAIN DONE" "$LOG" 2>/dev/null || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do
    sleep "${POLL_SECONDS:-60}"
done

REMOVAL_LOG="$(dirname "$LOG")/pod_removal.log"
if grep -q "CHAIN DONE" "$LOG" 2>/dev/null; then
    echo "$(date -u '+%F %H:%M') removing $POD_ID: run finished" >> "$REMOVAL_LOG"
else
    echo "$(date -u '+%F %H:%M') removing $POD_ID: ${MAX_HOURS}h deadline reached" >> "$REMOVAL_LOG"
fi
runpodctl remove pod "$POD_ID" >> "$REMOVAL_LOG" 2>&1
