#!/bin/bash
#
# System-prompt extraction: run the assistant-axis pipeline (steps 1-4) for one
# Olmo 3 checkpoint. Defaults run Instruct-SFT on the 275 roles.
# Set MODEL, ROLES_JSON and OUT for another checkpoint or role list.
#
# Run on the pod after bootstrap.sh, inside tmux:
#   tmux new -s run
#   bash run_pilot.sh 2>&1 | tee /workspace/persona-shift/phase-4/run.log
#
# Requires OPENROUTER_API_KEY (for step 3), either exported or in a .env file
# next to this script. Reads the role list from the repo checkout unless ROLES_JSON is set.

set -euo pipefail

MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct-SFT}"
WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-/opt/assistant-axis}"
# The default directory follows the model name, so another checkpoint never lands in the SFT directory.
OUT="${OUT:-$WORKSPACE/persona-shift/phase-4/$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
LAYERS="8,16,24"  # quarter, mid, three-quarter depth of the 32-layer model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROLES_JSON="${ROLES_JSON:-$SCRIPT_DIR/../../configs/phase-4/roles_275.json}"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

ROLES=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['roles']))" "$ROLES_JSON")
echo "Model:  $MODEL"
echo "Roles:  $ROLES"
echo "Output: $OUT"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$WORKSPACE/hf"
export VLLM_CACHE_ROOT="$WORKSPACE/vllm-cache"

cd "$CODE_DIR/pipeline"

# The pipeline does not validate the model name: a typo fails late, and an unlisted
# model runs with inferred layer settings. Stop here unless the fork knows the model.
uv run python -c "
import sys
from assistant_axis.models import MODEL_CONFIGS
if sys.argv[1] not in MODEL_CONFIGS:
    sys.exit(f'MODEL {sys.argv[1]!r} is not in MODEL_CONFIGS; known: {sorted(MODEL_CONFIGS)}')
" "$MODEL"

echo "=== Step 1: generate responses (vLLM) ==="
# 0.85, not the pipeline default 0.95: on a 24GB RTX 4090 the default leaves
# no headroom for runtime allocations and the engine dies with CUDA OOM
uv run 1_generate.py \
    --model "$MODEL" \
    --output_dir "$OUT/responses" \
    --gpu_memory_utilization 0.85 \
    --roles $ROLES

# 1_generate.py swallows per-role engine errors and exits 0; do not enter
# step 2 without actual responses on disk
RESPONSE_COUNT=$(find "$OUT/responses" -name '*.jsonl' | wc -l)
if [ "$RESPONSE_COUNT" -eq 0 ]; then
    echo "FATAL: step 1 produced no response files" >&2
    exit 1
fi
echo "Step 1 produced $RESPONSE_COUNT response files"

echo "=== Step 2: extract activations (layers $LAYERS) ==="
# batch_size 8, not the pipeline's 16: the HF forward computes all 32 layers
# regardless of which are stored, and batch 16 at ~700-token sequences OOMs a
# 24GB RTX 4090 next to the 14GB of bf16 weights.
uv run 2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUT/responses" \
    --output_dir "$OUT/activations" \
    --layers "$LAYERS" \
    --batch_size 8 \
    --roles $ROLES

# Steps 3-4 write scores-<judge>/ and vectors-<judge>/; see ../judge_and_build_vectors.sh.
CODE_DIR="$CODE_DIR" bash "$SCRIPT_DIR/../judge_and_build_vectors.sh" "$OUT"

echo "=== Done ==="
ls -la "$OUT"
