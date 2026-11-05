#!/bin/bash
#
# Judge one run's responses, build per-role vectors from those scored 3, then the assistant axis (assistant-axis steps 3-5).
#
#   bash scripts/judge_and_build_vectors.sh data/phase-4/dpo-sysprompt
#
# Writes scores-<judge>/, vectors-<judge>/ and, when the run has the default role, axis-<judge>.pt
# Judge settings (OpenRouter, Z.AI pinned)
# Needs OPENROUTER_API_KEY.

set -euo pipefail

OUT="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${CODE_DIR:-$SCRIPT_DIR/../external/assistant-axis}"
JUDGE_MODEL="${JUDGE_MODEL:-z-ai/glm-5.3-flash}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-z-ai}"
JUDGE_TAG="$(basename "$JUDGE_MODEL")"

for env_file in "$SCRIPT_DIR/pod/.env" "$SCRIPT_DIR/../.env"; do
    if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$env_file" ]; then
        set -a; source "$env_file"; set +a
    fi
done

echo "Run:   $OUT"
echo "Judge: $JUDGE_MODEL via $JUDGE_PROVIDER, writing scores-$JUDGE_TAG and vectors-$JUDGE_TAG"

cd "$CODE_DIR/pipeline"

echo "=== Step 3: judge responses, start $(date -u +%H:%M:%S) UTC ==="
START=$SECONDS
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$OPENROUTER_API_KEY" \
uv run 3_judge.py \
    --responses_dir "$OUT/responses" \
    --output_dir "$OUT/scores-$JUDGE_TAG" \
    --judge_model "$JUDGE_MODEL" \
    --max_tokens 4096 \
    --batch_size "${JUDGE_BATCH_SIZE:-200}" \
    --judge_extra_body "{\"provider\": {\"order\": [\"$JUDGE_PROVIDER\"], \"allow_fallbacks\": false}, \"reasoning\": {\"effort\": \"minimal\", \"exclude\": true}}"
echo "=== Step 3 took $(( (SECONDS - START) / 60 )) min ==="

echo "=== Step 4: per-role vectors, start $(date -u +%H:%M:%S) UTC ==="
START=$SECONDS
uv run 4_vectors.py \
    --activations_dir "$OUT/activations" \
    --scores_dir "$OUT/scores-$JUDGE_TAG" \
    --output_dir "$OUT/vectors-$JUDGE_TAG"
echo "=== Step 4 took $(( (SECONDS - START) / 60 )) min ==="

# 5_axis.py exits 1 without a default vector.
if [ -f "$OUT/vectors-$JUDGE_TAG/default.pt" ]; then
    echo "=== Step 5: assistant axis ==="
    uv run 5_axis.py \
        --vectors_dir "$OUT/vectors-$JUDGE_TAG" \
        --output "$OUT/axis-$JUDGE_TAG.pt"
else
    echo "=== Step 5 skipped: no default vector ==="
fi
