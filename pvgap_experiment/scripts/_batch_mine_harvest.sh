#!/bin/bash
# Batch-mine all harvested OA papers through mine_echem_rules (standard mode).
# Expects OPENAI_API_KEY in env.
set -u
cd "D:/Obsidian/data/PHD/Self-Planning Prompting+reasoning LLM做电化学可解释性"
HARVEST_DIR="pvgap_experiment/data/literature/raw/oa_harvest"
STAGE_DIR="pvgap_experiment/data/echem_rules/staging/oa_harvest"
mkdir -p "$STAGE_DIR"
LOG="/tmp/batch_mine_harvest.log"
: > "$LOG"
for f in "$HARVEST_DIR"/*.txt; do
    bn=$(basename "$f" .txt)
    # truncate to 20 chars for source_key (miner caps at 20)
    key=$(echo "$bn" | sed 's/[^A-Za-z0-9_-]/_/g' | cut -c1-20)
    out="$STAGE_DIR/${bn}.jsonl"
    if [ -f "$out" ]; then
        echo "SKIP $bn (already mined)" | tee -a "$LOG"
        continue
    fi
    echo "--- MINE $bn ---" | tee -a "$LOG"
    python -m pvgap_experiment.scripts.mine_echem_rules \
        --source "$f" --out "$out" --source-key "$key" \
        --max-segments 30 --model gpt-4o >> "$LOG" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0)
    echo "  $bn: staged $n" | tee -a "$LOG"
done
echo "=== BATCH DONE ===" | tee -a "$LOG"
wc -l "$STAGE_DIR"/*.jsonl 2>/dev/null | tail -20 | tee -a "$LOG"
