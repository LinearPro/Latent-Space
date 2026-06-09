#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INFERENCE_CONFIG="${INFERENCE_CONFIG:-configs/inference.json}"
EXTRACT_CONFIG="${EXTRACT_CONFIG:-configs/extract_answers.json}"
EVALUATE_CONFIG="${EVALUATE_CONFIG:-configs/evaluate.json}"
MERGE_CONFIG="${MERGE_CONFIG:-configs/merge.json}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-configs/analysis.json}"
TRAJECTORY_CONFIG="${TRAJECTORY_CONFIG:-configs/trajectory.json}"
BOUNDARY_CONFIG="${BOUNDARY_CONFIG:-configs/physical_boundary.json}"

NUM_PARTS="${NUM_PARTS:-$(python -c "import json; print(json.load(open('${INFERENCE_CONFIG}')).get('num_parts', 2))")}"
OUTPUT_DIR="${OUTPUT_DIR:-$(python -c "import json; print(json.load(open('${INFERENCE_CONFIG}')).get('output_dir', 'outputs'))")}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$(python -c "import json; print(json.load(open('${INFERENCE_CONFIG}')).get('output_prefix', 'predictions'))")}"

echo "Running inference shards with ${INFERENCE_CONFIG}"
for part in $(seq 1 "$NUM_PARTS"); do
    python scripts/run_inference.py --config "$INFERENCE_CONFIG" --part "$part"
done

echo "Combining shard predictions"
cat "${OUTPUT_DIR}/${OUTPUT_PREFIX}"_part*.jsonl > "${OUTPUT_DIR}/${OUTPUT_PREFIX}_predictions.jsonl"

echo "Extracting boxed answers"
python scripts/extract_answers.py --config "$EXTRACT_CONFIG"

echo "Evaluating answers"
python scripts/evaluate_answers.py --config "$EVALUATE_CONFIG"

echo "Merging correctness labels"
python scripts/merge_results.py --config "$MERGE_CONFIG"

echo "Computing latent-space metrics"
python scripts/compute_latent_metrics.py --config "$ANALYSIS_CONFIG"

echo "Plotting average trajectory"
python scripts/plot_trajectory.py --config "$TRAJECTORY_CONFIG"

echo "Plotting physical boundary and ROC"
python scripts/plot_phase_boundary.py --config "$BOUNDARY_CONFIG"

echo "Pipeline complete."
