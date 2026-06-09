# How Reasoning Breaks: Latent-Space Analysis Code

This repository contains the inference, answer extraction, evaluation, and latent-space analysis code used for the NeurIPS submission in `docs/paper.pdf`.

The pipeline is:

1. Start one or more vLLM OpenAI-compatible servers.
2. Run model inference on a JSONL benchmark.
3. Extract the final `\boxed{...}` answer.
4. Compare model answers with reference answers.
5. Merge correctness labels back into generations.
6. Compute latent-space friction and incoherence metrics.
7. Plot correct-vs-incorrect trajectories and phase-boundary/ROC figures.

## Repository Layout

```text
.
├── configs/                  # Editable JSON/env configs for each pipeline stage
├── data/                     # Put benchmark JSONL files here
├── docs/                     # Paper manuscript and documentation assets
├── figures/                  # Generated plots
├── logs/                     # vLLM logs
├── outputs/                  # Predictions, labels, and metrics
├── scripts/                  # Command-line Python entry points and shell helpers
├── src/latent_reasoning/     # Shared config and JSONL utilities
├── docs/paper.pdf
└── requirements.txt
```

The scripts under `scripts/` are the public command-line entry points. Shared helper code lives under `src/latent_reasoning/`, and stage-specific parameters live under `configs/`.

## Config Files

Most Python entry points accept `--config configs/<stage>.json`. Command-line arguments override values from the config file.

| Config | Used by |
| --- | --- |
| `configs/inference.json` | `scripts/run_inference.py` |
| `configs/extract_answers.json` | `scripts/extract_answers.py` |
| `configs/evaluate.json` | `scripts/evaluate_answers.py` |
| `configs/merge.json` | `scripts/merge_results.py` |
| `configs/analysis.json` | `scripts/compute_latent_metrics.py` |
| `configs/trajectory.json` | `scripts/plot_trajectory.py` |
| `configs/physical_boundary.json` | `scripts/plot_phase_boundary.py` |
| `configs/vllm.env.example` | Example environment config for `scripts/start_vllm.sh` |

## Installation

Create an environment with a PyTorch build that matches your CUDA driver, then install the project dependencies.

```bash
conda create -n latent-space python=3.10 -y
conda activate latent-space

# Install torch following the official PyTorch/CUDA instructions for your machine.
# Then install the remaining dependencies:
pip install -r requirements.txt
```

For long-context inference, make sure your vLLM, CUDA, and GPU memory are compatible with the model length you request.

## Input Data Format

Inference expects a JSONL file. Each row should contain a prompt field and, optionally, a reference-answer field:

```json
{"id": "aime2024_001", "query": "Problem text ...", "response": "42"}
```

The default field names are:

| Field | Meaning |
| --- | --- |
| `id` | Example id. If absent, the row index is used. |
| `query` | User prompt/problem text. |
| `response` | Reference answer used for evaluation. |

You can override these names with command-line arguments.

## 1. Start vLLM

Set the model path and launch one server per GPU:

```bash
cp configs/vllm.env.example configs/vllm.env
# Edit configs/vllm.env, then:
set -a
source configs/vllm.env
set +a
scripts/start_vllm.sh
```

Useful options:

```bash
scripts/start_vllm.sh --help
```

If your model does not use DeepSeek-R1-style reasoning tags, add `--disable-reasoning`.

## 2. Run Sharded Inference

Run one process per vLLM server. The example below uses two shards:

```bash
python scripts/run_inference.py --config configs/inference.json --part 1
python scripts/run_inference.py --config configs/inference.json --part 2
```

Each shard writes `outputs/aime2024_qwen3_part1.jsonl`, `outputs/aime2024_qwen3_part2.jsonl`, etc. The script appends to existing files and skips completed ids, so interrupted runs can be resumed.

Combine shards before evaluation:

```bash
cat outputs/aime2024_qwen3_part*.jsonl > outputs/aime2024_qwen3_predictions.jsonl
```

## 3. Extract Boxed Answers

```bash
python scripts/extract_answers.py --config configs/extract_answers.json
```

The extractor uses the last `\boxed{...}` block, which is usually the final answer in chain-of-thought generations.

## 4. Evaluate Answers

```bash
python scripts/evaluate_answers.py --config configs/evaluate.json
```

This writes one row per example with `standard_answer`, `model_answer`, and `is_correct`.

## 5. Merge Correctness Labels

```bash
python scripts/merge_results.py --config configs/merge.json
```

The merged file is the input for latent-space analysis.

## 6. Compute Latent-Space Metrics

```bash
python scripts/compute_latent_metrics.py --config configs/analysis.json
```

By default, the script computes:

| Metric family | Windows |
| --- | --- |
| First-token percentages | `10,20,40,80,100` |
| Last-token percentages | `5,10,20` |
| First-token counts | `2000,4000,6000,8000` |

You can override these:

```bash
python scripts/compute_latent_metrics.py \
  --config configs/analysis.json \
  --model_path /path/to/model \
  --gpu_ids 0,1,2,3 \
  --first_pcts 10,25,50,100 \
  --last_pcts 10,20 \
  --first_tokens 1024,2048
```

The analysis reconstructs the full context as `query + pred` unless `--prediction_includes_prompt` is set.

## 7. Plot Figures

Average correct-vs-incorrect trajectory:

```bash
python scripts/plot_trajectory.py --config configs/trajectory.json
```

Phase boundary and ROC curve:

```bash
python scripts/plot_phase_boundary.py --config configs/physical_boundary.json
```

Run the full pipeline after vLLM is already running:

```bash
scripts/run_pipeline.sh
```

## Reproducing the Paper Setup

The manuscript `docs/paper.pdf` describes the evaluated models and datasets. To reproduce a paper run:

1. Download the benchmark JSONL files and model checkpoints named in the paper.
2. Format each benchmark as JSONL with `id`, `query`, and `response`.
3. Run the full pipeline above once per model and dataset.
4. Compare generated figures and metrics with the paper results.

## Notes and Limitations

The answer verifier is intentionally lightweight and rule-based. For new datasets with complex symbolic answers, inspect a sample of false positives/false negatives and extend `check_equivalence` in `scripts/evaluate_answers.py`.

Latent-space analysis is memory intensive because it requests all hidden states. Use fewer GPUs, smaller batches, or shorter sequences if you encounter OOM errors. The script processes one example at a time per worker and records `error` fields instead of stopping the whole run when an individual example fails.
