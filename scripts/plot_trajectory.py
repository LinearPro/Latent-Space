import os
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, require_args
from latent_reasoning.io import ensure_parent_dir

def load_jsonl(path):
    import pandas as pd

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_trajectory(input_file, output_file, pcts):
    import matplotlib.pyplot as plt

    df = load_jsonl(input_file)
    if df.empty:
        raise ValueError(f"No rows found in {input_file}")

    ensure_parent_dir(output_file)

    x_cols = [f'incoh_first_{p}pct' for p in pcts]
    y_cols = [f'fric_first_{p}pct' for p in pcts]
    missing = [col for col in x_cols + y_cols + ['is_correct'] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mean_stats = df.groupby('is_correct')[x_cols + y_cols].mean(numeric_only=True)

    plt.figure(figsize=(10, 8))

    if False in mean_stats.index:
        x_vals_f = mean_stats.loc[False, x_cols].values
        y_vals_f = mean_stats.loc[False, y_cols].values
        plt.plot(x_vals_f, y_vals_f, marker='o', color='red', label='Incorrect (Average)', linewidth=2, markersize=8)

    if True in mean_stats.index:
        x_vals_t = mean_stats.loc[True, x_cols].values
        y_vals_t = mean_stats.loc[True, y_cols].values
        plt.plot(x_vals_t, y_vals_t, marker='o', color='blue', label='Correct (Average)', linewidth=2, markersize=8)

    labels = [f'{p}%' for p in pcts]
    for val, color, offset in [(True, 'blue', 10), (False, 'red', -15)]:
        if val in mean_stats.index:
            for i, label in enumerate(labels):
                plt.annotate(label,
                             (mean_stats.loc[val, x_cols[i]], mean_stats.loc[val, y_cols[i]]),
                             textcoords="offset points",
                             xytext=(0, offset),
                             ha='center',
                             color=color,
                             fontweight='bold')

    plt.xlabel('Incoherence (incoh)')
    plt.ylabel('Friction (fric)')
    plt.title('Average Trajectory: Correct vs Incorrect CoT')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved average trajectory plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average latent-space trajectories for correct and incorrect CoTs.")
    parser.add_argument("--input-file")
    parser.add_argument("--output-file", default="figures/average_trajectory_comparison.png")
    parser.add_argument("--pcts", default="10,20,40,80,100", help="Comma-separated first-percent windows to plot.")
    args = parse_args_with_config(parser)
    require_args(args, ["input_file"])

    pcts = [int(x) for x in args.pcts] if isinstance(args.pcts, list) else [int(x.strip()) for x in args.pcts.split(",") if x.strip()]
    plot_trajectory(args.input_file, args.output_file, pcts)
