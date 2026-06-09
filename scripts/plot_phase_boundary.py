import json
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, require_args
from latent_reasoning.io import ensure_parent_dir

def load_jsonl(input_file):
    data = []
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if 'diff norm1' in item:
                    item['diff norms1'] = item['diff norm1']
                    item['diff norms2'] = item['diff norm2']
                if 'is_correct' in item:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    return data

def extract_features(item, x_metric, y_metric):
    import numpy as np

    if x_metric in item and y_metric in item:
        x_value = item.get(x_metric)
        y_value = item.get(y_metric)
        if x_value is None or y_value is None:
            return None
        return float(x_value), float(y_value)

    d1 = item.get('diff norms1')
    d2 = item.get('diff norms2')
    a1 = item.get('angle1')
    a2 = item.get('angle2')
    if not (d1 and d2 and a1 and a2):
        return None

    fric = np.array(d2) - np.array(d1)
    incoh = np.array(a2) - np.array(a1)
    if len(fric) < 5:
        return None

    mean_fric = np.mean(fric)
    final_incoh = np.mean(incoh[int(len(incoh) * 0.90):])
    return float(mean_fric), float(final_incoh)

def plot_roc_curve(y_test, y_score, output_file):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Correctness Classification ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    return roc_auc

def plot_phase_boundary_with_score(input_file, output_file, callback_file, roc_file, x_metric, y_metric):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    print(f"Running physical-metric evaluation and phase plotting: {input_file}")
    
    data = load_jsonl(input_file)

    X_list, y_list = [], []
    callback_data = []

    for item in data:
        try:
            features = extract_features(item, x_metric, y_metric)
            if features is None:
                continue
            mean_fric, final_incoh = features
            label = 1 if item.get('is_correct', False) else 0
            
            callback_record = {
                "is_correct": bool(label),
                "friction": float(mean_fric),
                "incoherence": float(final_incoh)
            }
            callback_data.append(callback_record)

            X_list.append([mean_fric, final_incoh])
            y_list.append(label)
        except (TypeError, ValueError):
            continue

    if callback_file:
        ensure_parent_dir(callback_file)
        print(f"Saving feature data to {callback_file}")
        with open(callback_file, "w", encoding="utf-8") as f:
            for record in callback_data:
                f.write(json.dumps(record) + "\n")

    X = np.array(X_list)
    y = np.array(y_list)
    
    n_correct = np.sum(y == 1)
    n_incorrect = np.sum(y == 0)
    print(f"Dataset summary: total={len(y)}, correct={n_correct}, incorrect={n_incorrect}")
    
    if n_correct < 10 or n_incorrect < 10:
        raise ValueError("Too few positive or negative examples. Use a balanced file with enough correct and incorrect samples.")

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(C=1.0, penalty='l2')
    )
    
    # 简单切分训练测试集以计算验证准确率
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    y_score = model.predict_proba(X_test)[:, 1]
    print(f"\nPhysical metric accuracy = {accuracy:.2%}")
    if roc_file:
        ensure_parent_dir(roc_file)
        roc_auc = plot_roc_curve(y_test, y_score, roc_file)
        print(f"ROC AUC = {roc_auc:.3f}. Saved ROC curve to {roc_file}")

    model.fit(X, y)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # 去除画图用的极端值
    df_temp = pd.DataFrame(X, columns=['Friction', 'Incoherence'])
    mask = (df_temp['Friction'] < df_temp['Friction'].quantile(0.98)) & \
           (df_temp['Incoherence'] < df_temp['Incoherence'].quantile(0.98))
    X_plot = X[mask]
    y_plot = y[mask]
    
    # 创建网格
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # 预测概率
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # 绘制等高线
    contour = plt.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6)
    cbar = plt.colorbar(contour)
    cbar.set_label("Probability of Correctness $P(Correct)$", fontsize=12)

    # 绘制 50% 决策边界
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')

    # 绘制散点
    plt.scatter(X_plot[y_plot==1, 0], X_plot[y_plot==1, 1], c='white', edgecolors='#1f77b4', s=50, label='Correct', alpha=0.8, linewidth=1.5)
    plt.scatter(X_plot[y_plot==0, 0], X_plot[y_plot==0, 1], c='white', edgecolors='#d62728', s=50, marker='X', label='Incorrect', alpha=0.8, linewidth=1.5)

    plt.text(x_min + (x_max-x_min)*0.05, y_max*0.95, 
             f"Prediction Accuracy: {accuracy:.1%}", 
             fontsize=14, fontweight='bold', color='black',
             bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title("Thermodynamic Phase Diagram with Predictive Score", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(r"Cognitive Effort ($\bar{\Phi}_{diss}$)", fontsize=12, fontweight='bold')
    plt.ylabel(r"Logical Misalignment ($\mathcal{I}_{final}$)", fontsize=12, fontweight='bold')

    plt.legend(loc='lower right', framealpha=0.9)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    
    ensure_parent_dir(output_file)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved phase boundary plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a physical phase boundary and ROC curve from latent-space metrics.")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str, default="figures/physical_boundary_with_score.png")
    parser.add_argument("--callback-file", type=str, default="outputs/physical_features.jsonl")
    parser.add_argument("--roc-file", type=str, default="figures/roc_curve.png")
    parser.add_argument("--x-metric", type=str, default="fric_first_100pct", help="Scalar friction metric column. Falls back to sequence fields if absent.")
    parser.add_argument("--y-metric", type=str, default="incoh_last_10pct", help="Scalar incoherence metric column. Falls back to sequence fields if absent.")
    args = parse_args_with_config(parser)
    require_args(args, ["input_file"])
    
    plot_phase_boundary_with_score(
        args.input_file,
        args.output_file,
        args.callback_file,
        args.roc_file,
        args.x_metric,
        args.y_metric,
    )
