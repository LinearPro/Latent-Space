import json
import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, require_args
from latent_reasoning.io import ensure_parent_dir

def merge_datasets(eval_result_path, pred_extract_path, output_path, keep_fields=None):
    """
    Merge correctness labels with prediction records by row order.
    """
    keep_fields = keep_fields or ["id", "query", "response", "pred", "extracted_answer"]
    
    if not os.path.exists(eval_result_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_result_path}")
    if not os.path.exists(pred_extract_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_extract_path}")

    merged_data = []

    try:
        print("Merging data...")
        with open(eval_result_path, 'r', encoding='utf-8') as f_eval, \
             open(pred_extract_path, 'r', encoding='utf-8') as f_pred:
            
            for line_idx, (line_eval, line_pred_src) in enumerate(zip(f_eval, f_pred)):
                try:
                    data_eval = json.loads(line_eval)
                    data_pred_src = json.loads(line_pred_src)
                    
                    is_correct = data_eval.get("is_correct", False)

                    entry = {field: data_pred_src.get(field) for field in keep_fields if field in data_pred_src}
                    entry["is_correct"] = is_correct
                    if "model_answer" in data_eval:
                        entry["model_answer"] = data_eval["model_answer"]
                    if "standard_answer" in data_eval:
                        entry["standard_answer"] = data_eval["standard_answer"]
                    merged_data.append(entry)
                    
                except json.JSONDecodeError:
                    print(f"Line {line_idx+1} JSON decode failed, skipped.")

        ensure_parent_dir(output_path)
        if merged_data:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for entry in merged_data:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"Done. Output: {output_path}")
            print(f"Merged rows: {len(merged_data)}")
            
            correct_count = sum(1 for d in merged_data if d['is_correct'])
            print(f"Correct: {correct_count} (accuracy: {correct_count/len(merged_data)*100:.2f}%)")
        else:
            print("No valid rows were merged.")

    except Exception as e:
        print(f"Failed during merge: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge evaluation correctness labels back into prediction JSONL records.")
    parser.add_argument("--eval-file")
    parser.add_argument("--predictions-file")
    parser.add_argument("--output-file")
    parser.add_argument(
        "--keep-fields",
        default="id,query,response,pred,extracted_answer",
        help="Comma-separated fields to keep from the prediction file.",
    )
    args = parse_args_with_config(parser)
    require_args(args, ["eval_file", "predictions_file", "output_file"])

    if isinstance(args.keep_fields, list):
        keep_fields = [str(field).strip() for field in args.keep_fields if str(field).strip()]
    else:
        keep_fields = [field.strip() for field in args.keep_fields.split(",") if field.strip()]
    merge_datasets(args.eval_file, args.predictions_file, args.output_file, keep_fields)
