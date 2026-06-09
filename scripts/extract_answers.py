import json
import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, require_args
from latent_reasoning.io import ensure_parent_dir

def extract_boxed_content(text):
    """
    从文本中提取 \boxed{...} 的内容。
    逻辑：
    1. 寻找文本中 '最后一个' \boxed{ 的位置（通常最后的才是最终答案）。
    2. 使用堆栈/计数器方法解析嵌套的大括号，确保提取完整且准确。
    """
    if not text or not isinstance(text, str):
        return None

    # 关键步骤：使用 rfind 找最后一个 \boxed{，这在 CoT（思维链）中很关键
    # 如果你想找第一个，可以改成 text.find("\\boxed{")
    target = "\\boxed{"
    start_index = text.rfind(target)
    
    if start_index == -1:
        return None
    
    # 内容开始的位置（跳过 \boxed{ 这7个字符）
    content_start = start_index + len(target)
    
    balance = 1
    current_index = content_start
    
    while current_index < len(text):
        char = text[current_index]
        
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
            
        if balance == 0:
            return text[content_start:current_index]
        
        current_index += 1
        
    return None

def process_file(input_path, output_path, pred_field="pred", answer_field="extracted_answer"):
    print(f"Processing file: {input_path}")
    
    success_count = 0
    total_count = 0
    
    ensure_parent_dir(output_path)

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                data = json.loads(line)
                
                pred_text = data.get(pred_field, "")
                
                # 执行提取
                answer = extract_boxed_content(pred_text)
                
                data[answer_field] = answer
                
                # 简单的统计
                if answer:
                    success_count += 1
                
                # 写入新文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num} 行不是有效的 JSON 数据，已跳过。")
            except Exception as e:
                print(f"错误: 处理第 {line_num} 行时发生未知错误: {e}")

    print("-" * 30)
    print("Done.")
    print(f"Total rows: {total_count}")
    print(f"Extracted rows: {success_count}")
    print(f"Extraction rate: {success_count/total_count:.2%}" if total_count > 0 else "N/A")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the last LaTeX \\boxed{...} answer from model predictions.")
    parser.add_argument("--input-file")
    parser.add_argument("--output-file")
    parser.add_argument("--pred-field", default="pred")
    parser.add_argument("--answer-field", default="extracted_answer")
    args = parse_args_with_config(parser)
    require_args(args, ["input_file", "output_file"])

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    process_file(args.input_file, args.output_file, args.pred_field, args.answer_field)
