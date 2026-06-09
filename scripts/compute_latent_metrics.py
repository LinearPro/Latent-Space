import os
# 优化显存分配策略，减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, parse_int_list, require_args
from latent_reasoning.io import ensure_parent_dir

# ==================== 辅助结构 ====================

def get_empty_metrics(first_pcts=None, last_pcts=None, first_tokens=None):
    """生成空的结果字典结构，确保报错或异常时数据字段对齐"""
    first_pcts = first_pcts or [10, 20, 40, 80, 100]
    last_pcts = last_pcts or [5, 10, 20]
    first_tokens = first_tokens or [2000, 4000, 6000, 8000]
    keys = []
    # 1. 前 10%, 20%, 40%, 80%, 100%
    for p in first_pcts:
        keys.extend([f"fric_first_{p}pct", f"incoh_first_{p}pct"])
    # 2. 后 5%, 10%, 20%
    for p in last_pcts:
        keys.extend([f"fric_last_{p}pct", f"incoh_last_{p}pct"])
    # 3. 前 2000, 4000, 6000, 8000 tokens
    for n in first_tokens:
        keys.extend([f"fric_first_{n}", f"incoh_first_{n}"])
    return {k: None for k in keys}

# ==================== 核心计算逻辑 ====================

def calculate_scalar_metrics(hidden_states, first_pcts=None, last_pcts=None, first_tokens=None):
    import numpy as np
    import torch
    import torch.nn.functional as F

    """
    计算指标的函数。
    它会自动根据传入 tensor 的 device 决定是在 CPU 还是 GPU 上计算。
    """
    num_layers = len(hidden_states) - 1
    first_pcts = first_pcts or [10, 20, 40, 80, 100]
    last_pcts = last_pcts or [5, 10, 20]
    first_tokens = first_tokens or [2000, 4000, 6000, 8000]
    if num_layers < 1: 
        return None

    # 获取维度信息
    first_layer = hidden_states[1][0] # [Seq, Hidden]
    seq_len = first_layer.shape[0]
    device = first_layer.device
    
    if seq_len < 2: 
        return None

    # 初始化累加器 (float32 以保证精度)
    micro_sum_diffs = torch.zeros(seq_len - 1, dtype=torch.float32, device=device)
    micro_sum_angles = torch.zeros(seq_len - 1, dtype=torch.float32, device=device)
    macro_sum_states = torch.zeros((seq_len, first_layer.shape[-1]), dtype=torch.float32, device=device)

    # 逐层计算
    for i in range(1, len(hidden_states)):
        # 确保转为 float32 进行计算
        curr = hidden_states[i][0].float()
        macro_sum_states += curr
        
        diffs = curr[1:] - curr[:-1]
        micro_sum_diffs += torch.norm(diffs, p=2, dim=-1)
        
        h_curr, h_prev = curr[1:], curr[:-1]
        cos = F.cosine_similarity(h_curr, h_prev, dim=-1, eps=1e-8).clamp(-1, 1)
        micro_sum_angles += torch.acos(cos) * (180.0 / np.pi)
        
        # 显式删除中间变量，节省内存
        del curr, diffs, cos

    # 1. Macro Indicators
    macro_states = macro_sum_states / num_layers
    macro_diffs = macro_states[1:] - macro_states[:-1]
    n1 = torch.norm(macro_diffs, p=2, dim=-1)
    
    h_curr_m, h_prev_m = macro_states[1:], macro_states[:-1]
    cos_m = F.cosine_similarity(h_curr_m, h_prev_m, dim=-1, eps=1e-8).clamp(-1, 1)
    a1 = torch.acos(cos_m) * (180.0 / np.pi)

    # 2. Micro Indicators
    n2 = micro_sum_diffs / num_layers
    a2 = micro_sum_angles / num_layers
    
    # 3. Scalars (序列长度的差异)
    fric_seq = n2 - n1
    incoh_seq = a2 - a1
    
    L = len(fric_seq) # L = seq_len - 1
    
    def get_mean(seq, start, end):
        # 限制越界
        start = max(0, min(start, L))
        end = max(0, min(end, L))
        if start >= end: return None
        val = seq[start:end].mean()
        if torch.isnan(val): return None
        return val.item()

    metrics = {}
    
    # ---------------- 需求 1: 前 10%, 20%, 40%, 80%, 100% ----------------
    for p in first_pcts:
        end_idx = max(1, int(L * (p / 100.0)))
        metrics[f"fric_first_{p}pct"] = get_mean(fric_seq, 0, end_idx)
        metrics[f"incoh_first_{p}pct"] = get_mean(incoh_seq, 0, end_idx)
        
    # ---------------- 需求 2: 后 5%, 10%, 20% ----------------
    for p in last_pcts:
        start_idx = L - max(1, int(L * (p / 100.0)))
        metrics[f"fric_last_{p}pct"] = get_mean(fric_seq, start_idx, L)
        metrics[f"incoh_last_{p}pct"] = get_mean(incoh_seq, start_idx, L)
        
    # ---------------- 需求 3: 前 2000, 4000, 6000, 8000 tokens ----------------
    for n in first_tokens:
        end_idx = n 
        # 如果截断长度超出了实际长度，直接赋 None，拒绝计算被污染的均值
        if end_idx > L:
            metrics[f"fric_first_{n}"] = None
            metrics[f"incoh_first_{n}"] = None
        else:
            metrics[f"fric_first_{n}"] = get_mean(fric_seq, 0, end_idx)
            metrics[f"incoh_first_{n}"] = get_mean(incoh_seq, 0, end_idx)
    
    del micro_sum_diffs, micro_sum_angles, macro_sum_states, macro_states, fric_seq, incoh_seq
    return metrics

# ==================== 工作进程 ====================

def worker_process(rank, gpu_ids, lines_chunk, model_path, return_dict, args_dict):
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    gpu_id = gpu_ids[rank]
    device_str = f"cuda:{gpu_id}"
    device = torch.device(device_str)
    print(f"[Worker {rank}] GPU {gpu_id} 开始处理 {len(lines_chunk)} 条数据")

    empty_metrics = get_empty_metrics(args_dict["first_pcts"], args_dict["last_pcts"], args_dict["first_tokens"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载基础模型 AutoModel，从而剥离 lm_head
        model = AutoModel.from_pretrained(
            model_path, device_map={"": gpu_id}, torch_dtype=torch.bfloat16, 
            trust_remote_code=True, attn_implementation="eager"
        )
        model.eval()
    except Exception as e:
        print(f"[Worker {rank}] 模型加载失败: {e}")
        return_dict[rank] = [{
            "is_correct": None, "error": f"Model load error: {str(e)}", **empty_metrics
        } for _ in lines_chunk]
        return

    local_results = []
    
    for idx, line in tqdm(enumerate(lines_chunk), desc=f"GPU {gpu_id}", position=rank, total=len(lines_chunk)):
        metrics_dict, error_msg = None, None
        hidden_states = None  # 用于在两个 try 块之间传递数据
        
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            local_results.append({"is_correct": None, "error": "JSON error", **empty_metrics})
            continue
            
        # ==============================================================================
        # 解析数据：在 final_test.jsonl 中，pred 已经包含了 query 和原始生成的文本
        # ==============================================================================
        query_text = item.get(args_dict["prompt_field"], item.get("query", ""))
        generated_text = item.get(args_dict["prediction_field"], item.get("pred", ""))
        
        # 提取基础标识符
        base_res = {}
        if "id" in item:
            base_res["id"] = item["id"]
        base_res["is_correct"] = item.get("is_correct")

        if not generated_text:
            base_res.update({"error": "Empty text", **empty_metrics})
            local_results.append(base_res)
            continue

        # ==============================================================================
        # 阶段一：前向推理 (高 OOM 风险区)
        # ==============================================================================
        try:
            # 1. Tokenize prompt to find the generation boundary.
            query_tokens = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids
            query_len = query_tokens.shape[1]
            
            # 2. Reconstruct the full context when predictions only store generated text.
            if args_dict["prediction_includes_prompt"] or (query_text and generated_text.startswith(query_text)):
                full_text = generated_text
            else:
                full_text = query_text + args_dict["prompt_prediction_separator"] + generated_text
            full_input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            
            with torch.no_grad():
                # 3. 将全量文本喂给模型，确保绝对位置编码和上文逻辑完全连贯
                outputs = model(input_ids=full_input_ids, output_hidden_states=True)
                
                # 4. 【核心截取逻辑】：去除 query 对应的内部状态，仅保留生成部分的隐状态
                # 注意：从 query_len - 1 开始截取，这样我们保留了 query 的最后一个 token，
                # 它是作为生成接下来第一个回答 token 的前驱上下文，用于计算初始的 diffs/angles
                start_idx = max(0, query_len - 1)
                
                # 防止由于数据异常导致的截取越界（比如 pred 里根本没有生成出内容）
                if start_idx >= full_input_ids.shape[1] - 1:
                    hidden_states = None
                    error_msg = "Sequence too short after stripping query."
                else:
                    # 必须用 .clone() 深拷贝切片，以便下面彻底释放庞大的 outputs 显存
                    hidden_states = tuple(h[:, start_idx:, :].clone() for h in outputs.hidden_states)
                
            # 立即清理输入和输出对象，释放显存
            del query_tokens, full_input_ids, outputs
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                error_msg = "OOM in Stage 1: Forward Pass"
                tqdm.write(f"\n[Worker {rank}] {error_msg} at item {idx}. Sequence length might be too long.")
            else:
                error_msg = f"Runtime Error in Stage 1: {str(e)}"
            
            # 致命细节：强制切断异常对象 e 的引用，防止 Traceback 锁死显存
            e = None 
            if 'query_tokens' in locals(): del query_tokens
            if 'full_input_ids' in locals(): del full_input_ids
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()
            gc.collect()

        # ==============================================================================
        # 阶段二：指标计算 (如果阶段一成功)
        # ==============================================================================
        if hidden_states is not None:
            try:
                metrics_dict = calculate_scalar_metrics(
                    hidden_states,
                    args_dict["first_pcts"],
                    args_dict["last_pcts"],
                    args_dict["first_tokens"],
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    error_msg = "OOM in Stage 2: Metric Calculation"
                    tqdm.write(f"\n[Worker {rank}] {error_msg} at item {idx}.")
                else:
                    error_msg = f"Runtime Error in Stage 2: {str(e)}"
                
                # 同样强制清理异常对象
                e = None
            finally:
                # 无论计算是否成功，都必须释放庞大的 hidden_states
                del hidden_states
                torch.cuda.empty_cache()
                gc.collect()

        # ---------------- 保存结果 ----------------
        # 仅保存核心指标和标识符，不存原始长文本
        res = base_res.copy() 
        if metrics_dict is not None:
            res["error"] = None
            res.update(metrics_dict)
        else:
            res["error"] = error_msg or "Processing Failed or Sequence too short"
            res.update(empty_metrics)
            
        local_results.append(res)
            
        # 每一轮结束后的常规清理，确保完全干净的状态迎接下一条数据
        torch.cuda.empty_cache()

    return_dict[rank] = local_results
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated CUDA device ids, e.g. 0,1,2,3.")
    parser.add_argument("--prompt_field", type=str, default="query")
    parser.add_argument("--prediction_field", type=str, default="pred")
    parser.add_argument("--prediction_includes_prompt", action="store_true")
    parser.add_argument("--prompt_prediction_separator", type=str, default="\n")
    parser.add_argument("--first_pcts", type=str, default="10,20,40,80,100")
    parser.add_argument("--last_pcts", type=str, default="5,10,20")
    parser.add_argument("--first_tokens", type=str, default="2000,4000,6000,8000")
    args = parse_args_with_config(parser)
    require_args(args, ["input_file", "output_file", "model_path"])
    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError as exc:
        raise ImportError("torch is required for latent-space analysis. Install dependencies with `pip install -r requirements.txt`.") from exc

    args_dict = {
        "prompt_field": args.prompt_field,
        "prediction_field": args.prediction_field,
        "prediction_includes_prompt": args.prediction_includes_prompt,
        "prompt_prediction_separator": args.prompt_prediction_separator,
        "first_pcts": parse_int_list(args.first_pcts),
        "last_pcts": parse_int_list(args.last_pcts),
        "first_tokens": parse_int_list(args.first_tokens),
    }

    print(f"Reading input file: {args.input_file}")
    with open(args.input_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    total_input_lines = len(lines)
    print(f"Total lines to process: {total_input_lines}")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA device is available. This analysis requires GPU memory for hidden states.")

    if args.gpu_ids:
        gpu_ids = parse_int_list(args.gpu_ids)
    else:
        requested_gpus = args.num_gpus or available_gpus
        gpu_ids = list(range(min(requested_gpus, available_gpus)))

    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        raise ValueError("No GPU ids selected.")

    chunk_size = (len(lines) + num_gpus - 1) // num_gpus
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(num_gpus):
        if rank < len(chunks):
            chunk_data = chunks[rank]
        else:
            chunk_data = []
        p = mp.Process(target=worker_process, args=(rank, gpu_ids, chunk_data, args.model_path, return_dict, args_dict))
        p.start()
        processes.append(p)
    
    for p in processes: p.join()

    print(f"Saving to {args.output_file}...")
    ensure_parent_dir(args.output_file)
    total_output_lines = 0
    with open(args.output_file, 'w') as f:
        for rank in range(num_gpus):
            if rank in return_dict:
                results = return_dict[rank]
                total_output_lines += len(results)
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Done. Processed {total_output_lines}/{total_input_lines} lines.")

if __name__ == "__main__":
    main()
