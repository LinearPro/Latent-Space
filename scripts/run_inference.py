import json
import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_reasoning.config import parse_args_with_config, require_args
from latent_reasoning.io import load_jsonl, ensure_parent_dir

DEFAULT_MODEL_NAME = "Qwen3-Thinking"
DEFAULT_PORTS = [8011, 8022]


def get_item_id(item, index):
    return item.get("id", item.get("question_id", index))


def get_prompt(item, prompt_field):
    if prompt_field in item:
        return item[prompt_field]
    for fallback in ("query", "question", "prompt", "problem"):
        if fallback in item:
            return item[fallback]
    raise KeyError(f"Cannot find prompt field. Tried {prompt_field}, query, question, prompt, problem.")


def get_output_filename(output_dir, output_prefix, part_num):
    return os.path.join(output_dir, f"{output_prefix}_part{part_num}.jsonl")


def parse_ports(value):
    if isinstance(value, list):
        return [int(p) for p in value]
    ports = [int(p.strip()) for p in value.split(",") if p.strip()]
    if not ports:
        raise argparse.ArgumentTypeError("At least one port is required.")
    return ports


def build_parser():
    parser = argparse.ArgumentParser(description="Run batched inference against one or more vLLM OpenAI-compatible servers.")
    parser.add_argument("--input-file", help="Input JSONL file. Each line should contain a prompt field.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for prediction JSONL files.")
    parser.add_argument("--output-prefix", default="predictions", help="Output file prefix.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Served model name exposed by vLLM.")
    parser.add_argument("--ports", type=parse_ports, default=DEFAULT_PORTS, help="Comma-separated vLLM ports, e.g. 8011,8022.")
    parser.add_argument("--part", type=int, default=1, help="1-based shard index to run.")
    parser.add_argument("--num-parts", type=int, default=None, help="Total number of shards. Defaults to len(--ports).")
    parser.add_argument("--prompt-field", default="query", help="JSON field containing the user prompt.")
    parser.add_argument("--reference-field", default="response", help="Optional JSON field containing the reference answer.")
    parser.add_argument("--max-tokens", type=int, default=45000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--quiet", action="store_true", help="Do not stream generations to stdout.")
    return parser

def main():
    args = parse_args_with_config(build_parser())
    require_args(args, ["input_file"])
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for inference. Install dependencies with `pip install -r requirements.txt`.") from exc

    args.ports = parse_ports(args.ports) if isinstance(args.ports, str) else args.ports
    num_parts = args.num_parts or len(args.ports)
    if args.part < 1 or args.part > num_parts:
        raise ValueError(f"--part must be in [1, {num_parts}], got {args.part}.")
    if args.part > len(args.ports):
        raise ValueError(f"--part {args.part} has no matching port in --ports={args.ports}.")

    port = args.ports[args.part - 1]
    print(f"Config: part {args.part}/{num_parts}, port {port}")
    
    api_url = f"http://{args.host}:{port}/v1"
    output_file = get_output_filename(args.output_dir, args.output_prefix, args.part)

    client = OpenAI(base_url=api_url, api_key=args.api_key, timeout=args.timeout)
    print(f"Server: {api_url}")
    print(f"Model: {args.model_name}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    all_tasks = load_jsonl(args.input_file)
    
    total_count = len(all_tasks)
    start = (args.part - 1) * total_count // num_parts
    end = args.part * total_count // num_parts
    tasks = all_tasks[start:end]
    print(f"Task range: [{start}, {end}) out of {total_count}")
    
    ensure_parent_dir(output_file)
    
    finished_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "id" in data:
                        finished_ids.add(data["id"])
                except json.JSONDecodeError:
                    pass
    print(f"Already finished: {len(finished_ids)}")

    with open(output_file, "a", encoding="utf-8") as f_out:
        for local_index, item in tqdm(list(enumerate(tasks)), desc=f"Part {args.part}"):
            item_id = get_item_id(item, start + local_index)
            if item_id in finished_ids:
                continue

            query = get_prompt(item, args.prompt_field)
            
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=True
                )
                
                generated_text = ""
                finish_reason = None
                
                if not args.quiet:
                    print(f"\n\n[ID: {item_id}] Generating:", flush=True)
                    print("-" * 50)
                
                for chunk in response:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        content = delta.content
                        generated_text += content
                        if not args.quiet:
                            print(content, end="", flush=True)
                    
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                
                if not args.quiet:
                    print("\n" + "-" * 50)

                result = {
                    "id": item_id,
                    "query": query,
                    "response": item.get(args.reference_field, ""),
                    "pred": generated_text,
                    "finish_reason": finish_reason if finish_reason else "stop"
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\nID {item_id} failed: {e}")

    print(f"\nPart {args.part} finished. Output: {output_file}")

if __name__ == "__main__":
    main()
