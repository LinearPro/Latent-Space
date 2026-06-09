import argparse
import json
from pathlib import Path


def load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return data


def parse_args_with_config(parser, argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to a JSON config file.")
    config_args, remaining = config_parser.parse_known_args(argv)

    config = load_config(config_args.config)
    if config:
        parser.set_defaults(**config)

    parser.add_argument("--config", default=config_args.config, help="Path to a JSON config file.")
    return parser.parse_args(remaining)


def require_args(args, names):
    missing = [name for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required arguments: {formatted}")


def parse_int_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]

