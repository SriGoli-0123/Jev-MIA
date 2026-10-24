#!/usr/bin/env python3
"""Create a tiny randomly initialized Qwen3 backbone with the real Qwen3-0.6B-Base tokenizer.

Used only so the full pipeline (upstream SFT -> extraction -> attacks) can be smoke-tested
on a laptop CPU in minutes. It has the Qwen3 architecture class and tokenizer of theta_0 but
none of its weights, so its numbers mean nothing scientifically.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia.paths import BASE_MODEL, BASE_REVISION  # noqa: E402


def build(out_dir, source=BASE_MODEL, revision=BASE_REVISION, seed=0, hidden=64, layers=2, heads=4, kv_heads=2):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    reference = AutoConfig.from_pretrained(source, revision=revision)
    values = reference.to_dict()
    values.pop("layer_types", None)
    values.update(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                  num_key_value_heads=kv_heads, head_dim=hidden // heads, intermediate_size=2 * hidden,
                  max_window_layers=layers)
    config = type(reference)(**values)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    info = {"purpose": "smoke test only; random weights", "tokenizer_source": source, "tokenizer_revision": revision,
            "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
            "hidden_size": hidden, "layers": layers}
    (out_dir / "tiny_backbone.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.out, seed=args.seed, hidden=args.hidden, layers=args.layers)))


if __name__ == "__main__":
    main()
