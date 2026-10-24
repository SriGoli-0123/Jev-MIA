"""Thin bridge to upstream NanoJev: SFT invocation and probability extraction.

Training always runs the unmodified upstream entry point scripts/train_pipeline_decisions.py
(as a subprocess). Local CPU smoke tests run the same ``main()`` through
tools/upstream_sft_cpu_shim.py, which only redirects CUDA calls to CPU.

Inference reuses upstream pieces: ``local_checkpoint_files`` and the DecisionModel class
loader from predict_toy_decisions.py, ``prepare_examples`` for byte-identical tokenized
candidate paths, and ``pack_complete_questions`` for complete-question batching. The
only difference from upstream DecisionPredictor is that the device is selectable (it
hard-requires CUDA) and raw candidate logits are returned instead of rounded answers.
"""
import contextlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from mia.paths import CPU_SHIM_ENTRYPOINT, REPO_ROOT, UPSTREAM_SFT_ENTRYPOINT, ensure_upstream_on_path

TRAINER_FLAGS = {  # config key -> upstream CLI flag
    "objective": "--objective", "loss": "--loss", "set_head": "--set-head", "head_steps": "--head-steps",
    "batch_questions": "--batch-questions", "microbatch_questions": "--microbatch-questions",
    "max_microbatch_tokens": "--max-microbatch-tokens", "eval_every": "--eval-every",
    "max_length": "--max-length", "backbone_lr": "--backbone-lr", "head_lr": "--head-lr",
    "head_warmup_lr": "--head-warmup-lr", "precision": "--precision", "reward_samples": "--reward-samples",
}
TRAINER_SWITCHES = {"gradient_checkpointing": "--gradient-checkpointing",
                    "disable_native_triton": "--disable-native-triton"}


def resolve_steps(trainer, n_train_questions):
    """Explicit ``steps`` wins; otherwise convert ``epochs`` over training questions to optimizer steps."""
    if trainer.get("steps"):
        return int(trainer["steps"])
    epochs = float(trainer["epochs"])
    return max(1, math.ceil(epochs * n_train_questions / int(trainer["batch_questions"])))


def sft_command(sft_config, input_dir, output_dir, seed, n_train_questions, cpu_shim=False, init_checkpoint=None):
    trainer = sft_config["trainer"]
    entry = CPU_SHIM_ENTRYPOINT if cpu_shim else UPSTREAM_SFT_ENTRYPOINT
    cmd = [sys.executable, str(entry), "--input", str(input_dir), "--output-dir", str(output_dir),
           "--model", str(sft_config["base_model"]), "--revision", str(sft_config.get("base_revision", "main")),
           "--seed", str(seed), "--steps", str(resolve_steps(trainer, n_train_questions))]
    if init_checkpoint:
        cmd += ["--init-checkpoint", str(init_checkpoint)]
    for key, flag in TRAINER_FLAGS.items():
        if trainer.get(key) is not None:
            cmd += [flag, str(trainer[key])]
    for key, flag in TRAINER_SWITCHES.items():
        if trainer.get(key):
            cmd.append(flag)
    if cpu_shim and trainer.get("precision", "bf16") != "fp32":
        raise ValueError("The CPU smoke-test shim requires trainer.precision: fp32")
    return cmd


def run_logged(cmd, log_path, env=None):
    """Run a command from the repository root, streaming output to the console and a log file."""
    log_path = Path(log_path)
    with log_path.open("x", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, **(env or {})})
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}; see {log_path}")


def resolve_device(device):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class DecisionScorer:
    """Load a frozen NanoJev checkpoint once and return complete-question logits."""

    def __init__(self, checkpoint_dir, device="auto", precision="fp32", max_length=None,
                 microbatch_questions=32, max_microbatch_tokens=65536):
        ensure_upstream_on_path()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from predict_toy_decisions import load_decision_model_class, local_checkpoint_files, read_json

        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        self.device = torch.device(resolve_device(device))
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("bf16 autocast is only used on CUDA, matching upstream evaluation")
        self.root, paths = local_checkpoint_files(checkpoint_dir)
        self.run_config = read_json(paths["run_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                       trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True,
                                                 trust_remote_code=False)
        body_config.use_cache = False
        body = AutoModel.from_config(body_config, attn_implementation="sdpa", trust_remote_code=False).float()
        model = load_decision_model_class()(body, self.run_config["set_head"])
        weights = load_file(str(paths["weights"]), device="cpu")
        model.load_state_dict(weights, strict=True)
        del weights
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.precision = precision
        self.max_length = int(max_length or self.run_config.get("max_length", 512))
        self.microbatch_questions = microbatch_questions
        self.max_microbatch_tokens = max_microbatch_tokens
        self._torch = torch

    def score(self, states):
        """``states``: [{"id", "state", "questions"}] in the upstream request schema.

        Returns {(state_id, qid): {"type", "candidate_ids", "logits": float64 array}}.
        """
        from predict_toy_decisions import prepare_examples
        from train_pipeline_decisions import pack_complete_questions
        torch = self._torch
        if not states:
            return {}
        examples = prepare_examples({"states": states}, self.tokenizer, self.max_length)
        # Length-sorted packing only reduces padding; masks make each path independent of its batch.
        examples.sort(key=lambda ex: (max(map(len, ex["leaf_tokens"])), len(ex["leaf_tokens"])))
        out = {}
        use_autocast = self.precision == "bf16" and self.device.type == "cuda"
        with torch.inference_mode():
            for group in pack_complete_questions(examples, self.microbatch_questions, self.max_microbatch_tokens):
                autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast
                            else contextlib.nullcontext())
                with autocast:
                    logits, _ = self.model(group, self.tokenizer.pad_token_id)
                for example, values in zip(group, logits):
                    z = values[:len(example["candidate_ids"])].double().cpu().numpy()
                    if not np.isfinite(z).all():
                        raise RuntimeError(f"Non-finite logits for {example['id']}")
                    out[(example["state_id"], example["qid"])] = {
                        "type": example["type"], "candidate_ids": list(example["candidate_ids"]), "logits": z}
        return out
