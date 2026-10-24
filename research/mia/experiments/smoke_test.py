#!/usr/bin/env python3
"""End-to-end smoke test of the whole pipeline on a tiny dataset, through the real CLIs.

  --mode cpu  (laptop, minutes): tiny randomly initialized Qwen3 backbone with the real
              Qwen3-0.6B-Base tokenizer; upstream SFT runs through the CPU shim.
  --mode gpu  (CUDA node): real Qwen/Qwen3-0.6B-Base and the unmodified upstream trainer,
              a few steps only. Run this before any full-size training.

Checks, in order: tiny dataset generation; upstream schema validation; shadow/target SFT;
probability extraction parity with upstream's own saved dev predictions; member/non-member
IDs against the SFT files actually trained; confidence MIA; RIFT feasibility (2 shadow seeds);
RIFT MIA with the gate overridden (smoke only); cross-primitive (6 tiny models); paired IN/OUT.
Numbers produced here are meaningless; only the plumbing is being tested.

  python research/mia/experiments/smoke_test.py --mode cpu
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from mia.config import dump_config, load_config  # noqa: E402
from mia.data.splits import load_dataset, read_jsonl, sft_state_ids  # noqa: E402
from mia.paths import CONFIG_DIR, DEFAULT_ARTIFACTS, MIA_ROOT, REPO_ROOT  # noqa: E402
from mia.provenance import new_run_dir, utc_stamp, write_json  # noqa: E402

EXP = MIA_ROOT / "experiments"
SMOKE = CONFIG_DIR / "smoke"


def run(cmd, report, name):
    print(f"\n=== {name}\n$ {' '.join(map(str, cmd))}", flush=True)
    start = time.time()
    subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT, check=True)
    report.append({"step": name, "seconds": round(time.time() - start, 1), "status": "ok"})


def check_parity(model_run, precision, tolerance):
    """Our extractor vs upstream evaluate_pipeline's predictions_dev.jsonl for the same checkpoint."""
    from mia.nanojev_bridge import DecisionScorer
    checkpoint = Path(model_run) / "checkpoint"
    scorer = DecisionScorer(checkpoint, precision=precision)
    rows = read_jsonl(Path(model_run) / "sft" / "dev.jsonl")
    ours = scorer.score([{k: r[k] for k in ("id", "state", "questions")} for r in rows])
    worst = 0.0
    for pred in read_jsonl(checkpoint / "predictions_dev.jsonl"):
        sid, qid = pred["id"].rsplit(":", 1)
        z = ours[(sid, qid)]["logits"]
        p = np.exp(z - z.max())
        worst = max(worst, float(np.abs(p / p.sum() - np.asarray(pred["student_probs"])).max()))
    if worst > tolerance:
        raise AssertionError(f"Extraction differs from upstream predictions by {worst} > {tolerance}")
    return worst


def check_membership(model_run, data_dir):
    meta = json.loads((Path(model_run) / "run_meta.json").read_text())
    _, splits, _ = load_dataset(data_dir)
    on_disk = sft_state_ids(Path(model_run) / "sft")
    every = set().union(*map(set, on_disk.values()))
    for pool in meta["membership"]["member_pools"]:
        assert set(splits["pools"][pool]) <= set(on_disk["train"]), f"{pool} not fully trained"
    for pool in meta["membership"]["nonmember_pools"]:
        assert not set(splits["pools"][pool]) & every, f"{pool} leaked into SFT files"
    return {"run": str(model_run), "members_trained": True, "nonmembers_absent": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--stages", nargs="+", default=["core", "rift", "xp", "pio"],
                        choices=["core", "rift", "xp", "pio"])
    args = parser.parse_args()
    root = new_run_dir(args.artifacts_root, "smoke", f"{args.mode}_{utc_stamp()}")
    py, report = sys.executable, []
    common = ["--artifacts-root", root]

    if args.mode == "cpu":
        backbone = root / "tiny_qwen3"
        run([py, MIA_ROOT / "tools" / "make_tiny_backbone.py", "--out", backbone], report, "tiny backbone")
        sft = load_config(SMOKE / "sft_cpu.yaml")
        sft["base_model"], sft["base_revision"] = str(backbone), "main"
        precision, tolerance, shim = "fp32", 1e-5, ["--cpu-shim"]
    else:
        sft = load_config(SMOKE / "sft_gpu.yaml")
        precision, tolerance, shim = sft["trainer"].get("precision", "bf16"), 2e-2, []
    sft_path = root / "sft.yaml"
    dump_config(sft, sft_path)
    data = root / "data"
    run([py, MIA_ROOT / "data" / "generate_mia_data.py", "--config", SMOKE / "data.yaml", "--seed", args.seed,
         "--out", data], report, "generate tiny dataset")
    models = root / "models"

    def train(role, seed, name, variant="standard"):
        run([py, EXP / f"train_{role}.py", "--config", sft_path, "--seed", seed, "--data", data, "--variant", variant,
             "--run-name", name, *common, *shim], report, f"train {name}")
        return models / name

    if "core" in args.stages:
        s1, t1 = train("shadow", 1, "shadow_s1"), train("target", 3, "target_s3")
        worst = check_parity(s1, precision, tolerance)
        report.append({"step": "probability extraction parity", "max_abs_prob_diff": worst, "status": "ok"})
        report.append({"step": "membership ids", "status": "ok",
                       "checks": [check_membership(s1, data), check_membership(t1, data)]})
        run([py, EXP / "run_confidence_mia.py", "--config", SMOKE / "confidence_mia.yaml", "--seed", 0, "--data", data,
             "--shadow-run", s1, "--target-run", t1, "--run-name", "confidence", *common], report, "confidence MIA")
    if "rift" in args.stages:
        s1, t1 = models / "shadow_s1", models / "target_s3"
        s2 = train("shadow", 2, "shadow_s2")
        run([py, EXP / "run_rift_feasibility.py", "--config", SMOKE / "rift.yaml", "--seed", 0, "--data", data,
             "--shadow-runs", s1, s2, "--run-name", "feasibility", *common], report, "RIFT feasibility")
        run([py, EXP / "run_rift_mia.py", "--config", SMOKE / "rift.yaml", "--seed", 0, "--data", data,
             "--feasibility", root / "rift_feasibility" / "feasibility" / "verdict.json", "--shadow-run", s1,
             "--target-run", t1, "--override-feasibility-gate", "smoke test of plumbing only",
             "--run-name", "rift_mia", *common], report, "RIFT MIA (gate overridden, smoke)")
    if "xp" in args.stages:
        shadows = [train("shadow", 21 + i, f"shadow_xp_{p}", f"xp_{p}") for i, p in enumerate(("boolean", "choice", "score"))]
        targets = [train("target", 31 + i, f"target_xp_{p}", f"xp_{p}") for i, p in enumerate(("boolean", "choice", "score"))]
        run([py, EXP / "run_cross_primitive.py", "--config", SMOKE / "cross_primitive.yaml", "--seed", 0,
             "--data", data, "--shadow-runs", *shadows, "--target-runs", *targets, "--run-name", "cross_primitive",
             *common], report, "cross-primitive MIA")
    if "pio" in args.stages:
        run([py, EXP / "run_paired_in_out.py", "--config", SMOKE / "paired_in_out.yaml", "--seed", 0, "--data", data,
             "--train-seeds", 11, 12, "--sft-config", sft_path, "--run-name", "paired", *common, *shim],
            report, "paired IN/OUT")
    write_json(root / "smoke_report.json", {"mode": args.mode, "stages": args.stages, "steps": report})
    print(f"\nSMOKE TEST PASSED ({args.mode}); report: {root / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
