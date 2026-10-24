"""GPU-only: our extractor must agree with upstream DecisionPredictor on a real trained checkpoint.

Skipped unless CUDA is available and MIA_PARITY_CHECKPOINT points to a checkpoint directory
(the ``checkpoint/`` folder of a train_shadow.py run). MIA_PARITY_DATA may point to an SFT dir.
"""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CHECKPOINT = os.environ.get("MIA_PARITY_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CHECKPOINT),
                                reason="needs CUDA and MIA_PARITY_CHECKPOINT")


def test_bf16_extraction_matches_upstream_decision_predictor():
    from mia.data.splits import read_jsonl
    from mia.nanojev_bridge import DecisionScorer
    from mia.paths import ensure_upstream_on_path
    ensure_upstream_on_path()
    from predict_toy_decisions import DecisionPredictor

    sft = Path(os.environ.get("MIA_PARITY_DATA", Path(CHECKPOINT).parent / "sft"))
    rows = read_jsonl(sft / "dev.jsonl")[:16]
    states = [{k: r[k] for k in ("id", "state", "questions")} for r in rows]
    upstream = DecisionPredictor(CHECKPOINT, precision="bf16").predict({"states": states})
    ours = DecisionScorer(CHECKPOINT, device="cuda", precision="bf16").score(states)
    worst = 0.0
    for state in upstream["states"]:
        for qid, answer in state["answers"].items():
            result = ours[(state["id"], qid)]
            p = np.exp(result["logits"] - result["logits"].max())
            p /= p.sum()
            reference = np.array([answer["probabilities"][c] for c in result["candidate_ids"]])
            worst = max(worst, float(np.abs(p - reference).max()))
    assert worst < 2e-2  # bf16; batch composition differs between the two code paths
