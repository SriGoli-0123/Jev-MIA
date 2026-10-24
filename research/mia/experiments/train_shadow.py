#!/usr/bin/env python3
"""Train the shadow victim: theta_0 --SFT(D_M^S)--> theta_S, using NanoJev's unmodified trainer.

D_NM^S is never written to any SFT file. The checkpoint is frozen after training
(run_meta.json records its SHA-256; nothing in research/mia writes to it again).

  python research/mia/experiments/train_shadow.py --config research/mia/configs/shadow_sft.yaml \
      --data research/mia/artifacts/data/data_seed20261024 --seed 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia.experiments._sft import main_for_role  # noqa: E402

if __name__ == "__main__":
    main_for_role("shadow")
