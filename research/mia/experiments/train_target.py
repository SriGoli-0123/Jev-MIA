#!/usr/bin/env python3
"""Train the target victim: theta_0 --SFT(D_M^T)--> theta_T, same procedure as the shadow, different data.

  python research/mia/experiments/train_target.py --config research/mia/configs/target_sft.yaml \
      --data research/mia/artifacts/data/data_seed20261024 --seed 1001
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia.experiments._sft import main_for_role  # noqa: E402

if __name__ == "__main__":
    main_for_role("target")
