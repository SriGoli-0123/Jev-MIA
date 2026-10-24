"""Locations of this research package and of the upstream NanoJev code it reuses.

Everything under research/mia imports upstream NanoJev modules from scripts/
instead of copying them. Upstream modules import each other by bare name
(for example ``from predict_toy_decisions import ...``), so scripts/ must be
on sys.path before any of them is imported.
"""
import sys
from pathlib import Path

MIA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MIA_ROOT.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = MIA_ROOT / "configs"
DEFAULT_ARTIFACTS = MIA_ROOT / "artifacts"

# The upstream snapshot this extension was written against. provenance.py checks
# that it is an ancestor of the working tree HEAD and records both.
UPSTREAM_REPOSITORY = "https://github.com/TianyuCodings/NanoJev"
UPSTREAM_COMMIT = "45b629283ddeb3b3784806f352bbf5e97f71dd40"

# Base checkpoint theta_0: the revision upstream used for its support-decision toy run
# (research/toy_results/run_manifest.json, run "base_teacher").
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

UPSTREAM_SFT_ENTRYPOINT = SCRIPTS_DIR / "train_pipeline_decisions.py"
CPU_SHIM_ENTRYPOINT = MIA_ROOT / "tools" / "upstream_sft_cpu_shim.py"


def ensure_upstream_on_path():
    """Append (not prepend) scripts/ so upstream names never shadow the stdlib."""
    path = str(SCRIPTS_DIR)
    if path not in sys.path:
        sys.path.append(path)
