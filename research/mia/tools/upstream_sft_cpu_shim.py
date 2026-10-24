#!/usr/bin/env python3
"""Run NanoJev's unmodified train_pipeline_decisions.main() on CPU. Smoke tests only.

Upstream training hard-requires CUDA. This shim makes the CUDA-only calls in that
script resolve to CPU equivalents and then calls the original ``main()`` with the
original arguments, so the training loop, loss, batching, dev selection and
checkpoint format are exactly upstream's. It refuses bf16 (use --precision fp32).
Never use it for reported experiments.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

_RealGenerator = torch.Generator


class _CPUGeneratorMeta(type):
    def __call__(cls, *args, **kwargs):
        kwargs.pop("device", None)
        return _RealGenerator(device="cpu")

    def __instancecheck__(cls, obj):
        return isinstance(obj, _RealGenerator)


class _CPUGenerator(metaclass=_CPUGeneratorMeta):
    pass


def install():
    if torch.cuda.is_available():
        raise SystemExit("CUDA is available: run scripts/train_pipeline_decisions.py directly, not the CPU shim")
    if "--precision" not in sys.argv or sys.argv[sys.argv.index("--precision") + 1] != "fp32":
        raise SystemExit("The CPU shim requires --precision fp32")
    real_is_available = torch.cuda.is_available

    def is_available():
        # Only the trainer's own CUDA gate sees True; torch internals (e.g. optimizers) still see CPU.
        return sys._getframe(1).f_globals.get("__name__") == "train_pipeline_decisions" or real_is_available()

    torch.cuda.is_available = is_available
    torch.cuda.is_bf16_supported = lambda *a, **k: False
    torch.cuda.get_device_name = lambda *a, **k: "cpu (research/mia smoke-test shim; not a reported run)"
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.manual_seed_all = lambda *a, **k: None  # torch.manual_seed already seeds the CPU generator
    torch.nn.Module.cuda = lambda self, device=None: self
    torch.Generator = _CPUGenerator


if __name__ == "__main__":
    install()
    import train_pipeline_decisions
    sys.argv[0] = str(REPO_ROOT / "scripts" / "train_pipeline_decisions.py")
    train_pipeline_decisions.main()
