"""Run directories that are never overwritten, plus the provenance every run records."""
import datetime as _dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from mia.paths import MIA_ROOT, REPO_ROOT, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY


def utc_stamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info():
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    ancestor = None
    if head is not None:
        try:
            subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", UPSTREAM_COMMIT, head],
                           check=True, capture_output=True)
            ancestor = True
        except subprocess.CalledProcessError:
            ancestor = False
        except OSError:
            ancestor = None
    return {"head": head, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "dirty_paths": status.splitlines() if status else [],
            "upstream_repository": UPSTREAM_REPOSITORY, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_commit_is_ancestor_of_head": ancestor,
            "research_mia_tree_sha256": tree_sha256()}


def tree_sha256(root=MIA_ROOT):
    """Content hash of the research code itself, so uncommitted edits are still identifiable."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or rel.parts[0] == "artifacts" or "__pycache__" in rel.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt", ".ini", ".sh"}:
            continue
        digest.update(str(rel).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def package_versions():
    names = ("torch", "transformers", "safetensors", "numpy", "scipy", "scikit-learn", "pandas", "PyYAML")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def new_run_dir(artifacts_root, kind, name):
    """Create artifacts_root/kind/name; refuse to reuse an existing directory."""
    path = Path(artifacts_root).expanduser().resolve() / kind / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, obj, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(obj), ensure_ascii=False, indent=2, allow_nan=False,
                               default=_json_default) + "\n", encoding="utf-8")
    return path


def _finite(value):
    """NaN/inf (e.g. an undefined test statistic) become null instead of invalid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _finite(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return _finite(value.tolist())
    return value


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_provenance(run_dir, *, config, config_path, seed, extra=None):
    record = {
        "created_utc": utc_stamp(),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "seed": seed,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": json_sha256(config),
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": package_versions(),
        **(extra or {}),
    }
    write_json(Path(run_dir) / "provenance.json", record)
    return record
