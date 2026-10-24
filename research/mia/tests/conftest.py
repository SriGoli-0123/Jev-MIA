"""Shared fixtures. Everything here is CPU-only and needs no model download."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/ -> import mia

from mia.config import load_config  # noqa: E402
from mia.data.generate_mia_data import generate, write_dataset  # noqa: E402
from mia.paths import CONFIG_DIR  # noqa: E402

SEED = 1234


@pytest.fixture(scope="session")
def smoke_config():
    return load_config(CONFIG_DIR / "smoke" / "data.yaml")


@pytest.fixture(scope="session")
def generated(smoke_config):
    """(records, splits) for the tiny smoke pools."""
    return generate(smoke_config, SEED)


@pytest.fixture(scope="session")
def dataset_dir(smoke_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("mia") / "data"
    write_dataset(smoke_config, SEED, out)
    return out
